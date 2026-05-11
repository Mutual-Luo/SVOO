import os
import time
from contextlib import contextmanager
from typing import Iterable

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _warmup_mode() -> str:
    return os.environ.get("SVOO_TRITON_WARMUP_MODE", "compile").strip().lower()


def _effective_seq_len(seq_len: int) -> int:
    mode = _warmup_mode()
    if mode in ("full", "profile", "benchmark"):
        return int(seq_len)
    return min(int(seq_len), max(1, _env_int("SVOO_TRITON_WARMUP_SEQ_LEN", 4096)))


def _as_cuda_device(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda":
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return device


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


@contextmanager
def _preserve_rng_state(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _valid_centroids(num_q_centroids, num_k_centroids) -> bool:
    return (
        num_q_centroids is not None
        and num_k_centroids is not None
        and int(num_q_centroids) > 0
        and int(num_k_centroids) > 0
    )


def _warmup_wan_block_kernels(hidden_dim: int, dtype: torch.dtype, device: torch.device):
    from ..kernels.triton.layernorm import triton_layernorm_forward
    from ..kernels.triton.modulate import (
        triton_modulate_gate_residual_forward,
        triton_modulate_shift_forward,
    )

    x = torch.zeros(1, 1, hidden_dim, dtype=dtype, device=device).contiguous()
    w = torch.ones(hidden_dim, dtype=dtype, device=device)
    b = torch.zeros(hidden_dim, dtype=dtype, device=device)

    _ = triton_layernorm_forward(x, w, b, 1e-6, elementwise_affine=True)

    scale_1d = torch.zeros(1, 1, hidden_dim, dtype=torch.float32, device=device)
    shift_1d = torch.zeros(1, 1, hidden_dim, dtype=torch.float32, device=device)
    gate_1d = torch.zeros(1, 1, hidden_dim, dtype=torch.float32, device=device)
    y = triton_modulate_shift_forward(x, scale_1d, shift_1d, output_dtype=dtype)
    _ = triton_modulate_gate_residual_forward(x, y, gate_1d, output_dtype=dtype)

    scale_b = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    shift_b = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    gate_b = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    y = triton_modulate_shift_forward(x, scale_b, shift_b, output_dtype=dtype)
    _ = triton_modulate_gate_residual_forward(x, y, gate_b, output_dtype=dtype)

    _sync(device)


def _warmup_rmsnorm(hidden_dim: int, seq_len: int, dtype: torch.dtype, device: torch.device):
    from ..kernels.triton.rmsnorm import triton_rmsnorm_forward

    x = torch.zeros(1, seq_len, hidden_dim, dtype=dtype, device=device).contiguous()
    w = torch.ones(hidden_dim, dtype=dtype, device=device)
    _ = triton_rmsnorm_forward(x, w, 1e-6)
    _sync(device)


def _warmup_permute(
    num_head: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    inverse_seq_len: int | None = None,
):
    from ..kernels.triton.permute import (
        apply_inverse_permutation_triton,
        permute_tensor_by_labels_triton,
    )

    x = torch.zeros(1, num_head, seq_len, head_dim, dtype=dtype, device=device)
    labels = torch.zeros(num_head, seq_len, dtype=torch.int64, device=device)
    x_perm, sorted_idx = permute_tensor_by_labels_triton(x, labels, dim=2)
    _ = apply_inverse_permutation_triton(x_perm, sorted_idx, dim=2)
    del x, labels, x_perm, sorted_idx

    if inverse_seq_len is not None and inverse_seq_len != seq_len:
        x_full = torch.zeros(1, num_head, inverse_seq_len, head_dim, dtype=dtype, device=device)
        sorted_full = (
            torch.arange(inverse_seq_len, dtype=torch.int32, device=device)
            .expand(num_head, -1)
            .contiguous()
        )
        _ = apply_inverse_permutation_triton(x_full, sorted_full, dim=2)
        del x_full, sorted_full

    _sync(device)


def _warmup_cocluster(
    num_head: int,
    head_dim: int,
    seq_len: int,
    num_q_centroids: int,
    num_k_centroids: int,
    dtype: torch.dtype,
    device: torch.device,
    cfg_values: Iterable[int],
):
    from ..co_clustering import co_cluster_tokens

    for cfg in cfg_values:
        batch_heads = int(cfg) * num_head
        q = torch.zeros(batch_heads, seq_len, head_dim, dtype=dtype, device=device)
        k = torch.zeros(batch_heads, seq_len, head_dim, dtype=dtype, device=device)
        _ = co_cluster_tokens(
            q,
            k,
            int(num_q_centroids),
            int(num_k_centroids),
            max_iters=1,
        )
        _sync(device)
        del q, k, _


def _split_sizes(total: int, blocks: int, num_head: int, device: torch.device) -> torch.Tensor:
    base = total // blocks
    remainder = total % blocks
    sizes = torch.full((1, num_head, blocks), base, dtype=torch.long, device=device)
    if remainder:
        sizes[:, :, :remainder] += 1
    return sizes


def _warmup_flashinfer_sparse(
    num_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    if not _env_flag("SVOO_FLASHINFER_WARMUP", default=True):
        return

    from ..co_clustering import dynamic_block_sparse_fwd_flashinfer

    seq_len = max(16, _env_int("SVOO_FLASHINFER_WARMUP_SEQ_LEN", 128))
    num_blocks = max(1, min(seq_len, _env_int("SVOO_FLASHINFER_WARMUP_BLOCKS", 4)))

    q = torch.zeros(1, num_head, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    block_mask = torch.ones(1, num_head, num_blocks, num_blocks, dtype=torch.bool, device=device)
    block_sizes = _split_sizes(seq_len, num_blocks, num_head, device)

    _ = dynamic_block_sparse_fwd_flashinfer(
        q,
        k,
        v,
        block_mask,
        block_sizes,
        block_sizes,
        is_cpu=False,
    )
    _sync(device)
    del q, k, v, block_mask, block_sizes, _


def warmup_svoo_triton_kernels(
    *,
    model_name: str,
    num_head: int,
    head_dim: int,
    hidden_dim: int,
    seq_len: int,
    num_q_centroids,
    num_k_centroids,
    dtype: torch.dtype,
    device,
    cfg_values=(1,),
    inverse_seq_len: int | None = None,
    include_rmsnorm: bool = False,
    include_wan_block_kernels: bool = False,
    include_flashinfer_sparse: bool = False,
):
    """Precompile SVOO kernels before the tqdm inference loop.

    The default `compile` mode intentionally uses a smaller sequence length.
    SVOO kernels avoid specialising on sequence length, so this keeps compile
    cost outside the progress bar without running a full-resolution clustering
    pass before every inference.
    """
    if not _env_flag("SVOO_TRITON_WARMUP", default=True):
        return

    device = _as_cuda_device(device)
    if device.type != "cuda":
        return

    strict = _env_flag("SVOO_TRITON_WARMUP_STRICT", default=False)
    start = time.time()
    warmup_seq_len = _effective_seq_len(seq_len)
    warmup_inverse_seq_len = None
    if inverse_seq_len is not None:
        warmup_inverse_seq_len = (
            inverse_seq_len
            if _warmup_mode() in ("full", "profile", "benchmark")
            else min(int(inverse_seq_len), warmup_seq_len + max(0, int(inverse_seq_len) - int(seq_len)))
        )
    print(
        f"[SVOO] Precompiling kernels for {model_name} "
        f"(mode={_warmup_mode()}, seq_len={warmup_seq_len}/{seq_len})...",
        flush=True,
    )

    try:
        with _preserve_rng_state(device):
            if include_wan_block_kernels:
                print(f"[SVOO]   warmup block kernels: hidden_dim={hidden_dim}", flush=True)
                _warmup_wan_block_kernels(hidden_dim, dtype, device)

            if include_rmsnorm:
                print(f"[SVOO]   warmup rmsnorm: seq_len={warmup_seq_len}, hidden_dim={hidden_dim}", flush=True)
                _warmup_rmsnorm(hidden_dim, warmup_seq_len, dtype, device)

            print(f"[SVOO]   warmup permute: heads={num_head}, seq_len={warmup_seq_len}, head_dim={head_dim}", flush=True)
            _warmup_permute(num_head, head_dim, warmup_seq_len, dtype, device, warmup_inverse_seq_len)

            if _valid_centroids(num_q_centroids, num_k_centroids):
                cfg_text = ",".join(str(int(cfg)) for cfg in cfg_values)
                print(
                    "[SVOO]   warmup co-cluster: "
                    f"cfg={cfg_text}, heads={num_head}, seq_len={warmup_seq_len}, "
                    f"head_dim={head_dim}, q_centroids={num_q_centroids}, "
                    f"k_centroids={num_k_centroids}",
                    flush=True,
                )
                _warmup_cocluster(
                    num_head,
                    head_dim,
                    warmup_seq_len,
                    num_q_centroids,
                    num_k_centroids,
                    dtype,
                    device,
                    cfg_values,
                )

            if include_flashinfer_sparse:
                print(
                    f"[SVOO]   warmup FlashInfer sparse: heads={num_head}, head_dim={head_dim}",
                    flush=True,
                )
                _warmup_flashinfer_sparse(num_head, head_dim, dtype, device)

        _cleanup(device)
        elapsed = time.time() - start
        print(f"[SVOO] Kernel warmup finished in {elapsed:.1f}s.", flush=True)
    except Exception as exc:
        _cleanup(device)
        message = (
            "[SVOO] Kernel warmup failed; inference may compile kernels inside "
            f"the progress bar. Set SVOO_TRITON_WARMUP_STRICT=1 to fail fast. "
            f"{type(exc).__name__}: {exc}"
        )
        if strict:
            raise RuntimeError(message) from exc
        print(message, flush=True)
