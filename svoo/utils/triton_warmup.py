import os
import time
from typing import Iterable

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "", "false", "no", "off")


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
):
    """Precompile Triton kernels used by SVOO before the tqdm inference loop."""
    if not _env_flag("SVOO_TRITON_WARMUP", default=True):
        return

    device = _as_cuda_device(device)
    if device.type != "cuda":
        return

    strict = _env_flag("SVOO_TRITON_WARMUP_STRICT", default=False)
    start = time.time()
    print(f"[SVOO] Precompiling Triton kernels for {model_name}...", flush=True)

    try:
        if include_wan_block_kernels:
            _warmup_wan_block_kernels(hidden_dim, dtype, device)

        if include_rmsnorm:
            _warmup_rmsnorm(hidden_dim, seq_len, dtype, device)

        _warmup_permute(num_head, head_dim, seq_len, dtype, device, inverse_seq_len)

        if _valid_centroids(num_q_centroids, num_k_centroids):
            _warmup_cocluster(
                num_head,
                head_dim,
                seq_len,
                num_q_centroids,
                num_k_centroids,
                dtype,
                device,
                cfg_values,
            )

        _cleanup(device)
        elapsed = time.time() - start
        print(f"[SVOO] Triton warmup finished in {elapsed:.1f}s.", flush=True)
    except Exception as exc:
        _cleanup(device)
        message = (
            "[SVOO] Triton warmup failed; inference may compile kernels inside "
            f"the progress bar. Set SVOO_TRITON_WARMUP_STRICT=1 to fail fast. "
            f"{type(exc).__name__}: {exc}"
        )
        if strict:
            raise RuntimeError(message) from exc
        print(message, flush=True)
