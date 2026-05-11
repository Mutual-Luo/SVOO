import os
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "", "false", "no", "off")


@triton.jit
def _fill_variable_block_kv_indices_kernel(
    base_ptr,
    lengths_ptr,
    starts_ptr,
    out_ptr,
    num_segments,
    BLOCK_M: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)

    seg_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    valid_seg = seg_offsets < num_segments

    lengths = tl.load(lengths_ptr + seg_offsets, mask=valid_seg, other=0)
    starts = tl.load(starts_ptr + seg_offsets, mask=valid_seg, other=0)
    bases = tl.load(base_ptr + seg_offsets, mask=valid_seg, other=0)

    mask = valid_seg[:, None] & (token_offsets[None, :] < lengths[:, None])
    out_offsets = starts[:, None] + token_offsets[None, :]
    values = bases[:, None] + token_offsets[None, :]
    tl.store(out_ptr + out_offsets, values, mask=mask)


def _block_mask_map_to_expanded_indices(
    block_mask_map: torch.Tensor,
    block_col_sz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = block_mask_map.device
    dtype_i = torch.int32

    row_lengths = (
        block_mask_map.to(dtype_i) * block_col_sz[:, None, :].to(dtype_i)
    ).sum(-1, dtype=dtype_i)
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=device),
            torch.cumsum(row_lengths.flatten(), 0, dtype=dtype_i),
        ],
        dim=0,
    )

    col_offset = (
        torch.cumsum(block_col_sz.to(dtype_i), 1, dtype=dtype_i)
        - block_col_sz.to(dtype_i)
    )
    head_len = block_col_sz.sum(1, dtype=dtype_i)
    head_offset = torch.cumsum(head_len, 0, dtype=dtype_i) - head_len

    h_idx, _, c_idx = block_mask_map.nonzero(as_tuple=True)
    lengths = block_col_sz[h_idx, c_idx].to(dtype_i)
    base = head_offset[h_idx] + col_offset[h_idx, c_idx]

    if lengths.numel() == 0:
        kv_indices = torch.empty((0,), dtype=dtype_i, device=device)
        return kv_indptr, kv_indices

    starts = torch.cumsum(lengths, 0, dtype=dtype_i) - lengths
    total = int(kv_indptr[-1].item())
    kv_indices = torch.empty((total,), dtype=dtype_i, device=device)
    if total > 0:
        block_m = 16
        block_l = 128
        grid = (
            triton.cdiv(lengths.numel(), block_m),
            triton.cdiv(int(lengths.max().item()), block_l),
        )
        _fill_variable_block_kv_indices_kernel[grid](
            base,
            lengths,
            starts,
            kv_indices,
            lengths.numel(),
            BLOCK_M=block_m,
            BLOCK_L=block_l,
            num_warps=8,
        )

    return kv_indptr, kv_indices


def _memory_efficient_plan(
    self,
    flashinfer_sparse,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = False,
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
) -> None:
    q_data_type = flashinfer_sparse.canonicalize_torch_dtype(q_data_type)
    if kv_data_type is None:
        kv_data_type = q_data_type
    kv_data_type = flashinfer_sparse.canonicalize_torch_dtype(kv_data_type)
    self._o_dtype = q_data_type

    if logits_soft_cap is None:
        logits_soft_cap = 0.0

    num_blocks_row = block_row_sz.shape[-1]
    num_blocks_col = block_col_sz.shape[-1]

    qo_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
            torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)
    last_block_len = torch.full(
        (num_blocks_row * num_kv_heads,),
        1,
        dtype=torch.int32,
        device=block_mask_map.device,
    )

    kv_indptr, kv_indices = _block_mask_map_to_expanded_indices(
        block_mask_map, block_col_sz
    )
    kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)
    kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking)

    self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
    self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
    self._paged_kv_indices_buf = kv_indices.to(self.device, non_blocking=non_blocking)
    self._paged_kv_last_page_len = last_block_len.to(
        self.device, non_blocking=non_blocking
    )
    torch.cuda.synchronize()
    self._mask_mode = (
        flashinfer_sparse.MaskMode.CAUSAL.value
        if causal
        else flashinfer_sparse.MaskMode.NON_CAUSAL.value
    )

    assert num_qo_heads % num_kv_heads == 0, (
        "num_qo_heads must be a multiple of num_kv_heads"
    )
    assert num_blocks_row * num_kv_heads + 1 == kv_indptr_host.shape[0]
    assert kv_indptr_host[-1].item() == kv_indices_host.shape[0], (
        f"{kv_indptr_host[-1].item()} != {kv_indices_host.shape[0]}"
    )
    assert num_kv_heads == block_mask_map.shape[0]
    assert num_kv_heads == block_row_sz.shape[0]
    assert num_kv_heads == block_col_sz.shape[0]
    assert num_blocks_row == block_mask_map.shape[1]
    assert num_blocks_col == block_mask_map.shape[2]

    if self._backend == "auto":
        self._backend = flashinfer_sparse.determine_attention_backend(
            self.device,
            flashinfer_sparse.PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            self._mask_mode == flashinfer_sparse.MaskMode.CUSTOM.value,
            q_data_type,
            kv_data_type,
        )

    get_module_args = (
        q_data_type,
        kv_data_type,
        self._o_dtype,
        kv_indptr_host.dtype,
        head_dim,
        head_dim,
        flashinfer_sparse.PosEncodingMode[pos_encoding_mode].value,
        False,
        logits_soft_cap > 0,
        use_fp16_qk_reduction,
    )
    self._cached_module = flashinfer_sparse.get_batch_prefill_module(
        self._backend, *get_module_args
    )

    kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]
    required_size = len(kv_lens_arr_host)
    if required_size > self._kv_lens_buffer.shape[0]:
        self._kv_lens_buffer = torch.empty(
            (required_size,), dtype=torch.int32, device=self.device
        )
    self._kv_lens_buffer[:required_size].copy_(kv_lens_arr_host)

    args = [
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        kv_lens_arr_host,
        qo_indptr_host[-1].item(),
        num_blocks_row * num_kv_heads,
        num_qo_heads // num_kv_heads,
        1,
        1,
        False,
        head_dim,
        head_dim,
        causal,
        -1,
    ]
    if self._backend == "fa2":
        args.append(-1)
        args.append(False)
        args.append(0)
    self._plan_info = self._cached_module.plan(*args)

    self._pos_encoding_mode = pos_encoding_mode
    self._use_fp16_qk_reduction = use_fp16_qk_reduction
    self._logits_soft_cap = logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta
    self._num_kv_heads = num_kv_heads
    self._gqa_group_size = num_qo_heads // num_kv_heads


def _patch_variable_block_sparse_wrapper_class(flashinfer_sparse) -> None:
    cls = flashinfer_sparse.VariableBlockSparseAttentionWrapper
    if getattr(cls, "_svoo_memory_efficient_plan_patched", False):
        return

    original_plan = cls.plan

    def plan(self, *plan_args, **plan_kwargs):
        if _env_flag("SVOO_FLASHINFER_MEM_EFFICIENT_PLAN", default=True):
            return _memory_efficient_plan(
                self, flashinfer_sparse, *plan_args, **plan_kwargs
            )
        return original_plan(self, *plan_args, **plan_kwargs)

    cls._svoo_original_plan = original_plan
    cls.plan = plan
    cls._svoo_memory_efficient_plan_patched = True


def make_variable_block_sparse_attention_wrapper(
    flashinfer_sparse,
    *args,
    **kwargs,
):
    _patch_variable_block_sparse_wrapper_class(flashinfer_sparse)
    return flashinfer_sparse.VariableBlockSparseAttentionWrapper(*args, **kwargs)
