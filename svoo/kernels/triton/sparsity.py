import os

import torch
import triton
import triton.language as tl


@triton.jit
def _counts_from_sorted_probabilities_kernel(
    sorted_weights,
    counts,
    lower_margins,
    upper_margins,
    n_cols: tl.constexpr,
    threshold: tl.constexpr,
    block_n: tl.constexpr,
    return_margins: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_n)
    row_start = row * n_cols

    total = tl.full((), 0.0, tl.float32)
    start = 0
    while start < n_cols:
        cols = start + offs
        mask = cols < n_cols
        values = tl.load(sorted_weights + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(values, axis=0)
        start += block_n

    target = total * threshold
    prefix = tl.full((), 0.0, tl.float32)
    count = tl.full((), n_cols, tl.int32)
    found = tl.full((), False, tl.int1)
    lower_margin = tl.full((), float("inf"), tl.float32)
    upper_margin = tl.full((), float("inf"), tl.float32)

    start = 0
    while start < n_cols:
        cols = start + offs
        mask = cols < n_cols
        values = tl.load(sorted_weights + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        block_cumsum = tl.cumsum(values, axis=0) + prefix
        hit = (block_cumsum >= target) & mask
        hit_any = tl.sum(hit.to(tl.int32), axis=0) > 0
        local_count = tl.min(tl.where(hit, offs + 1, block_n), axis=0)
        local_idx = local_count - 1
        hit_prefix = tl.sum(tl.where(offs == local_idx, block_cumsum, 0.0), axis=0)
        hit_weight = tl.sum(tl.where(offs == local_idx, values, 0.0), axis=0)
        update = hit_any & (~found)
        count = tl.where(update, start + local_count, count)
        lower_margin = tl.where(update, target - (hit_prefix - hit_weight), lower_margin)
        upper_margin = tl.where(update, hit_prefix - target, upper_margin)
        found = found | hit_any
        prefix += tl.sum(values, axis=0)
        start += block_n

    tl.store(counts + row, count)
    if return_margins:
        tl.store(lower_margins + row, lower_margin)
        tl.store(upper_margins + row, upper_margin)


def counts_from_sorted_probabilities_triton(
    sorted_weights: torch.Tensor,
    threshold: float,
    *,
    return_margins: bool = False,
):
    if not sorted_weights.is_cuda:
        raise ValueError("sorted_weights must be a CUDA tensor")
    if sorted_weights.size(-1) <= 0:
        raise ValueError("sorted_weights last dimension must be non-empty")

    original_shape = sorted_weights.shape[:-1]
    n_cols = sorted_weights.size(-1)
    weights_2d = sorted_weights.contiguous().reshape(-1, n_cols)
    counts = torch.empty(weights_2d.size(0), device=sorted_weights.device, dtype=torch.int32)
    lower_margins = torch.empty(weights_2d.size(0), device=sorted_weights.device, dtype=torch.float32)
    upper_margins = torch.empty(weights_2d.size(0), device=sorted_weights.device, dtype=torch.float32)

    try:
        block_n = int(os.environ.get("SVOO_SPARSITY_TRITON_BLOCK_N", "1024"))
    except ValueError:
        block_n = 1024
    block_n = min(max(128, triton.next_power_of_2(block_n)), 4096)

    _counts_from_sorted_probabilities_kernel[(weights_2d.size(0),)](
        weights_2d,
        counts,
        lower_margins,
        upper_margins,
        n_cols,
        float(threshold),
        block_n,
        bool(return_margins),
        num_warps=8,
    )
    if return_margins:
        return (
            counts.reshape(original_shape),
            lower_margins.reshape(original_shape),
            upper_margins.reshape(original_shape),
        )
    return counts.reshape(original_shape)
