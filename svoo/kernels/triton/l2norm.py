import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _l2_norm_fwd_fused(
    X,
    Y,
    x_stride_m,
    x_stride_n,
    y_stride_m,
    y_stride_n,
    M,
    N: tl.constexpr,
    N2: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Keep offset math in int64 because large row counts can overflow int32.
    rows_i64 = rows.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    x_ptr = X + rows_i64[:, None] * x_stride_m + cols_i64[None, :] * x_stride_n
    y_ptr = Y + rows_i64[:, None] * y_stride_m + cols_i64[None, :] * y_stride_n

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    l2_norm_sq = tl.sum(x * x, axis=1, keep_dims=True)
    l2_norm = tl.sqrt(l2_norm_sq + eps)
    x_norm = x / l2_norm

    x_norm = x_norm.to(Y.type.element_ty)
    tl.store(y_ptr, x_norm, mask=mask)


def triton_l2norm_forward(x, eps=1e-8):
    """Normalize the last dimension with a Triton kernel."""
    assert x.is_cuda, "Input must be on CUDA"

    [x_flat], batched, batch_size = flatten_if_batched(x)
    M, N = x_flat.shape
    y = torch.empty_like(x_flat, dtype=torch.float32)

    num_warps = 4 if N <= 256 else 8
    N2 = triton.next_power_of_2(N)

    if N <= 256:
        BLOCK_M = 64
    elif N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 16

    _l2_norm_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        x_flat,
        y,
        x_flat.stride(0),
        x_flat.stride(1),
        y.stride(0),
        y.stride(1),
        M,
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    if batched:
        y = y.reshape(batch_size, -1, y.shape[-1])

    return y
