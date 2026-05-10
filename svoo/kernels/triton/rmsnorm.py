import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    Rstd,
    x_stride,
    y_stride,
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

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.reshape(rstd, (BLOCK_M, 1))

    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask)


def triton_rmsnorm_forward(x, w, eps):
    """Apply RMSNorm over the last dimension."""
    assert x.is_contiguous(), "Input must be contiguous"
    [x], batched, batch_size = flatten_if_batched(x)

    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _rms_norm_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        x,
        y,
        w,
        rstd,
        x.stride(0),
        y.stride(0),
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
