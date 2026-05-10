import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _layer_norm_param_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
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

    mean = tl.sum(x, axis=1, keep_dims=True) / N
    var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean, mask=row_mask)
    tl.store(Rstd + rows, _rstd, mask=row_mask)

    x_hat = (x - mean) * rstd
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    x_hat = x_hat * w + b

    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask)


def triton_layernorm_param_forward(x, w, b, eps):
    [x], batched, batch_size = flatten_if_batched(x)

    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _layer_norm_param_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        x,
        y,
        w,
        b,
        mean,
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


@triton.jit
def _layer_norm_noparam_fwd_fused(
    X,
    Y,
    Mean,
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

    mean = tl.sum(x, axis=1, keep_dims=True) / N
    var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean, mask=row_mask)
    tl.store(Rstd + rows, _rstd, mask=row_mask)

    x_hat = (x - mean) * rstd
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask)


def triton_layernorm_noparam_forward(x, eps):
    if not x.is_contiguous():
        x = x.contiguous()

    [x], batched, batch_size = flatten_if_batched(x)

    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _layer_norm_noparam_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        x,
        y,
        mean,
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


def triton_layernorm_forward(x, w, b, eps, elementwise_affine=True):
    if not x.is_contiguous():
        x = x.contiguous()

    if elementwise_affine:
        assert w is not None and b is not None
        return triton_layernorm_param_forward(x, w, b, eps)
    assert w is None and b is None
    return triton_layernorm_noparam_forward(x, eps)
