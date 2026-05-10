import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _modulate_shift_fwd_fused(
    X,
    Y,
    SCALE,
    SHIFT,
    x_stride,
    y_stride,
    M,
    N: tl.constexpr,
    N2: tl.constexpr,
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

    scale = tl.load(SCALE + cols, mask=col_mask, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = x * (1 + scale) + shift

    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask)


def triton_modulate_shift_forward(x, scale, shift, output_dtype=torch.float32):
    """Compute x * (1 + scale) + shift."""
    if scale is not None and torch.is_tensor(scale) and scale.dim() == 2:
        return triton_modulate_shift_batched_forward(x, scale, shift, output_dtype=output_dtype)

    [x], batched, batch_size = flatten_if_batched(x)

    M, N = x.shape
    y = torch.empty_like(x, dtype=output_dtype)
    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _modulate_shift_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        x,
        y,
        scale,
        shift,
        x.stride(0),
        y.stride(0),
        M,
        N,
        N2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    if batched:
        y = y.reshape(batch_size, -1, y.shape[-1])

    return y


@triton.jit
def _modulate_gate_residual_fwd_fused(
    R,
    X,
    Y,
    GATE,
    r_stride,
    x_stride,
    y_stride,
    M,
    N: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    r_ptr = R + rows[:, None] * r_stride + cols[None, :]
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    r = tl.load(r_ptr, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    gate = tl.load(GATE + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = r + x * gate

    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask)


def triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=torch.float32):
    """Compute residual + x * gate."""
    if gate is not None and torch.is_tensor(gate) and gate.dim() == 2:
        return triton_modulate_gate_residual_batched_forward(residual, x, gate, output_dtype=output_dtype)

    [residual, x], batched, batch_size = flatten_if_batched(residual, x)

    M, N = x.shape
    y = torch.empty_like(x, dtype=output_dtype)
    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _modulate_gate_residual_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        residual,
        x,
        y,
        gate,
        residual.stride(0),
        x.stride(0),
        y.stride(0),
        M,
        N,
        N2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    if batched:
        y = y.reshape(batch_size, -1, y.shape[-1])

    return y


@triton.jit
def _modulate_shift_batched_fwd_fused(
    X,
    Y,
    SCALE,
    SHIFT,
    x_stride,
    y_stride,
    scale_stride_b,
    shift_stride_b,
    M,
    L,
    N: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    b = rows // L

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    scale_ptr = SCALE + b[:, None] * scale_stride_b + cols[None, :]
    shift_ptr = SHIFT + b[:, None] * shift_stride_b + cols[None, :]
    scale = tl.load(scale_ptr, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr, mask=mask, other=0.0).to(tl.float32)

    y = x * (1 + scale) + shift
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask)


def triton_modulate_shift_batched_forward(x, scale, shift, output_dtype=torch.float32):
    """Compute per-sample x * (1 + scale) + shift for x [B, L, D]."""
    if x.dim() != 3:
        raise ValueError(f"Expected x to be 3D [B, L, D], got shape {tuple(x.shape)}")
    if scale.dim() != 2 or shift.dim() != 2:
        raise ValueError(
            f"Expected scale/shift to be 2D [B, D], got scale {tuple(scale.shape)} shift {tuple(shift.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not shift.is_contiguous():
        shift = shift.contiguous()

    B, L, D = x.shape
    if scale.shape != (B, D) or shift.shape != (B, D):
        raise ValueError(
            f"Shape mismatch: x {tuple(x.shape)}, scale {tuple(scale.shape)}, shift {tuple(shift.shape)}"
        )

    x2 = x.view(B * L, D)
    M, N = x2.shape
    y2 = torch.empty_like(x2, dtype=output_dtype)

    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _modulate_shift_batched_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        x2,
        y2,
        scale,
        shift,
        x2.stride(0),
        y2.stride(0),
        scale.stride(0),
        shift.stride(0),
        M,
        L,
        N,
        N2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    return y2.view(B, L, D)


@triton.jit
def _modulate_gate_residual_batched_fwd_fused(
    R,
    X,
    Y,
    GATE,
    r_stride,
    x_stride,
    y_stride,
    gate_stride_b,
    M,
    L,
    N: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    b = rows // L

    r_ptr = R + rows[:, None] * r_stride + cols[None, :]
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    r = tl.load(r_ptr, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    gate_ptr = GATE + b[:, None] * gate_stride_b + cols[None, :]
    gate = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)

    y = r + x * gate
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask)


def triton_modulate_gate_residual_batched_forward(residual, x, gate, output_dtype=torch.float32):
    """Compute per-sample residual + x * gate for x [B, L, D]."""
    if residual.dim() != 3 or x.dim() != 3:
        raise ValueError(
            f"Expected residual/x to be 3D [B, L, D], got residual {tuple(residual.shape)} x {tuple(x.shape)}"
        )
    if gate.dim() != 2:
        raise ValueError(f"Expected gate to be 2D [B, D], got {tuple(gate.shape)}")

    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()

    B, L, D = x.shape
    if residual.shape != (B, L, D) or gate.shape != (B, D):
        raise ValueError(
            f"Shape mismatch: residual {tuple(residual.shape)}, x {tuple(x.shape)}, gate {tuple(gate.shape)}"
        )

    r2 = residual.view(B * L, D)
    x2 = x.view(B * L, D)
    M, N = x2.shape
    y2 = torch.empty_like(x2, dtype=output_dtype)

    num_warps = 8
    N2 = triton.next_power_of_2(N)
    BLOCK_M = 32 if N <= 512 else 1

    _modulate_gate_residual_batched_fwd_fused[(triton.cdiv(M, BLOCK_M),)](
        r2,
        x2,
        y2,
        gate,
        r2.stride(0),
        x2.stride(0),
        y2.stride(0),
        gate.stride(0),
        M,
        L,
        N,
        N2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    return y2.view(B, L, D)
