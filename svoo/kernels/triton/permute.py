from typing import Optional

import torch
import triton
import triton.language as tl



@triton.jit
def _permute_kernel(
    X_ptr,
    IDX_ptr,
    Y_ptr,
    S,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S

    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_row_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)

    d_offsets = tl.arange(0, D)

    src_ptrs = X_ptr + (pid_bh * S + src_row_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + s_offsets[:, None])     * D + d_offsets[None, :]

    full_mask = token_mask[:, None]

    values = tl.load(src_ptrs, mask=full_mask, other=0.0)
    tl.store(dst_ptrs, values, mask=full_mask)


@triton.jit
def _inverse_permute_kernel(
    X_ptr,
    IDX_ptr,
    Y_ptr,
    S,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S

    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_pos_idx = s_offsets.to(tl.int32)
    dst_pos_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)

    d_offsets = tl.arange(0, D)

    src_ptrs = X_ptr + (pid_bh * S + src_pos_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + dst_pos_idx[:, None]) * D + d_offsets[None, :]

    full_mask = token_mask[:, None]

    values = tl.load(src_ptrs, mask=full_mask, other=0.0)
    tl.store(dst_ptrs, values, mask=full_mask)


def permute_tensor_by_labels_triton(
    tensor: torch.Tensor,
    labels: Optional[torch.Tensor],
    dim: int,
    *,
    sorted_indices: Optional[torch.Tensor] = None,
):
    """
    Permute a CUDA tensor of shape [B, H, S, D] along the sequence dimension.
    """
    assert dim == 2, "permute_tensor_by_labels currently only supports dim==2 (sequence dimension)"
    assert tensor.dim() == 4, "Expected tensor shape [B,H,S,D]"
    assert tensor.is_cuda, "permute_tensor_by_labels requires CUDA tensors"
    
    B, H, S, D = tensor.shape
    BH = B * H

    if sorted_indices is not None:
        sorted_indices = sorted_indices.to(torch.int32).contiguous()
    else:
        assert labels is not None, "Either `labels` or `sorted_indices` must be provided."
        labels = labels.to(tensor.device)
        sorted_indices = torch.argsort(labels, dim=-1).to(torch.int32).contiguous()

    inp_flat = tensor.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(inp_flat)

    BLOCK_S = 64
    n_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_s_tiles)

    _permute_kernel[grid](inp_flat, sorted_indices, out_flat, S, D, BLOCK_S, num_warps=4)

    permuted_tensor = out_flat.reshape(B, H, S, D)
    return permuted_tensor, sorted_indices


def apply_inverse_permutation_triton(
    permuted_tensor: torch.Tensor,
    sorted_indices: torch.Tensor,
    dim: int,
):
    """Undo `permute_tensor_by_labels_triton` for tensors of shape [B, H, S, D]."""
    assert dim == 2, "apply_inverse_permutation currently only supports dim==2"
    assert permuted_tensor.dim() == 4, "Expected tensor shape [B,H,S,D]"
    assert permuted_tensor.is_cuda, "apply_inverse_permutation requires CUDA tensors"

    B, H, S, D = permuted_tensor.shape
    BH = B * H

    sorted_indices = sorted_indices.to(torch.int32).contiguous()

    inp_flat = permuted_tensor.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(inp_flat)

    BLOCK_S = 64
    n_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_s_tiles)

    _inverse_permute_kernel[grid](inp_flat, sorted_indices, out_flat, S, D, BLOCK_S, num_warps=4)

    original_tensor = out_flat.reshape(B, H, S, D)
    return original_tensor
