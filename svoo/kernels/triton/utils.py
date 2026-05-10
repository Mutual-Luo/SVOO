def flatten_if_batched(*tensors):
    """Flatten 3-D [B, N, D] tensors to [B * N, D]."""
    if not tensors:
        raise ValueError("At least one tensor must be provided.")

    first = tensors[0]
    assert len(first.shape) in [
        2,
        3,
    ], "Input tensors must be batched (3D) or not batched (2D)"

    if len(first.shape) == 3:
        batched = True
        batch_size = first.shape[0]
        assert all(t.shape[0] == batch_size for t in tensors), "All input tensors must have the same batch size"
        assert all(
            t.shape[1] == first.shape[1] for t in tensors
        ), "All input tensors must have the same sequence length"
        flat_tensors = [t.reshape(-1, t.shape[-1]) for t in tensors]
    else:
        batched = False
        batch_size = None
        flat_tensors = list(tensors)

    return flat_tensors, batched, batch_size
