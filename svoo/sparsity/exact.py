"""Attention sparsity profiling helpers."""

import os
import re

import torch

DEFAULT_SPARSITY_THRESHOLD = 0.95
DEFAULT_AUTO_MEMORY_FRACTION = 0.45

# Conservative estimate for the temporary tensors used by the exact PyTorch
# implementation: scores, probabilities, sorted values, sort indices, and cumsum.
_SORT_WORKSPACE_BYTES_PER_SCORE = 24
_SPARSITY_HEADER_RE = re.compile(r"\[Sparsity\] Layer (\d+) \| Type: \w+ \| Step: (\d+)")
_SPARSITY_HEAD_RE = re.compile(r"\s*Head\s+(\d+):\s+Sparsity=")
_SPARSITY_COMPLETION_CACHE = {}


def _auto_query_chunk_size(query: torch.Tensor, key: torch.Tensor) -> int:
    """Pick the largest safe exact query chunk size from currently free VRAM."""
    batch, num_heads, seq_len_q, _ = query.size()
    seq_len_k = key.size(2)

    if not query.is_cuda:
        return seq_len_q

    try:
        free_bytes, _ = torch.cuda.mem_get_info(query.device)
    except Exception:
        return min(128, seq_len_q)

    try:
        memory_fraction = float(os.environ.get("SVOO_SPARSITY_AUTO_MEM_FRACTION", DEFAULT_AUTO_MEMORY_FRACTION))
    except ValueError:
        memory_fraction = DEFAULT_AUTO_MEMORY_FRACTION
    memory_fraction = min(0.90, max(0.05, memory_fraction))

    bytes_per_query = max(1, batch * num_heads * seq_len_k * _SORT_WORKSPACE_BYTES_PER_SCORE)
    chunk_size = int(free_bytes * memory_fraction / bytes_per_query)
    if chunk_size >= 16:
        chunk_size = (chunk_size // 8) * 8
    return max(1, min(seq_len_q, chunk_size))


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _resume_enabled() -> bool:
    return os.environ.get("SVOO_SPARSITY_RESUME", "0") not in ("0", "", "false", "False")


def has_completed_sparsity_entry(output_file: str, layer_idx: int, step: int, num_heads: int) -> bool:
    """
    Return True when a raw sparsity log already contains all heads for a layer/step.

    This is an opt-in resume guard for long exact offline runs. It never changes
    sparsity values; it only avoids recomputing entries already present in the
    raw text file.
    """
    if not _resume_enabled() or not output_file or not os.path.exists(output_file):
        return False

    target_layer = int(layer_idx)
    target_step = int(step)
    required_heads = set(range(int(num_heads)))

    try:
        stat = os.stat(output_file)
        cache = _SPARSITY_COMPLETION_CACHE.get(output_file)
        file_id = (stat.st_dev, stat.st_ino)
        if cache is None or cache["file_id"] != file_id or stat.st_size < cache["offset"]:
            cache = {
                "file_id": file_id,
                "offset": 0,
                "current_key": None,
                "heads_by_entry": {},
            }
            _SPARSITY_COMPLETION_CACHE[output_file] = cache

        with open(output_file, "r", encoding="utf-8") as f:
            f.seek(cache["offset"])
            for line in f:
                header_match = _SPARSITY_HEADER_RE.match(line)
                if header_match:
                    current_layer = int(header_match.group(1))
                    current_step = int(header_match.group(2))
                    cache["current_key"] = (current_layer, current_step)
                    cache["heads_by_entry"].setdefault(cache["current_key"], set())
                    continue

                if cache["current_key"] is not None:
                    head_match = _SPARSITY_HEAD_RE.match(line)
                    if head_match:
                        cache["heads_by_entry"].setdefault(cache["current_key"], set()).add(int(head_match.group(1)))

            cache["offset"] = f.tell()

        return required_heads.issubset(cache["heads_by_entry"].get((target_layer, target_step), set()))
    except OSError:
        return False


def _counts_from_sorted_probabilities(sorted_weights: torch.Tensor, threshold: float) -> torch.Tensor:
    if sorted_weights.is_cuda and os.environ.get("SVOO_SPARSITY_USE_TRITON_PROB_COUNTS", "0") == "1":
        try:
            from ..kernels.triton.sparsity import counts_from_sorted_probabilities_triton

            counts, lower_margins, upper_margins = counts_from_sorted_probabilities_triton(
                sorted_weights, threshold, return_margins=True
            )

            try:
                boundary_tol = float(os.environ.get("SVOO_SPARSITY_TRITON_BOUNDARY_TOL", "1e-6"))
            except ValueError:
                boundary_tol = 1e-6
            boundary_tol = max(0.0, boundary_tol)

            fallback_mask = (lower_margins <= boundary_tol) | (upper_margins <= boundary_tol)
            if torch.any(fallback_mask):
                flat_weights = sorted_weights.contiguous().reshape(-1, sorted_weights.size(-1))
                flat_counts = counts.reshape(-1)
                flat_mask = fallback_mask.reshape(-1)

                fallback_weights = flat_weights[flat_mask]
                cumsum_weights = torch.cumsum(fallback_weights, dim=-1)
                target_mass = (threshold * cumsum_weights[..., -1:]).contiguous()
                fallback_counts = (
                    torch.searchsorted(cumsum_weights, target_mass, right=False, out_int32=True).squeeze(-1) + 1
                )
                flat_counts[flat_mask] = fallback_counts
                del fallback_weights, cumsum_weights, target_mass, fallback_counts

            return counts
        except Exception:
            pass

    cumsum_weights = torch.cumsum(sorted_weights, dim=-1)
    target_mass = (threshold * cumsum_weights[..., -1:]).contiguous()
    counts = torch.searchsorted(cumsum_weights, target_mass, right=False, out_int32=True).squeeze(-1) + 1
    del cumsum_weights, target_mass
    return counts


def _exact_counts_from_scores(attn_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    # Keep the operation order identical to the full slow reference. Sorting
    # logits and reconstructing softmax mass is close, but can move the
    # threshold by one token on floating-point boundary cases.
    scores = attn_scores.float()
    weights = torch.softmax(scores, dim=-1)
    del scores, attn_scores

    sorted_weights = torch.sort(weights, dim=-1, descending=True).values
    counts = _counts_from_sorted_probabilities(sorted_weights, threshold)
    del weights, sorted_weights
    return counts


def _sample_query_rows(query: torch.Tensor, sample_size: int):
    """Return a deterministic uniform subset of query rows, or all rows."""
    seq_len_q = query.size(2)
    sample_size = int(sample_size or 0)
    if sample_size <= 0 or sample_size >= seq_len_q:
        return query, seq_len_q

    if sample_size == 1:
        positions = torch.tensor([seq_len_q // 2], device=query.device, dtype=torch.long)
    else:
        positions = torch.linspace(0, seq_len_q - 1, steps=sample_size, device=query.device)
        positions = torch.unique_consecutive(positions.round().long())

    return query.index_select(2, positions), int(positions.numel())


@torch.no_grad()
def compute_exact_attention_sparsity(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    batch_size: int = 0,
    threshold: float = DEFAULT_SPARSITY_THRESHOLD,
    query_sample_size: int = 0,
):
    """
    Compute per-head attention sparsity.

    Sparsity is the minimal token ratio needed to cover `threshold` cumulative
    attention mass. By default all query rows are used. Set `query_sample_size`
    to a positive value to estimate the per-head average from a deterministic
    uniform subset of query rows. A non-positive `batch_size` enables automatic
    exact chunking based on available VRAM.
    """
    batch, num_heads, seq_len_q, dim = query.size()
    seq_len_k = key.size(2)
    threshold = float(threshold)
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError(f"sparsity threshold must be in (0, 1], got {threshold}")

    query, sampled_seq_len_q = _sample_query_rows(query, query_sample_size)
    profile_seq_len_q = query.size(2)

    auto_batch_size = batch_size is None or batch_size <= 0
    if auto_batch_size:
        batch_size = _auto_query_chunk_size(query, key)
    else:
        batch_size = min(int(batch_size), profile_seq_len_q)

    scale = dim**-0.5
    key_t = key.transpose(-2, -1)
    num_tokens_threshold_list = []

    q_start = 0
    while q_start < profile_seq_len_q:
        q_end = min(q_start + batch_size, profile_seq_len_q)
        query_batch = query[:, :, q_start:q_end, :]

        try:
            # Pass the scaled scores as a temporary so _exact_counts_from_scores
            # can release them before sorting probabilities.
            num_tokens = _exact_counts_from_scores(
                torch.matmul(query_batch, key_t).float().mul_(scale),
                threshold,
            )
        except RuntimeError as error:
            if not auto_batch_size or batch_size <= 1 or not _is_cuda_oom(error):
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            continue

        num_tokens_threshold_list.append(num_tokens)
        q_start = q_end

    num_tokens_threshold = torch.cat(num_tokens_threshold_list, dim=2)
    sparsity_per_query = num_tokens_threshold.float() / float(seq_len_k)
    sparsity_per_head = sparsity_per_query.mean(dim=(0, 2))
    avg_sparsity = sparsity_per_head.mean().item()

    stats = {
        "attn_scores_shape": [batch, num_heads, sampled_seq_len_q, seq_len_k],
        "seq_len": seq_len_q,
        "sampled_seq_len": sampled_seq_len_q,
        "chunk_size": batch_size,
    }
    return avg_sparsity, sparsity_per_head.detach().cpu().tolist(), stats
