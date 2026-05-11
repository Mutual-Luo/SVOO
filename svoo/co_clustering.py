import importlib
import json
import os
import threading
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from svoo.kernels.triton.l2norm import triton_l2norm_forward
from svoo.utils.flashinfer_sparse import make_variable_block_sparse_attention_wrapper


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _maybe_autotune(configs, *, key):
    if _env_flag("SVOO_TRITON_INPROCESS_AUTOTUNE", default=False):
        return triton.autotune(configs, key=key)
    return lambda fn: fn


_TUNE_CACHE = None
_TUNE_LOCK = threading.Lock()


def _tune_mode() -> str:
    return os.environ.get("SVOO_TRITON_TUNE", "fixed").strip().lower()


def _tune_enabled() -> bool:
    return _tune_mode() not in ("0", "false", "no", "off", "none", "fixed")


def _tune_force() -> bool:
    return _tune_mode() in ("1", "true", "yes", "on", "force", "retune")


def _tune_cache_path() -> str:
    explicit = os.environ.get("SVOO_TRITON_TUNE_CACHE")
    if explicit:
        return explicit
    cache_root = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.getcwd(), ".triton_cache")
    return os.path.join(cache_root, "svoo_tuning.json")


def _load_tune_cache() -> dict:
    global _TUNE_CACHE
    if _TUNE_CACHE is not None:
        return _TUNE_CACHE
    path = _tune_cache_path()
    try:
        with open(path, "r") as f:
            _TUNE_CACHE = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _TUNE_CACHE = {}
    return _TUNE_CACHE


def _save_tune_cache(cache: dict) -> None:
    path = _tune_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _device_tune_key(device) -> str:
    device = torch.device(device)
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return f"{props.name}|sm{props.major}{props.minor}|torch{torch.__version__}|triton{triton.__version__}"


def _cache_key(kernel_name: str, device, dtype, **meta) -> str:
    parts = {
        "kernel": kernel_name,
        "device": _device_tune_key(device),
        "dtype": str(dtype),
        **{k: int(v) for k, v in meta.items()},
    }
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def _config_candidates(configs, *, max_configs_env: str):
    max_configs = _env_int(max_configs_env, 0)
    if max_configs <= 0 or max_configs >= len(configs):
        return configs
    return configs[:max_configs]


def _config_dict(config) -> dict:
    data = {k: int(v) for k, v in config.kwargs.items()}
    data["num_warps"] = int(config.num_warps)
    data["num_stages"] = int(config.num_stages)
    return data


def _tune_or_load(kernel_name: str, cache_key: str, configs, bench_one, default: dict, *, max_configs_env: str) -> dict:
    if not _tune_enabled():
        return default

    with _TUNE_LOCK:
        cache = _load_tune_cache()
        if not _tune_force() and cache_key in cache:
            return {k: int(v) for k, v in cache[cache_key].items()}

    candidates = _config_candidates(configs, max_configs_env=max_configs_env)
    bench_warmup = _env_int("SVOO_TRITON_TUNE_WARMUP", 5)
    bench_rep = _env_int("SVOO_TRITON_TUNE_REP", 20)

    print(
        f"[SVOO] Tuning {kernel_name}: {len(candidates)} configs "
        f"(cache miss, mode={_tune_mode()})",
        flush=True,
    )
    timings = []
    for config in candidates:
        meta = _config_dict(config)
        try:
            ms = triton.testing.do_bench(
                lambda meta=meta: bench_one(meta),
                warmup=bench_warmup,
                rep=bench_rep,
            )
            timings.append((float(ms), meta))
        except Exception as exc:
            print(f"[SVOO]   skip {kernel_name} config {meta}: {type(exc).__name__}: {exc}", flush=True)

    if timings:
        _, best = min(timings, key=lambda item: item[0])
    else:
        best = default

    with _TUNE_LOCK:
        cache = _load_tune_cache()
        cache[cache_key] = best
        _save_tune_cache(cache)

    print(f"[SVOO] Best {kernel_name} config: {best}", flush=True)
    return best


def _require_flashinfer_sparse():
    """Load flashinfer sparse module or raise a clear, actionable error."""
    try:
        return importlib.import_module("flashinfer.sparse")
    except Exception as e:
        raise ImportError(
            "Failed to import flashinfer sparse backend (`flashinfer.sparse`). "
            "Install a FlashInfer build that contains sparse APIs, or set "
            "`SVG_SPARSE_ATTN_BACKEND=triton` to use Triton sparse attention.\n"
            f"Original error: {e}"
        ) from e

def density_calculation(dynamic_map, q_cluster_sizes, k_cluster_sizes):
    """
    Calculate the density of the dynamic map. Currently only batch size = 1 and head size = 1 are supported.

    Input:
        dynamic_map: [cfg, num_heads, qc_num, kc_num]
        q_cluster_sizes: [cfg, num_heads, qc_num]
        k_cluster_sizes: [cfg, num_heads, kc_num]
    """
    cfg, num_heads, qc_num, kc_num = dynamic_map.shape

    # Calculate the block size of each block
    clustered_block_size = q_cluster_sizes[:, :, :, None] * k_cluster_sizes[:, :, None, :]
    masked_block_size = clustered_block_size * dynamic_map

    # Calculate the density of each block
    density = torch.sum(masked_block_size, dim=(2, 3)) / torch.sum(clustered_block_size, dim=(2, 3))
    return density


# Chunk-wise centroid update for sorted cluster IDs.


@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D] - ORIGINAL ORDER
    sorted_idx_ptr,  # *i32        [B, N]    - indices after sort
    sorted_cluster_ptr,  # *i32        [B, N]    - cluster ids in sorted order
    sum_ptr,  # *f32        [B, K, D]
    count_ptr,  # *i32        [B, K]
    B,
    N,
    D: tl.constexpr,
    K: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many tokens (points) each program processes
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.
    """
    # program indices - 2-D launch grid: (chunk_id, batch_id)
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b
    chunk_start = pid_chunk * BLOCK_N  # position of the first token handled by this program

    # Nothing to do - out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base = sorted_idx_ptr + b * N
    cid_batch_base = sorted_cluster_ptr + b * N

    # helper aranges
    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, D)

    # first token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx, mask=valid_tok, other=-1)  # [BLOCK_N]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            # Gather with explicit strides (works for both contiguous and strided inputs).
            # Use int64 offsets to avoid overflow on large strides.
            b_i64 = b.to(tl.int64)
            all_tokens_idxs_i64 = all_tokens_idxs.to(tl.int64)
            offs_dim_i64 = offs_dim.to(tl.int64)
            x_ptrs = x_ptr + b_i64 * stride_x_b + all_tokens_idxs_i64[:, None] * stride_x_n + offs_dim_i64[None, :] * stride_x_d
            tok_valid = all_tokens_idxs[:, None] >= 0
            cluster_feats = tl.load(x_ptrs, mask=(cluster_mask[:, None] & tok_valid), other=0.0)  # [BLOCK_N, D]
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats)
            tl.atomic_add(count_ptr + b * K + cid, cluster_size)
def triton_centroid_update_sorted_euclid(
    x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor, *, BLOCK_N: int = 256
):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), B)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    _centroid_update_chunk_kernel[grid](
        x,  # original features
        sorted_idx_int,  # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B,
        N,
        D,
        K,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        BLOCK_N=BLOCK_N,
    )

    # Convert sums to means; replace empty clusters with old centroids
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    return centroids.to(x.dtype), centroid_cnts


# Triton kernel: compute nearest-centroid IDs (Euclidean distance)
# Inputs:
#   x           : (B, N, D)  float16 / float32
#   centroids   : (B, K, D)  same dtype as x
# Output:
#   cluster_ids : (B, N)     int32   - nearest centroid index per point
# Auto-tuning setup - explore various tile sizes / warp counts

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=4, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True


_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


@_maybe_autotune(_TUNE_CONFIGS, key=["K", "D"])
@triton.jit
def _euclid_assign_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D]
    c_ptr,  # *f16 / *f32 [B, K, D]
    x_sq_ptr,  # *f32         [B, N]
    out_ptr,  # *i32         [B, N]
    B,
    N,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)  # tile index along N dimension
    pid_b = tl.program_id(1)  # batch index

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    # Load x tile  (BLOCK_N, D)
    offs_d = tl.arange(0, D)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = x_ptr + pid_b * stride_x_b + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Pre-load x_sq for the tile  (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
    best_idx = tl.zeros((BLOCK_N,), tl.int32)
    # Iterate over centroids in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = c_ptr + pid_b * stride_c_b + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # Compute centroid squared norms (BLOCK_K,)
        cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)
    # Write results
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)
# Python wrapper


def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """Return nearest-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        x_sq      : (B, N)    float32 - ||x||^2 per point (on CUDA)

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"


    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int64)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)
    num_warps = _env_int("SVOO_EUCLID_ASSIGN_NUM_WARPS", 8)

    _euclid_assign_kernel[grid](
        x,
        centroids,
        x_sq,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_out_b,
        stride_out_n,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out


# 1. Euclidean
def _euclid_iter(x, x_sq, centroids):

    cluster_ids = euclid_assign_triton(x, centroids, x_sq)
    centroids_new, cluster_sizes = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)


    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids, cluster_sizes


# 2. Cosine
# 3. Dot-product
_euclid_iter_compiled = _euclid_iter


def batch_kmeans_Euclid(x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
        cluster_sizes: (B, n_clusters) LongTensor, number of points per cluster.
        n_iters: actual number of iterations executed (int)
    """
    B, N, D = x.shape

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x**2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(x, dim=1, index=indices[..., None].expand(-1, -1, D))  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        centroids_new, center_shift, cluster_ids, cluster_sizes = _euclid_iter_compiled(x, x_sq, centroids)

        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, cluster_sizes, it + 1


def weighted_softmax(scores, weights):
    """Softmax over scores after weighting each key cluster by its token count."""
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return softmax_out.to(input_dtype)


def identify_dynamic_map(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
    max_kc_ratio=0,
):
    """
    Args:
        min_kc_ratio: scalar or per-head ratio [H] for minimum key clusters kept.
        max_kc_ratio: optional scalar or per-head cap ratio. 0/None disables the cap.
                      Each query cluster keeps at most this ratio of key clusters.
    """
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()

    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if isinstance(min_kc_ratio, (list, tuple)) or (hasattr(min_kc_ratio, '__len__') and len(min_kc_ratio) == H):
        if not isinstance(min_kc_ratio, torch.Tensor):
            min_kc_ratio = torch.tensor(min_kc_ratio, device=device, dtype=torch.float32)
        for h in range(H):
            head_ratio = min_kc_ratio[h].item()
            if head_ratio > 0:
                preserve_length = int(head_ratio * kc_num)
                remove_indices[:, h, :, :preserve_length] = False
    elif isinstance(min_kc_ratio, (int, float)) and min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * kc_num)
        remove_indices[..., :preserve_length] = False

    if max_kc_ratio is not None:
        if isinstance(max_kc_ratio, (list, tuple)) or (hasattr(max_kc_ratio, "__len__") and len(max_kc_ratio) == H):
            if not isinstance(max_kc_ratio, torch.Tensor):
                max_kc_ratio = torch.tensor(max_kc_ratio, device=device, dtype=torch.float32)
            for h in range(H):
                head_ratio = float(max_kc_ratio[h].item())
                if head_ratio > 0:
                    cap = int(head_ratio * kc_num)
                    cap = max(1, cap)
                    if isinstance(min_kc_ratio, torch.Tensor) and min_kc_ratio.numel() == H:
                        cap = max(cap, int(float(min_kc_ratio[h].item()) * kc_num))
                    remove_indices[:, h, :, cap:] = True
        elif isinstance(max_kc_ratio, (int, float)) and float(max_kc_ratio) > 0:
            cap = int(float(max_kc_ratio) * kc_num)
            cap = max(1, cap)
            if isinstance(min_kc_ratio, torch.Tensor) and min_kc_ratio.numel() == H:
                for h in range(H):
                    cap_h = cap
                    head_min = float(min_kc_ratio[h].item())
                    if head_min > 0:
                        cap_h = max(cap_h, int(head_min * kc_num))
                    cap_h = max(1, cap_h)
                    remove_indices[:, h, :, cap_h:] = True
            else:
                if isinstance(min_kc_ratio, (int, float)) and float(min_kc_ratio) > 0:
                    cap = max(cap, int(float(min_kc_ratio) * kc_num))
                remove_indices[..., cap:] = True

    sorted_clusters_to_keep = ~remove_indices

    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map


def identify_dynamic_map_global(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
    key_tokens=None,
    k_labels=None,      # [B, S] key cluster labels
    num_frame=0,
    frame_size=0,
    context_length=0,
    timestep=0,
    layer_idx=0,
    num_layers=0,
    lambda_schedule="linear",
    diverse_top_p_k=0.0,
):
    """
    Identify the dynamic map with optional global diversity constraints.
    
    Strategy:
    1. If diverse_top_p_k > 0, use two-stage selection.
       - Stage 1 selects blocks by attention with budget p - diverse_top_p_k.
       - Stage 2 selects remaining blocks by diversity with budget diverse_top_p_k.
    2. Otherwise, select by the combined score.
    """
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device
    
    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()  # [B, H, 1, kc_num]
    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)  # [B, H, qc_num, kc_num]
    
    if key_tokens is not None and k_labels is not None:
        k_diversity = compute_cluster_diversity_vectorized(
            key_tokens, k_labels, k_cluster_sizes,
            num_frame, frame_size, context_length,
            kc_num, device
        )  # [B, H, kc_num]
        
        diversity_gains = k_diversity.unsqueeze(2).expand(-1, -1, qc_num, -1)
    else:
        diversity_gains = torch.zeros_like(weighted_attn_probs)
    
    if diversity_gains.max() > diversity_gains.min():
        diversity_gains = (diversity_gains - diversity_gains.min()) / (
            diversity_gains.max() - diversity_gains.min() + 1e-8
        )
    
    if diverse_top_p_k > 0 and diverse_top_p_k < p:
        attn_p = p - diverse_top_p_k
        
        sorted_attn, sorted_attn_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)
        cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
        remove_attn = cumsum_attn > attn_p
        remove_attn[..., 1:] = remove_attn[..., :-1].clone()
        remove_attn[..., 0] = False
        
        if isinstance(min_kc_ratio, (list, tuple)) or (hasattr(min_kc_ratio, '__len__') and len(min_kc_ratio) == H):
            if not isinstance(min_kc_ratio, torch.Tensor):
                min_kc_ratio = torch.tensor(min_kc_ratio, device=device, dtype=torch.float32)
            for h in range(H):
                head_ratio = min_kc_ratio[h].item()
                if head_ratio > 0:
                    preserve_length = int(head_ratio * kc_num)
                    remove_attn[:, h, :, :preserve_length] = False
        elif isinstance(min_kc_ratio, (int, float)) and min_kc_ratio > 0:
            preserve_length = int(min_kc_ratio * kc_num)
            remove_attn[..., :preserve_length] = False
        
        selected_by_attn_sorted = ~remove_attn  # [B, H, qc_num, kc_num]
        
        attn_selected_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
        attn_selected_map.scatter_(-1, sorted_attn_indices, selected_by_attn_sorted)
        
        remaining_diversity = diversity_gains.clone()
        remaining_diversity = remaining_diversity.masked_fill(attn_selected_map, float('-inf'))
        
        sorted_diversity, sorted_diversity_indices = torch.sort(remaining_diversity, dim=-1, descending=True)
        
        remaining_diversity_valid = sorted_diversity.clone()
        remaining_diversity_valid = remaining_diversity_valid.masked_fill(remaining_diversity_valid == float('-inf'), 0.0)
        
        remaining_diversity_sum = remaining_diversity_valid.sum(dim=-1, keepdim=True)
        remaining_diversity_probs = remaining_diversity_valid / (remaining_diversity_sum + 1e-8)
        
        cumsum_diversity = torch.cumsum(remaining_diversity_probs, dim=-1)
        remove_diversity = cumsum_diversity > diverse_top_p_k
        remove_diversity[..., 1:] = remove_diversity[..., :-1].clone()
        remove_diversity[..., 0] = False
        
        sorted_attn_selected = torch.gather(attn_selected_map, dim=-1, index=sorted_diversity_indices)
        remove_diversity = remove_diversity | sorted_attn_selected
        
        selected_by_diversity_sorted = ~remove_diversity
        
        diversity_selected_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
        diversity_selected_map.scatter_(-1, sorted_diversity_indices, selected_by_diversity_sorted)
        
        dynamic_map = attn_selected_map | diversity_selected_map
        
    else:
        lambda_weight = compute_lambda_schedule(
            timestep, layer_idx, num_layers, lambda_schedule
        )
        
        combined_scores = lambda_weight * weighted_attn_probs + (1 - lambda_weight) * diversity_gains
        
        sorted_scores, sorted_indices = torch.sort(combined_scores, dim=-1, descending=True)
        
        cumsum_scores = torch.cumsum(sorted_scores, dim=-1)
        remove_indices = cumsum_scores > p
        remove_indices[..., 1:] = remove_indices[..., :-1].clone()
        remove_indices[..., 0] = False
        
        if isinstance(min_kc_ratio, (list, tuple)) or (hasattr(min_kc_ratio, '__len__') and len(min_kc_ratio) == H):
            if not isinstance(min_kc_ratio, torch.Tensor):
                min_kc_ratio = torch.tensor(min_kc_ratio, device=device, dtype=torch.float32)
            for h in range(H):
                head_ratio = min_kc_ratio[h].item()
                if head_ratio > 0:
                    preserve_length = int(head_ratio * kc_num)
                    remove_indices[:, h, :, :preserve_length] = False
        elif isinstance(min_kc_ratio, (int, float)) and min_kc_ratio > 0:
            preserve_length = int(min_kc_ratio * kc_num)
            remove_indices[..., :preserve_length] = False
        
        sorted_clusters_to_keep = ~remove_indices
        
        dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
        dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    
    return dynamic_map


def compute_cluster_diversity_vectorized(
    tokens,  # [B, H, S, D]
    labels,
    cluster_sizes,  # [B, H, num_clusters]
    num_frame,
    frame_size,
    context_length,
    num_clusters,
    device
):
    """
    Compute cluster diversity gains from coverage and intra-cluster variance.
    Returns: [B, H, num_clusters].
    """
    B, H, S, D = tokens.shape
    video_length = num_frame * frame_size if num_frame > 0 and frame_size > 0 else S
    
    if labels.shape[0] == B * H:
        labels_reshaped = labels.view(B, H, S)
        labels_for_coverage = labels_reshaped[:, 0, :]  # [B, S]
        labels_for_variance = labels_reshaped  # [B, H, S]
    else:
        labels_for_coverage = labels  # [B, S]
        labels_for_variance = labels.unsqueeze(1).expand(-1, H, -1)  # [B, H, S]
    
    spatiotemporal_coverage = compute_spatiotemporal_coverage_fully_vectorized(
        labels_for_coverage, num_frame, frame_size, context_length, video_length, S, num_clusters, device
    )
    
    cluster_variance = compute_cluster_variance_fully_vectorized(
        tokens, labels_for_variance, cluster_sizes, num_clusters
    )
    
    spatiotemporal_coverage = spatiotemporal_coverage.unsqueeze(1).expand(-1, H, -1)
    
    diversity = spatiotemporal_coverage + cluster_variance
    
    return diversity


def compute_spatiotemporal_coverage_fully_vectorized(
    labels,  # [B, S]
    num_frame,
    frame_size,
    context_length,
    video_length,
    seq_len,
    num_clusters,
    device
):
    """
    Fully vectorized spatiotemporal coverage.
    Returns: [B, num_clusters].
    """
    B, S = labels.shape
    
    if num_frame == 0 or frame_size == 0:
        cluster_mask = (labels.unsqueeze(-1) == torch.arange(num_clusters, device=device)).float()
        coverage = cluster_mask.sum(dim=1) / seq_len
        return coverage
    
    frame_indices = torch.arange(S, device=device)
    video_mask = frame_indices < video_length
    frame_ids = torch.where(video_mask, frame_indices // frame_size, torch.full_like(frame_indices, -1))
    
    cluster_mask = (labels.unsqueeze(-1) == torch.arange(num_clusters, device=device)).float()
    
    video_mask_expanded = video_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, num_clusters).float()
    video_cluster_mask = cluster_mask * video_mask_expanded
    
    spatial_coverage = video_cluster_mask.sum(dim=1) / video_length
    
    frame_ids_expanded = frame_ids.unsqueeze(0).unsqueeze(-1).expand(B, -1, num_clusters)  # [B, S, num_clusters]
    valid_frame_mask = (frame_ids_expanded >= 0).float()  # [B, S, num_clusters]
    valid_cluster_mask = video_cluster_mask * valid_frame_mask  # [B, S, num_clusters]
    
    max_frame_id = num_frame - 1
    frame_ids_clamped = torch.clamp(frame_ids_expanded, 0, max_frame_id)  # [B, S, num_clusters]
    
    frame_onehot = F.one_hot(frame_ids_clamped.long(), num_classes=num_frame).float()
    
    valid_cluster_mask_expanded = valid_cluster_mask.unsqueeze(-1)  # [B, S, num_clusters, 1]
    frame_onehot = frame_onehot * valid_cluster_mask_expanded
    
    unique_frames_per_cluster = (frame_onehot.sum(dim=1) > 0).float().sum(dim=-1)  # [B, num_clusters]
    temporal_coverage = unique_frames_per_cluster / num_frame
    
    coverage = (temporal_coverage + spatial_coverage) / 2.0
    
    return coverage


def compute_cluster_variance_fully_vectorized(
    tokens,  # [B, H, S, D]
    labels,
    cluster_sizes,  # [B, H, num_clusters]
    num_clusters
):
    """
    Fully vectorized intra-cluster variance.
    Returns: [B, H, num_clusters].
    """
    B, H, S, D = tokens.shape
    
    if labels.dim() == 2:
        labels_expanded = labels.unsqueeze(1).expand(-1, H, -1)  # [B, H, S]
    else:
        labels_expanded = labels
    
    labels_onehot = F.one_hot(labels_expanded, num_classes=num_clusters).float()  # [B, H, S, num_clusters]
    
    tokens_float = tokens.float()
    cluster_sums = torch.einsum('bhsd,bhsk->bhkd', tokens_float, labels_onehot)
    
    cluster_sizes_expanded = cluster_sizes.unsqueeze(-1).float().clamp(min=1.0)  # [B, H, num_clusters, 1]
    centroids = cluster_sums / cluster_sizes_expanded
    
    
    labels_expanded_long = labels_expanded.long()  # [B, H, S]
    labels_gather_idx = labels_expanded_long.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, H, S, D]
    # centroids: [B, H, num_clusters, D], index: [B, H, S, D]
    token_centroids = torch.gather(
        centroids, 
        dim=2, 
        index=labels_gather_idx
    )  # [B, H, S, D]
    
    distances = torch.norm(tokens_float - token_centroids, dim=-1)  # [B, H, S]
    
    cluster_variance_sum = torch.einsum('bhs,bhsk->bhk', distances, labels_onehot)  # [B, H, num_clusters]
    
    cluster_variance = cluster_variance_sum / cluster_sizes_expanded.squeeze(-1).clamp(min=1.0)
    
    max_variance = 10.0
    cluster_variance = torch.clamp(cluster_variance / max_variance, 0.0, 1.0)
    
    return cluster_variance


def compute_lambda_schedule(timestep, layer_idx, num_layers, schedule_type="linear"):
    """
    Compute the diversity/attention mixing weight.
    Earlier timesteps/layers favor diversity; later ones favor attention.
    """
    if schedule_type == "constant":
        return 0.5
    
    if isinstance(timestep, torch.Tensor):
        timestep_val = timestep.item() if timestep.numel() == 1 else timestep[0].item()
    else:
        timestep_val = timestep
    
    if schedule_type == "linear":
        timestep_weight = timestep_val / 1000.0
    elif schedule_type == "cosine":
        import math
        timestep_weight = 0.5 * (1 - math.cos(timestep_val / 1000.0 * math.pi))
    else:
        timestep_weight = timestep_val / 1000.0
    
    if num_layers > 0:
        layer_weight = layer_idx / num_layers
    else:
        layer_weight = 0.5
    
    lambda_weight = 0.6 * timestep_weight + 0.4 * layer_weight

    return max(0.0, min(1.0, lambda_weight))


# Dynamic block sparse attention, Triton implementation.


@triton.jit
def _dynamic_block_sparse_fwd_kernel(
    Q,
    K,
    V,
    Out,
    dynamic_map,
    qc_cum_size,
    kc_cum_size,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_dmap_b,
    stride_dmap_h,
    stride_dmap_qc,
    stride_dmap_kc,
    stride_qcs_b,
    stride_qcs_h,
    stride_qcs_qc,
    stride_kcs_b,
    stride_kcs_h,
    stride_kcs_kc,
    B,
    H,
    S,
    D,
    scale,
    QC_NUM: tl.constexpr,
    KC_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for dynamic block sparse attention.
    Each program computes attention for one query block within a batch/head.
    Processes query block in chunks of BLOCK_M.
    Iterates through key blocks, checking dynamic_map.
    Processes key/value blocks in chunks of BLOCK_N.
    Uses online softmax.
    """
    pid = tl.program_id(axis=0)

    pid_q_block_global = pid  # 0 to B*H*QC_NUM - 1

    # Need to map pid (0.. B*H*QC_NUM-1) back to (b, h, q_block_idx)
    # q_block_idx changes fastest, then h, then b
    q_block_idx = pid_q_block_global % QC_NUM
    pid_h_temp = pid_q_block_global // QC_NUM
    h = pid_h_temp % H
    b = pid_h_temp // H
    qcs_offset = b * stride_qcs_b + h * stride_qcs_h
    q_start_offset = tl.load(qc_cum_size + qcs_offset + q_block_idx * stride_qcs_qc)
    q_end_offset = tl.load(qc_cum_size + qcs_offset + (q_block_idx + 1) * stride_qcs_qc)
    q_block_size = q_end_offset - q_start_offset

    # Early exit if the query block is empty
    if q_block_size == 0:
        return
    q_ptr_base = Q + b * stride_qb + h * stride_qh + q_start_offset * stride_qs
    k_ptr_base = K + b * stride_kb + h * stride_kh
    v_ptr_base = V + b * stride_vb + h * stride_vh
    out_ptr_base = Out + b * stride_ob + h * stride_oh + q_start_offset * stride_os
    dmap_ptr = dynamic_map + b * stride_dmap_b + h * stride_dmap_h + q_block_idx * stride_dmap_qc
    kcs_ptr = kc_cum_size + b * stride_kcs_b + h * stride_kcs_h
    offs_qm = tl.arange(0, BLOCK_M)  # Query block row offsets [0, 1, ..., BLOCK_M-1]
    offs_d = tl.arange(0, BLOCK_D)  # Dimension offsets [0, 1, ..., BLOCK_D-1]

    for q_chunk_start in range(0, q_block_size, BLOCK_M):
        q_chunk_rows = offs_qm + q_chunk_start
        q_rows_mask = q_chunk_rows < q_block_size  # Mask for valid rows in this Q chunk [BLOCK_M]
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max score
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp(scores - max)
        acc_o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Accumulated output
        q_ptr = q_ptr_base + q_chunk_rows[:, None] * stride_qs + offs_d[None, :]
        # Mask ensures we don't read out of bounds for the query block or dimension D
        mask_q = q_rows_mask[:, None] & (offs_d[None, :] < D)
        q_chunk = tl.load(q_ptr, mask=mask_q, other=0.0)  # Shape: [BLOCK_M, BLOCK_D]
        for k_block_idx in range(KC_NUM):
            is_active = tl.load(dmap_ptr + k_block_idx * stride_dmap_kc)
            if is_active:  # Process block only if it's active
                k_start_offset = tl.load(kcs_ptr + k_block_idx * stride_kcs_kc)
                k_end_offset = tl.load(kcs_ptr + (k_block_idx + 1) * stride_kcs_kc)
                k_block_size = k_end_offset - k_start_offset

                # Skip if the key block is empty (inside the active block check)
                if k_block_size > 0:

                    k_block_ptr_base = k_ptr_base + k_start_offset * stride_ks
                    v_block_ptr_base = v_ptr_base + k_start_offset * stride_vs
                    offs_kn = tl.arange(0, BLOCK_N)  # Key block row offsets [0, ..., BLOCK_N-1]
                    for k_chunk_start in range(0, k_block_size, BLOCK_N):
                        k_chunk_rows = offs_kn + k_chunk_start
                        k_rows_mask = k_chunk_rows < k_block_size  # Mask for valid rows in this K/V chunk [BLOCK_N]
                        k_ptr = k_block_ptr_base + k_chunk_rows[:, None] * stride_ks + offs_d[None, :]
                        v_ptr = v_block_ptr_base + k_chunk_rows[:, None] * stride_vs + offs_d[None, :]

                        # Mask ensures we don't read out of bounds for the key block or dimension D
                        mask_kv = k_rows_mask[:, None] & (offs_d[None, :] < D)
                        k_chunk = tl.load(k_ptr, mask=mask_kv, other=0.0)  # Shape: [BLOCK_N, BLOCK_D]
                        v_chunk = tl.load(v_ptr, mask=mask_kv, other=0.0)  # Shape: [BLOCK_N, BLOCK_D]
                        # QK^T: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
                        s_ij_chunk = tl.dot(q_chunk, k_chunk.T) * scale

                        # IMPORTANT: Mask out scores corresponding to padding in K before max/softmax
                        # Set scores for invalid K elements to -inf
                        s_ij_chunk = tl.where(k_rows_mask[None, :], s_ij_chunk, -float("inf"))
                        # Mask out scores for invalid Q elements as well (although q_chunk elements are 0, avoid potential issues)
                        s_ij_chunk = tl.where(q_rows_mask[:, None], s_ij_chunk, -float("inf"))
                        # Current max for this Q-K chunk interaction
                        m_ij_chunk = tl.max(s_ij_chunk, axis=1)  # Shape: [BLOCK_M]

                        # Update overall max (across K chunks seen so far for this Q chunk)
                        m_new = tl.maximum(m_i, m_ij_chunk)  # Shape: [BLOCK_M]

                        # Calculate scaled probabilities P_ij = exp(S_ij - m_new)
                        p_ij_chunk = tl.exp(s_ij_chunk - m_new[:, None])  # Shape: [BLOCK_M, BLOCK_N]
                        # Zero out probabilities for masked K elements before summing
                        p_ij_chunk = tl.where(k_rows_mask[None, :], p_ij_chunk, 0.0)

                        # Calculate scaling factor for previous accumulator state
                        exp_m_diff = tl.exp(m_i - m_new)  # Shape: [BLOCK_M]

                        # Update sum accumulator (denominator L)
                        l_i_chunk = tl.sum(p_ij_chunk, axis=1)  # Sum probabilities for this chunk, shape [BLOCK_M]
                        l_i = (l_i * exp_m_diff) + l_i_chunk  # Shape: [BLOCK_M]

                        # Update output accumulator O
                        # P_ij @ V_j: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
                        # Ensure p_ij_chunk is the correct dtype for dot product
                        p_ij_chunk_casted = p_ij_chunk.to(V.dtype.element_ty)
                        o_chunk = tl.dot(p_ij_chunk_casted, v_chunk)  # Shape: [BLOCK_M, BLOCK_D]

                        acc_o = (acc_o * exp_m_diff[:, None]) + o_chunk  # Shape: [BLOCK_M, BLOCK_D]

                        # Update max for the next K chunk/block
                        m_i = m_new
            # End of 'if is_active:' block
        # Normalize the accumulated output: O = acc_o / l_i
        # Add epsilon to l_i to avoid division by zero
        l_i_safe = tl.where(l_i == 0, 1.0, l_i)  # Avoid 0/0 -> NaN
        o_final_chunk = acc_o / (l_i_safe[:, None])
        o_final_chunk = tl.where(l_i[:, None] == 0, 0.0, o_final_chunk)  # Ensure output is 0 if l_i was 0
        out_ptr = out_ptr_base + q_chunk_rows[:, None] * stride_os + offs_d[None, :]
        # Mask ensures we don't write out of bounds for the query block or dimension D
        mask_out = q_rows_mask[:, None] & (offs_d[None, :] < D)
        tl.store(out_ptr, o_final_chunk.to(Out.dtype.element_ty), mask=mask_out)


def dynamic_block_sparse_fwd_triton(q, k, v, dynamic_map, qc_size, kc_size):
    """
    Launcher for the Triton dynamic block sparse attention kernel.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        dynamic_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num].
        qc_size (torch.Tensor): Query block sizes, shape [B, H, qc_num].
        kc_size (torch.Tensor): Key block sizes, shape [B, H, kc_num].

    Returns:
        torch.Tensor: Output tensor, shape [B, H, S, D].
    """
    B, H, S, D = q.shape
    qc_num = qc_size.shape[-1]
    kc_num = kc_size.shape[-1]
    dtype = q.dtype

    # Assertions and checks
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert dynamic_map.is_cuda and qc_size.is_cuda and kc_size.is_cuda
    assert q.dtype == k.dtype == v.dtype, "Input dtypes must match"
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert D in [16, 32, 64, 128], "Head dimension D must be 16, 32, 64, or 128 for efficient Triton dot"
    # Ensure sequence lengths match sum of block sizes (check on one batch/head for simplicity)
    assert S == torch.sum(qc_size[0, 0, :]), "Sum of qc_size must equal S"
    assert S == torch.sum(kc_size[0, 0, :]), "Sum of kc_size must equal S"
    # Ensure dynamic_map is boolean
    assert dynamic_map.dtype == torch.bool

    # Calculate scale factor (using float32 for stability)
    scale = D**-0.5

    # Precompute cumulative sizes (on CPU/GPU, keep on device)
    qc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(qc_size[..., :1]), qc_size], dim=-1), dim=-1).int()
    kc_cum_size = torch.cumsum(torch.cat([torch.zeros_like(kc_size[..., :1]), kc_size], dim=-1), dim=-1).int()

    # Output tensor
    out = torch.empty_like(q)

    # Triton kernel config
    # NOTE: tl.arange(0, BLOCK_*) requires BLOCK_* to be a power-of-two at compile time.
    # For small sequence lengths (e.g. unit tests), min(BLOCK_M, S) may become non-power-of-two and
    # Triton compilation fails. We therefore clamp to the largest power-of-two <= min(default, S).
    BLOCK_D = D

    def _pow2_floor(n: int) -> int:
        # largest power-of-two <= n, with a minimum of 1
        return 1 << (int(n).bit_length() - 1) if n > 0 else 1

    if S <= 1024:
        block_m_default = 64
        block_n_default = 64
    else:
        block_m_default = 128
        block_n_default = 64

    BLOCK_M = _pow2_floor(min(block_m_default, S))
    BLOCK_N = _pow2_floor(min(block_n_default, S))

    # Safety: keep reasonably sized tiles (avoid BLOCK_* = 1 unless S == 1)
    if S >= 16:
        BLOCK_M = max(BLOCK_M, 16)
        BLOCK_N = max(BLOCK_N, 16)

    # Launch grid: One program per query block per batch/head
    grid = (B * H * qc_num,)

    # Call the kernel
    _dynamic_block_sparse_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        dynamic_map,
        qc_cum_size,
        kc_cum_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dynamic_map.stride(0),
        dynamic_map.stride(1),
        dynamic_map.stride(2),
        dynamic_map.stride(3),
        qc_cum_size.stride(0),
        qc_cum_size.stride(1),
        qc_cum_size.stride(2),
        kc_cum_size.stride(0),
        kc_cum_size.stride(1),
        kc_cum_size.stride(2),
        B,
        H,
        S,
        D,
        scale,
        QC_NUM=qc_num,
        KC_NUM=kc_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return out


def dynamic_block_sparse_fwd_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    is_cpu: bool = True,
):
    """
    Launcher for the Flashinfer dynamic block sparse attention kernel.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        block_mask_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num]. Currently must on CPU.
        block_row_sz (torch.Tensor): Query block sizes, shape [B, H, qc_num]. Currently must on CPU.
        block_col_sz (torch.Tensor): Key block sizes, shape [B, H, kc_num]. Currently must on CPU.
        is_cpu (bool): Whether to run on CPU. Flashinfer default is to run on CPU. We switch to GPU for faster planning. Default is True.
    """
    flashinfer_sparse = _require_flashinfer_sparse()
    # Input shape check
    B, H, S, D = q.shape
    qc_num = block_row_sz.shape[-1]
    kc_num = block_col_sz.shape[-1]
    assert block_mask_map.shape == (B, H, qc_num, kc_num)

    assert (
        all(t.device == torch.device("cpu") for t in [block_mask_map, block_row_sz, block_col_sz]) if is_cpu else True
    )

    # Check if block_col_sz and block_row_sz are the same for each head
    assert torch.all(block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0])
    assert torch.all(block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0])


    workspace_bytes = _env_int("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", 128 * 1024 * 1024)
    f_buffer = torch.empty((workspace_bytes,), dtype=torch.uint8, device=q.device)
    flashinfer_backend = os.environ.get("SVOO_FLASHINFER_SPARSE_BACKEND", "auto").lower()
    if flashinfer_backend not in ("auto", "fa2", "fa3"):
        raise ValueError(
            "SVOO_FLASHINFER_SPARSE_BACKEND must be one of: auto, fa2, fa3; "
            f"got {flashinfer_backend!r}"
        )
    wrapper = make_variable_block_sparse_attention_wrapper(
        flashinfer_sparse, f_buffer, backend=flashinfer_backend
    )
    int_workspace_bytes = _env_int("SVOO_FLASHINFER_SPARSE_INT_WORKSPACE_BYTES", 0)
    if int_workspace_bytes > 0:
        i_buffer = torch.empty((int_workspace_bytes,), dtype=torch.uint8, device=q.device)
        wrapper.reset_workspace_buffer(
            float_workspace_buffer=f_buffer,
            int_workspace_buffer=i_buffer,
        )

    # Reshape inputs to (B * H, ...)
    q = q.reshape(B * H, S, D)
    k = k.reshape(B * H, S, D)
    v = v.reshape(B * H, S, D)
    block_mask_map = block_mask_map.reshape(B * H, qc_num, kc_num)
    block_row_sz = block_row_sz.reshape(B * H, qc_num)
    block_col_sz = block_col_sz.reshape(B * H, kc_num)

    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=B * H,
        num_kv_heads=B * H,
        head_dim=D,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    o = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
    del wrapper, f_buffer
    if int_workspace_bytes > 0:
        del i_buffer
    o = o.reshape(B, H, S, D)
    return o


# LOW-MEMORY CO-CLUSTERING
# Replace the original 3-step pipeline (matmul -> l2norm -> profile_assign) with 2
# Triton kernels, eliminating HBM materialisation of the (B, N, K) profile tensor.
#
#  Original peak memory (Wan21-1.3B 720p, qc=256/kc=1024):
#    profile_q  bf16:  (12,75600,1024)x2 B ~ 1.86 GB
#    profile_q  fp32:  (12,75600,1024)x4 B ~ 3.73 GB  (new alloc for l2norm)
#    Step B peak ~ 5.6 GB
#
#  New peak memory:
#    norms (B, N) fp32 ~ 7 MB
#    labels (B, N) int32 ~ 3.6 MB
#    register / SRAM tiles: <32 KB / thread block
# Autotune configs
_TUNE_PROFILE_NORM = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=ns, num_warps=wp)
    for BN in [16, 32, 64, 128]
    for BK in [16, 32]
    for wp in [4, 8]
    for ns in [1, 2]
    if BN * BK >= 16 * 16
]

_TUNE_COCL_ASSIGN = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_J": BJ}, num_stages=ns, num_warps=wp)
    for BN in [16, 32, 64]
    for BK in [16, 32]
    for BJ in [16, 32]
    for wp in [4, 8]
    for ns in [1, 2]
    if BK >= 16 and BJ >= 16
]
# Kernel 1 / 2 : profile row-norm  (Pass 1)
@_maybe_autotune(_TUNE_PROFILE_NORM, key=["K", "D"])
@triton.jit
def _profile_norm_kernel(
    x_ptr,      # (B, N, D)  tokens
    kc_ptr,     # (B, K, D)  profile-forming centroids
    norm_ptr,   # (B, N) fp32 output
    B,
    N,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_kcb, stride_kck, stride_kcd,
    stride_normb, stride_normn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute profile row norms tile-by-tile, never materialising (B, N, K).
    norm[b, n] = sqrt( sum_k (x[b,n,:] * kc[b,k,:])^2 )
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B:
        return

    n_start = pid_n * BLOCK_N
    n_offs  = n_start + tl.arange(0, BLOCK_N)
    n_mask  = n_offs < N
    d_offs  = tl.arange(0, D)

    # Load x_tile (BLOCK_N, D) - loaded ONCE for all k iterations
    x_ptrs = (x_ptr
               + pid_b * stride_xb
               + n_offs[:, None] * stride_xn
               + d_offs[None, :] * stride_xd)
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    sq_norm = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load kc_tile as (D, BLOCK_K) - matches _euclid_assign_kernel layout
        kc_ptrs = (kc_ptr
                   + pid_b * stride_kcb
                   + d_offs[:, None]  * stride_kcd
                   + k_offs[None, :]  * stride_kck)
        kc_tile = tl.load(kc_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)  # (D, BLOCK_K)

        # dot (BLOCK_N, BLOCK_K) = x_tile @ kc_tile  <- contraction over D
        dot = tl.dot(x_tile, kc_tile, allow_tf32=True)
        dot = tl.where(k_mask[None, :], dot, 0.0)

        sq_norm += tl.sum(dot * dot, axis=1)   # (BLOCK_N,)

    norms = tl.sqrt(sq_norm + 1e-8)
    norm_ptrs = norm_ptr + pid_b * stride_normb + n_offs * stride_normn
    tl.store(norm_ptrs, norms, mask=n_mask)
# Kernel 2 / 2 : fused profile-space nearest-centroid (Pass 2)
@_maybe_autotune(_TUNE_COCL_ASSIGN, key=["K", "J", "D"])
@triton.jit
def _fused_cocluster_assign_kernel(
    x_ptr,    # (B, N, D)  tokens
    kc_ptr,   # (B, K, D)  profile-forming centroids
    pc_ptr,   # (B, J, K)  fp32 normalised profile centroids
    norm_ptr, # (B, N)     fp32 profile row norms
    out_ptr,  # (B, N)     int32 output labels
    B,
    N,
    K: tl.constexpr,
    J: tl.constexpr,
    D: tl.constexpr,
    stride_xb,   stride_xn,   stride_xd,
    stride_kcb,  stride_kck,  stride_kcd,
    stride_pcb,  stride_pcj,  stride_pck,
    stride_normb, stride_normn,
    stride_outb,  stride_outn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """
    Fused profile-space argmin without materialising (B,N,K) or (B,N,J) in HBM.
    labels[b,n] = argmin_j  2 - 2*dot( norm(x[b,n]@kc[b].T), pc_norm[b,j] )

    Algorithm
    ---------
    Load x_tile (BLOCK_N, D) ONCE.
    Outer loop j_tile: init dot_nj = 0 (BLOCK_N, BLOCK_J)
      Inner loop k_tile:
        dot_xkc = x_tile @ kc_tile / norm      (BLOCK_N, BLOCK_K)
        pc_tile  loaded as (BLOCK_K, BLOCK_J)  (transposed from (B,J,K) layout)
        dot_nj  += dot_xkc @ pc_tile           (BLOCK_N, BLOCK_J)
      Update running best_dist / best_idx.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B:
        return

    n_start = pid_n * BLOCK_N
    n_offs  = n_start + tl.arange(0, BLOCK_N)
    n_mask  = n_offs < N
    d_offs  = tl.arange(0, D)

    # Load x_tile (BLOCK_N, D) - loaded ONCE, reused across all j and k iterations
    x_ptrs = (x_ptr
               + pid_b * stride_xb
               + n_offs[:, None] * stride_xn
               + d_offs[None, :] * stride_xd)
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    # Load profile row norms (BLOCK_N,)
    norm_ptrs = norm_ptr + pid_b * stride_normb + n_offs * stride_normn
    norms = tl.load(norm_ptrs, mask=n_mask, other=1.0)   # default 1.0 avoids /0

    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx  = tl.zeros((BLOCK_N,), tl.int32)

    for j_start in range(0, J, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < J

        # Accumulate normalised-profile * profile-centroid dot products
        dot_nj = tl.zeros((BLOCK_N, BLOCK_J), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # kc_tile: (D, BLOCK_K) - same layout trick as _euclid_assign_kernel
            kc_ptrs = (kc_ptr
                       + pid_b * stride_kcb
                       + d_offs[:, None]  * stride_kcd
                       + k_offs[None, :] * stride_kck)
            kc_tile = tl.load(kc_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

            # dot_xkc (BLOCK_N, BLOCK_K) = x_tile @ kc_tile / norm
            dot_xkc = tl.dot(x_tile, kc_tile, allow_tf32=True)
            dot_xkc = dot_xkc / norms[:, None]
            dot_xkc = tl.where(k_mask[None, :], dot_xkc, 0.0)

            # pc_tile: loaded as (BLOCK_K, BLOCK_J) - swapped indices relative to (B,J,K) storage
            # pc[b, j, k]  ->  address = pcb + j*stride_pcj + k*stride_pck
            # We want tile[k_local, j_local] = pc[b, j_start+j_local, k_start+k_local]
            # -> ptrs use k_offs[:, None]*stride_pck + j_offs[None,:]*stride_pcj
            pc_ptrs = (pc_ptr
                       + pid_b * stride_pcb
                       + k_offs[:, None]  * stride_pck
                       + j_offs[None, :] * stride_pcj)
            pc_tile = tl.load(pc_ptrs,
                               mask=(k_mask[:, None] & j_mask[None, :]),
                               other=0.0)   # (BLOCK_K, BLOCK_J) fp32

            # dot_nj (BLOCK_N,BLOCK_J) += dot_xkc (BLOCK_N,BLOCK_K) @ pc_tile (BLOCK_K,BLOCK_J)
            dot_nj += tl.dot(dot_xkc, pc_tile, allow_tf32=True)

        dist = 2.0 - 2.0 * dot_nj
        dist = tl.where(j_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update    = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx  = tl.where(update, (j_start + curr_idx).to(tl.int32), best_idx)

    out_ptrs = out_ptr + pid_b * stride_outb + n_offs * stride_outn
    tl.store(out_ptrs, best_idx, mask=n_mask)
# Python wrappers
def profile_norm_triton(
    x: torch.Tensor,
    kcentroids: torch.Tensor,
    *,
    BLOCK_N: int | None = None,
    BLOCK_K: int | None = None,
) -> torch.Tensor:
    """
    Compute profile row norms without materialising (B, N, K) in HBM.

    norm[b, n] = ||x[b, n, :] @ kcentroids[b, :, :].T||2

    Args:
        x:          (B, N, D) float16/bfloat16 tokens
        kcentroids: (B, K, D) float16/bfloat16 profile-forming centroids
    Returns:
        norms: (B, N) float32
    """
    assert x.is_cuda and kcentroids.is_cuda, "tensors must be on CUDA"
    B, N, D = x.shape
    K = kcentroids.shape[1]
    assert kcentroids.shape == (B, K, D), f"kcentroids shape mismatch: {kcentroids.shape}"

    x          = x.contiguous()
    kcentroids = kcentroids.contiguous()
    norms      = torch.empty(B, N, device=x.device, dtype=torch.float32)
    if BLOCK_N is None:
        BLOCK_N = _env_int("SVOO_PROFILE_NORM_BLOCK_N", 32)
    if BLOCK_K is None:
        BLOCK_K = _env_int("SVOO_PROFILE_NORM_BLOCK_K", 32)
    default_num_warps = 8 if K <= 256 else 4
    default_meta = {
        "BLOCK_N": int(BLOCK_N),
        "BLOCK_K": int(BLOCK_K),
        "num_warps": _env_int("SVOO_PROFILE_NORM_NUM_WARPS", default_num_warps),
        "num_stages": _env_int("SVOO_PROFILE_NORM_NUM_STAGES", 2),
    }
    cache_key = _cache_key("profile_norm", x.device, x.dtype, K=K, D=D)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    def _run(meta):
        _profile_norm_kernel[grid](
            x, kcentroids, norms,
            B, N, K, D,
            x.stride(0),          x.stride(1),          x.stride(2),
            kcentroids.stride(0), kcentroids.stride(1), kcentroids.stride(2),
            norms.stride(0),      norms.stride(1),
            BLOCK_N=meta["BLOCK_N"],
            BLOCK_K=meta["BLOCK_K"],
            num_warps=meta["num_warps"],
            num_stages=meta["num_stages"],
        )

    meta = _tune_or_load(
        "profile_norm",
        cache_key,
        _TUNE_PROFILE_NORM,
        _run,
        default_meta,
        max_configs_env="SVOO_PROFILE_NORM_TUNE_MAX_CONFIGS",
    )
    _run(meta)
    return norms


def fused_cocluster_assign_triton(
    x: torch.Tensor,
    kcentroids: torch.Tensor,
    profile_centroids: torch.Tensor,
    norms: torch.Tensor,
    *,
    BLOCK_N: int | None = None,
    BLOCK_K: int | None = None,
    BLOCK_J: int | None = None,
) -> torch.Tensor:
    """
    Fused profile-space nearest-centroid search without materialising (B,N,K) or (B,N,J).

    Args:
        x:                   (B, N, D)  float16/bfloat16 tokens
        kcentroids:          (B, K, D)  profile-forming centroids (same dtype as x)
        profile_centroids:   (B, J, K)  float32 L2-normalised profile centroids
        norms:               (B, N)     float32 profile row norms from profile_norm_triton
    Returns:
        labels: (B, N) int64, values in [0, J)
    """
    assert x.is_cuda, "tensors must be on CUDA"
    B, N, D = x.shape
    K = kcentroids.shape[1]
    J = profile_centroids.shape[1]
    assert kcentroids.shape       == (B, K, D), f"kcentroids shape: {kcentroids.shape}"
    assert profile_centroids.shape == (B, J, K), f"profile_centroids shape: {profile_centroids.shape}"
    assert norms.shape             == (B, N),    f"norms shape: {norms.shape}"
    assert profile_centroids.dtype == torch.float32, "profile_centroids must be float32"

    x                 = x.contiguous()
    kcentroids        = kcentroids.contiguous()
    profile_centroids = profile_centroids.contiguous()
    norms             = norms.contiguous()

    out  = torch.empty(B, N, device=x.device, dtype=torch.int32)
    if BLOCK_N is None:
        BLOCK_N = _env_int("SVOO_COCL_ASSIGN_BLOCK_N", 64)
    if BLOCK_K is None:
        BLOCK_K = _env_int("SVOO_COCL_ASSIGN_BLOCK_K", 32)
    if BLOCK_J is None:
        BLOCK_J = _env_int("SVOO_COCL_ASSIGN_BLOCK_J", 32)
    default_meta = {
        "BLOCK_N": int(BLOCK_N),
        "BLOCK_K": int(BLOCK_K),
        "BLOCK_J": int(BLOCK_J),
        "num_warps": _env_int("SVOO_COCL_ASSIGN_NUM_WARPS", 4),
        "num_stages": _env_int("SVOO_COCL_ASSIGN_NUM_STAGES", 2),
    }
    cache_key = _cache_key("cocluster_assign", x.device, x.dtype, K=K, J=J, D=D)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    def _run(meta):
        _fused_cocluster_assign_kernel[grid](
            x, kcentroids, profile_centroids, norms, out,
            B, N, K, J, D,
            x.stride(0),                 x.stride(1),                 x.stride(2),
            kcentroids.stride(0),        kcentroids.stride(1),        kcentroids.stride(2),
            profile_centroids.stride(0), profile_centroids.stride(1), profile_centroids.stride(2),
            norms.stride(0),             norms.stride(1),
            out.stride(0),               out.stride(1),
            BLOCK_N=meta["BLOCK_N"],
            BLOCK_K=meta["BLOCK_K"],
            BLOCK_J=meta["BLOCK_J"],
            num_warps=meta["num_warps"],
            num_stages=meta["num_stages"],
        )

    meta = _tune_or_load(
        "cocluster_assign",
        cache_key,
        _TUNE_COCL_ASSIGN,
        _run,
        default_meta,
        max_configs_env="SVOO_COCL_ASSIGN_TUNE_MAX_CONFIGS",
    )
    _run(meta)
    return out.long()
# Low-memory co-clustering
def co_cluster_tokens(
    q_flat,
    k_flat,
    num_q_centroids,
    num_k_centroids,
    max_iters=10,
):
    """Cluster query/key tokens with the final low-memory Triton path.

    This avoids materialising the large (B, N, K) profile tensor by using two
    Triton passes: one for profile row norms and one for fused profile-space
    nearest-centroid assignment.
    """
    B_heads, N, D = q_flat.shape
    device = q_flat.device
    q_indices  = torch.randint(0, N, (B_heads, num_q_centroids), device=device)
    qcentroids = torch.gather(q_flat, dim=1, index=q_indices[..., None].expand(-1, -1, D))

    k_indices  = torch.randint(0, N, (B_heads, num_k_centroids), device=device)
    kcentroids = torch.gather(k_flat, dim=1, index=k_indices[..., None].expand(-1, -1, D))

    for _ in range(max_iters):
        # Step A: Update Key Clusters
        # Profile of key[n] lives in Q-centroid space (dim = num_q_centroids)
        # Centroid-centroid profiles are small: (B, num_k_centroids, num_q_centroids)
        profile_centroids_k = torch.matmul(kcentroids, qcentroids.transpose(-2, -1))
        profile_centroids_k = triton_l2norm_forward(profile_centroids_k, eps=1e-8).contiguous()

        k_norms = profile_norm_triton(k_flat, qcentroids)       # (B, N)  no large tensor
        klabels = fused_cocluster_assign_triton(                 # (B, N)  no large tensor
            k_flat, qcentroids, profile_centroids_k, k_norms
        )
        kcentroids, k_cluster_sizes = triton_centroid_update_sorted_euclid(
            k_flat, klabels, kcentroids, BLOCK_N=128
        )

        # Step B: Update Query Clusters
        # Profile of query[n] lives in K-centroid space (dim = num_k_centroids)
        profile_centroids_q = torch.matmul(qcentroids, kcentroids.transpose(-2, -1))
        profile_centroids_q = triton_l2norm_forward(profile_centroids_q, eps=1e-8).contiguous()

        q_norms = profile_norm_triton(q_flat, kcentroids)        # (B, N)
        qlabels = fused_cocluster_assign_triton(                  # (B, N)
            q_flat, kcentroids, profile_centroids_q, q_norms
        )
        qcentroids, q_cluster_sizes = triton_centroid_update_sorted_euclid(
            q_flat, qlabels, qcentroids, BLOCK_N=128
        )

    return qlabels, qcentroids, q_cluster_sizes, klabels, kcentroids, k_cluster_sizes
