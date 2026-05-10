import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import flashinfer
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch.nn.attention.flex_attention import (
    flex_attention,
)
from diffusers.models.transformers.transformer_wan import _get_added_kv_projections, dispatch_attention_fn
from ...kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from ...kernels.triton.rmsnorm import triton_rmsnorm_forward
from ...co_clustering import (
    batch_kmeans_Euclid,
    co_cluster_tokens,
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
)
from ...sparsity import DEFAULT_SPARSITY_THRESHOLD, compute_exact_attention_sparsity, has_completed_sparsity_entry
from .placement import (
    ref_wan_hidden_states_placement,
    ref_wan_sparse_head_placement,
    wan_hidden_states_placement,
    wan_sparse_head_placement,
)
from .utils import (
    create_block_mask_cached,
    flashinfer_sparse_attn_forward,
    gen_temporal_mask,
    generate_temporal_head_mask_mod,
)

_ENABLE_FAST_KERNEL_ENV = os.environ.get("ENABLE_FAST_KERNEL", "1")
_should_use_fast_kernel = _ENABLE_FAST_KERNEL_ENV == "1"

if _should_use_fast_kernel:
    ENABLE_FAST_KERNEL = True
else:
    ENABLE_FAST_KERNEL = False
print(f"[SVOO] ENABLE_FAST_KERNEL={int(ENABLE_FAST_KERNEL)} for Wan.", flush=True)


flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


class WanAttn_SVGAttn_Processor2_0:
    version = None
    context_length = 0
    num_frame = 0
    frame_size = 0

    first_layers_fp = 0
    first_times_fp = 0

    num_sampled_rows = 32
    attention_masks = None
    sparsity = 0
    use_spargeattn = False

    block_mask = None
    temporal_mask_metadata = None
    
    measure_attention_sparsity = False
    sparsity_output_file = "attention_sparsity.txt"
    sparsity_batch_size = 0  # 0 auto-selects the largest safe exact query chunk.
    sparsity_query_samples = 0  # 0 uses all query rows.
    sparsity_threshold = DEFAULT_SPARSITY_THRESHOLD
    sparsity_start_step = 1
    
    _current_internal_timestep = None
    _inference_step_counter = 0
    _seen_timesteps = set()
    
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self._last_logged_step = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")


    def get_qkv(self, attn, hidden_states, encoder_hidden_states):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if hasattr(attn, "fused_projections") and attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def get_qk_norm(self, attn, query, key):
        if attn.norm_q is not None:
            query = triton_rmsnorm_forward(query, attn.norm_q.weight, attn.norm_q.eps)
        if attn.norm_k is not None:
            key = triton_rmsnorm_forward(key, attn.norm_k.weight, attn.norm_k.eps)
        return query, key

    def get_transpose_qkv(self, attn, query, key, value):
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        return query, key, value



        

    def get_o(self, attn, query, hidden_states, hidden_states_img):
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            if hidden_states_img.dim() == 4:
                hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = self.get_qkv(attn, hidden_states, encoder_hidden_states)

        query, key = self.get_qk_norm(attn, query, key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        query, key, value = self.get_transpose_qkv(attn, query, key, value)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        save_qk_enabled = os.getenv("SAVE_QK", "0") == "1"
        if save_qk_enabled and timestep is not None:
            if isinstance(timestep, torch.Tensor):
                current_step = timestep[0].item() if timestep.dim() > 0 else timestep.item()
            else:
                current_step = timestep
            
            save_start_step = int(os.getenv("SAVE_QK_START_STEP", "0"))
            save_dir = os.getenv("SAVE_QK_DIR", "")
            
            if current_step >= save_start_step and save_dir:
                step_dir = os.path.join(save_dir, f"step{current_step}")
                os.makedirs(step_dir, exist_ok=True)
                
                save_file = os.path.join(step_dir, f"layer{self.layer_idx}.pt")
                torch.save(
                    {
                        "q": query.detach().cpu(),
                        "k": key.detach().cpu(),
                        "timestep": current_step,
                        "layer_idx": self.layer_idx,
                    },
                    save_file,
                )
        if timestep is None:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:  # The main attention
            # Only Log Self-Attention
            if self.measure_attention_sparsity:
                self.log_attention_sparsity(query, key, value, timestep, "self")
            hidden_states = self.attention_core_logic(query, key, value, timestep)

        hidden_states = self.get_o(attn, query, hidden_states, hidden_states_img)

        return hidden_states

    def sample_mse(self, query, key, value):
        if self.attention_masks is None:
            raise ValueError("attention_masks is not set. This method is only for SVG pattern.")
        if not hasattr(self, 'num_sampled_rows') or self.num_sampled_rows is None:
            raise ValueError("num_sampled_rows is not set. This method is only for SVG pattern.")
        if not hasattr(self, 'sample_mse_max_row') or self.sample_mse_max_row is None:
            raise ValueError("sample_mse_max_row is not set. This method is only for SVG pattern.")
        assert len(self.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    def sparse_flashinfer_attention(self, query, key, value, temporal_mask_metadata):
        return flashinfer_sparse_attn_forward(query, key, value, temporal_mask_metadata)

    def sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):
        query_out, key_out, value_out = ref_wan_sparse_head_placement(
            query, key, value, best_mask_idx, context_length, num_frame, frame_size
        )
        return query_out, key_out, value_out

    def fast_sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_sparse_head_placement(
            query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
        )
        return query_out, key_out, value_out

    def hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_wan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    def fast_hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    def flash_attention(self, query, key, value):
        output_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        return output_hidden_states
    
    def flash_attention_spargeattn(self, query, key, value):
        from spas_sage_attn import spas_sage_attn_meansim_topk_cuda
        output_hidden_states = spas_sage_attn_meansim_topk_cuda(query, key, value, is_causal=False, dropout_p=0.0, topk=0.5)
        return output_hidden_states

    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep,
    ):
        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        if self.attention_masks is None:
            output_hidden_states = self.flash_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep is not None:
            if isinstance(timestep, torch.Tensor):
                if timestep.numel() > 0:
                    timestep_value = timestep[0].item()
                else:
                    timestep_value = None
            else:
                timestep_value = timestep
            
            if timestep_value is not None and timestep_value > self.first_times_fp:
                full_attention_flag = True

        if full_attention_flag:
            output_hidden_states = self.flash_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        else:
            if self.use_spargeattn:
                output_hidden_states = self.flash_attention_spargeattn(query, key, value)
                return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
            else:
                sampled_mses = self.sample_mse(query, key, value)
                best_mask_idx = torch.argmin(sampled_mses, dim=0)

                output_hidden_states = torch.zeros_like(query)
                query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

                query_out, key_out, value_out = self.fast_sparse_head_placement(
                    query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
                )

                hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

                self.fast_hidden_states_placement(
                    hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
                )

                return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)

    @torch.no_grad()
    def compute_attention_sparsity(
        self,
        query,
        key,
        value,
        attn_type="self",
        batch_size=0,
        threshold=DEFAULT_SPARSITY_THRESHOLD,
        query_sample_size=0,
    ):
        """
        Compute attention sparsity: the minimal token ratio needed to cover
        `threshold` cumulative attention mass for each head. When
        `query_sample_size` is positive, estimate from sampled query rows.
        """
        return compute_exact_attention_sparsity(
            query,
            key,
            batch_size=batch_size,
            threshold=threshold,
            query_sample_size=query_sample_size,
        )
    
    def _get_inference_step(self, timestep):
        """
        
        """
        cls = self.__class__
        
        # timestep is provided by the scheduler; just read it (no global synchronize here).
        if isinstance(timestep, torch.Tensor):
            internal_t = timestep[0].item()
        else:
            internal_t = timestep
        
        if cls._current_internal_timestep is not None and internal_t > cls._current_internal_timestep:
            cls._inference_step_counter = 0
            cls._seen_timesteps.clear()
        
        if internal_t not in cls._seen_timesteps:
            cls._seen_timesteps.add(internal_t)
            cls._inference_step_counter += 1
            cls._current_internal_timestep = internal_t
        
        return cls._inference_step_counter
    
    def log_attention_sparsity(self, query, key, value, timestep, attn_type="self"):
        measure_sparsity = getattr(self, 'measure_attention_sparsity', False)
        if not measure_sparsity:
            return
        
        inference_step = self._get_inference_step(timestep)
        
        start_step = getattr(self, 'sparsity_start_step', 1)
        if inference_step is None or inference_step < start_step:
            return

        if self._last_logged_step == inference_step:
            return
        self._last_logged_step = inference_step
        
        import os
        
        threshold = getattr(self, 'sparsity_threshold', DEFAULT_SPARSITY_THRESHOLD)
        batch_size = getattr(self, 'sparsity_batch_size', 0)
        query_sample_size = getattr(self, 'sparsity_query_samples', 0)
        output_file = self.sparsity_output_file
        if has_completed_sparsity_entry(output_file, self.layer_idx, inference_step, query.size(1)):
            return

        avg_sparsity, sparsity_per_head, stats = self.compute_attention_sparsity(
            query,
            key,
            value,
            attn_type,
            batch_size=batch_size,
            threshold=threshold,
            query_sample_size=query_sample_size,
        )

        seq_len_n = stats.get('seq_len', 0)
        sampled_seq_len_n = stats.get('sampled_seq_len', seq_len_n)

        log_lines = []
        sample_text = f" | Query Samples: {sampled_seq_len_n}/{seq_len_n}" if sampled_seq_len_n != seq_len_n else ""
        log_lines.append(f"[Sparsity] Layer {self.layer_idx} | Type: {attn_type} | Step: {inference_step} | Threshold: {threshold:.2%} | Avg Sparsity: {avg_sparsity:.4f} | n={seq_len_n}{sample_text}")

        for head_idx, head_sparsity in enumerate(sparsity_per_head):
            log_lines.append(f"  Head {head_idx:2d}: Sparsity={head_sparsity:.4f}")
        
        for line in log_lines:
            print(line)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            with open(output_file, "a") as f:
                for line in log_lines:
                    f.write(line + "\n")

def prepare_flexattention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    seq_len = context_length + num_frame * frame_size
    query, key, value = [
        torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)
    ]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
    _ = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


def prepare_flashinfer_attention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    temporal_mask_metadata = gen_temporal_mask(num_frame, frame_size, multiplier)

    return temporal_mask_metadata


class WanAttn_SAPAttn_Processor(WanAttn_SVGAttn_Processor2_0):
    num_layers = 0
    num_q_centroids = 0
    num_k_centroids = 0
    top_p_kmeans = 0
    min_kc_ratio = 0

    centroids_init = False
    q_centroids = None
    k_centroids = None

    kmeans_iter_init = 0
    kmeans_iter_step = 0
    zero_step_kmeans_init = False

    logging_file = None

    use_global_constraints = False
    lambda_schedule = "linear"
    diverse_top_p_k = 0.0
    
    use_routing_transformer_strategy = False
    
    mq1 = None
    mk1 = None
    mq2 = None
    mk2 = None
    q_bucket_labels = None
    k_bucket_labels = None
    q_sorted_indices = None
    k_sorted_indices = None
    q_bucket_cluster_centroids = None
    k_bucket_cluster_centroids = None
    buckets_init = False
    
    use_svoo = True  # Open-source release defaults to SVOO.
    enable_mem_save = bool(int(os.environ.get("SVG_ENABLE_MEM_SAVE", "0")))
    
    start_reuse_step = None
    reuse_interval = 1
    
    use_dynamic_min_kc_ratio = False
    sparsity_lookup = None
    dynamic_min_kc_ratio_min = None
    dynamic_min_kc_ratio_max = None

    def kmeans_init(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        
        if self.use_svoo:
            # SVOO co-clustering with Triton kernels.
            q_flat = query.view(cfg * num_heads, seq_len, dim)
            k_flat = key.view(cfg * num_heads, seq_len, dim)
            
            qlabels, qcentroids, qcluster_sizes, klabels, kcentroids, kcluster_sizes = co_cluster_tokens(
                q_flat, k_flat, self.num_q_centroids, self.num_k_centroids, max_iters=self.kmeans_iter_init
            )
            qiter = self.kmeans_iter_init
            kiter = self.kmeans_iter_init
        else:
            qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
                query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init
            )
            klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
                key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init
            )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def kmeans_step(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        
        if self.use_svoo:
            # SVOO co-clustering with Triton kernels.
            q_flat = query.view(cfg * num_heads, seq_len, dim)
            k_flat = key.view(cfg * num_heads, seq_len, dim)
            qlabels, qcentroids, qcluster_sizes, klabels, kcentroids, kcluster_sizes = co_cluster_tokens(
                q_flat, k_flat, self.num_q_centroids, self.num_k_centroids, max_iters=self.kmeans_iter_step
            )
            qiter = self.kmeans_iter_step
            kiter = self.kmeans_iter_step
        else:
            qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
                query.view(cfg * num_heads, seq_len, dim),
                n_clusters=self.num_q_centroids,
                max_iters=self.kmeans_iter_step,
                init_centroids=self.q_centroids,
            )
            klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
                key.view(cfg * num_heads, seq_len, dim),
                n_clusters=self.num_k_centroids,
                max_iters=self.kmeans_iter_step,
                init_centroids=self.k_centroids,
            )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def kmeans_clustering(self, query, key, layer_idx):
        """
        
        Args:
            query: query tensor
            key: key tensor
            layer_idx: layer index
        
        Returns:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter
        """
        cfg, num_heads, seq_len, dim = query.size()
        
        should_recluster = True
        current_step = getattr(self, 'current_inference_step', None)
        if os.getenv("SVG_REUSE_DEBUG", "0") not in ("0", "", "false", "False"):
            print(
                f"[reuse] current_step={current_step}, start_reuse_step={self.start_reuse_step}, reuse_interval={self.reuse_interval}"
            )
        if current_step is not None and self.start_reuse_step is not None and self.start_reuse_step > 0:
            if current_step >= self.start_reuse_step:
                offset = current_step - self.start_reuse_step
                should_recluster = (offset % self.reuse_interval == 0)
            else:
                should_recluster = True
            if os.getenv("SVG_REUSE_DEBUG", "0") not in ("0", "", "false", "False"):
                print(f"[reuse] offset={offset}, should_recluster={should_recluster}")
        
        if not should_recluster:
            if not hasattr(self, '_cached_clustering_results') or self._cached_clustering_results is None:
                if os.getenv("SVG_REUSE_DEBUG", "0") not in ("0", "", "false", "False"):
                    print("[reuse] No cached clustering results, forcing recluster")
                should_recluster = True
            else:
                if os.getenv("SVG_REUSE_DEBUG", "0") not in ("0", "", "false", "False"):
                    print("[reuse] Reusing cached clustering results")
                cached_results = self._cached_clustering_results
                qlabels = cached_results['qlabels']
                qcentroids = cached_results['qcentroids']
                qcluster_sizes = cached_results['qcluster_sizes']
                qiter = cached_results['qiter']
                klabels = cached_results['klabels']
                kcentroids = cached_results['kcentroids']
                kcluster_sizes = cached_results['kcluster_sizes']
                kiter = cached_results['kiter']
                
                return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

        if self.use_routing_transformer_strategy:
            if not self.buckets_init:
                q_norms = torch.norm(query, p=2, dim=-1)  # [cfg, H, S]
                k_norms = torch.norm(key, p=2, dim=-1)    # [cfg, H, S]
                
                q_norms_flat = q_norms.view(cfg * num_heads, seq_len)  # [cfg*H, S]
                k_norms_flat = k_norms.view(cfg * num_heads, seq_len)  # [cfg*H, S]
                
                q_bucket_labels = self._uniform_bucketing(q_norms_flat, self.mq1)  # [cfg*H, S]
                k_bucket_labels = self._uniform_bucketing(k_norms_flat, self.mk1)  # [cfg*H, S]
                
                q_sorted_indices = torch.argsort(q_bucket_labels, dim=-1)  # [cfg*H, S]
                k_sorted_indices = torch.argsort(k_bucket_labels, dim=-1)  # [cfg*H, S]
                q_inverse_indices = torch.argsort(q_sorted_indices, dim=-1)  # [cfg*H, S]
                k_inverse_indices = torch.argsort(k_sorted_indices, dim=-1)  # [cfg*H, S]
                
                self.q_bucket_labels = q_bucket_labels
                self.k_bucket_labels = k_bucket_labels
                self.q_sorted_indices = q_sorted_indices
                self.k_sorted_indices = k_sorted_indices
                self.q_inverse_indices = q_inverse_indices
                self.k_inverse_indices = k_inverse_indices
                self.buckets_init = False
            else:
                q_bucket_labels = self.q_bucket_labels
                k_bucket_labels = self.k_bucket_labels
                q_sorted_indices = self.q_sorted_indices
                k_sorted_indices = self.k_sorted_indices
                q_inverse_indices = self.q_inverse_indices
                k_inverse_indices = self.k_inverse_indices
            
            query_reshaped = query.view(cfg * num_heads, seq_len, dim)
            key_reshaped = key.view(cfg * num_heads, seq_len, dim)
            
            init_q_centroids = self.q_bucket_cluster_centroids if self.centroids_init else None
            qlabels, qcentroids, qcluster_sizes = self._bucket_clustering_parallel(
                query_reshaped, q_bucket_labels, self.mq1, self.mq2, 
                self.kmeans_iter_init if not self.centroids_init else self.kmeans_iter_step,
                init_centroids=init_q_centroids,
                sorted_indices=q_sorted_indices,
                inverse_indices=q_inverse_indices
            )
            self.q_bucket_cluster_centroids = qcentroids
            
            init_k_centroids = self.k_bucket_cluster_centroids if self.centroids_init else None
            klabels, kcentroids, kcluster_sizes = self._bucket_clustering_parallel(
                key_reshaped, k_bucket_labels, self.mk1, self.mk2,
                self.kmeans_iter_init if not self.centroids_init else self.kmeans_iter_step,
                init_centroids=init_k_centroids,
                sorted_indices=k_sorted_indices,
                inverse_indices=k_inverse_indices
            )
            self.k_bucket_cluster_centroids = kcentroids
            
            qlabels = q_bucket_labels * self.mq2 + qlabels   # [cfg*H, S]
            klabels = k_bucket_labels * self.mk2 + klabels   # [cfg*H, S]
            
            num_q_clusters = self.mq1 * self.mq2
            num_k_clusters = self.mk1 * self.mk2
            
            q_cluster_sizes_final = torch.zeros(cfg * num_heads, num_q_clusters, device=query.device, dtype=torch.long)
            k_cluster_sizes_final = torch.zeros(cfg * num_heads, num_k_clusters, device=query.device, dtype=torch.long)
            q_cluster_sizes_final.scatter_add_(1, qlabels, torch.ones_like(qlabels, dtype=torch.long))
            k_cluster_sizes_final.scatter_add_(1, klabels, torch.ones_like(klabels, dtype=torch.long))
            
            qcentroids_final = qcentroids.view(cfg * num_heads, num_q_clusters, dim)
            kcentroids_final = kcentroids.view(cfg * num_heads, num_k_clusters, dim)
            
            qcentroids = qcentroids_final
            kcentroids = kcentroids_final
            qcluster_sizes = q_cluster_sizes_final
            kcluster_sizes = k_cluster_sizes_final
            qiter = self.kmeans_iter_init if not self.centroids_init else self.kmeans_iter_step
            kiter = qiter
            self.centroids_init = False
                
        else:
            if not self.centroids_init:
                qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(
                    query, key, layer_idx
                )
                self.centroids_init = True
                print(f"Centroids initialized at layer {layer_idx}. Init step: {self.kmeans_iter_init}")
            else:
                qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(
                    query, key, layer_idx
                )
        
        if should_recluster:
            self._cached_clustering_results = {
                'qlabels': qlabels.detach(),
                'qcentroids': qcentroids.detach(),
                'qcluster_sizes': qcluster_sizes.detach(),
                'qiter': qiter,
                'klabels': klabels.detach(),
                'kcentroids': kcentroids.detach(),
                'kcluster_sizes': kcluster_sizes.detach(),
                'kiter': kiter,
            }
            
            if os.getenv("COMPUTE_REUSE_SIMILARITY", "0") == "1":
                current_step = getattr(self, 'current_inference_step', None)
                if current_step is not None:
                    if not hasattr(self, '_clustering_history'):
                        from collections import defaultdict
                        self._clustering_history = defaultdict(dict)
                    
                    if qlabels.dim() == 2 and klabels.dim() == 2:
                        cfg_actual = qlabels.size(0) // num_heads
                        if cfg_actual >= 1:
                            for head_idx in range(num_heads):
                                qlabels_pos = qlabels[head_idx].cpu().clone()  # [seq_len]
                                klabels_pos = klabels[head_idx].cpu().clone()  # [seq_len]
                                
                                position_key = (layer_idx, head_idx)
                                
                                if current_step > 0 and position_key in self._clustering_history:
                                    prev_steps = [s for s in self._clustering_history[position_key].keys() if s < current_step]
                                    if prev_steps:
                                        prev_step = max(prev_steps)
                                        prev_qlabels, prev_klabels = self._clustering_history[position_key][prev_step]
                                        
                                        try:
                                            from sklearn.metrics import normalized_mutual_info_score
                                            if isinstance(prev_qlabels, torch.Tensor):
                                                prev_qlabels_np = prev_qlabels.numpy()
                                                prev_klabels_np = prev_klabels.numpy()
                                            else:
                                                prev_qlabels_np = prev_qlabels
                                                prev_klabels_np = prev_klabels
                                            
                                            q_nmi = normalized_mutual_info_score(prev_qlabels_np, qlabels_pos.numpy(), average_method='arithmetic')
                                            k_nmi = normalized_mutual_info_score(prev_klabels_np, klabels_pos.numpy(), average_method='arithmetic')
                                            avg_sim = (q_nmi + k_nmi) / 2.0
                                            
                                            print(f"[Reuse Similarity] Step {prev_step}->{current_step}, Layer {layer_idx}, Head {head_idx}: "
                                                  f"Q={q_nmi:.4f}, K={k_nmi:.4f}, Avg={avg_sim:.4f}")
                                        except ImportError:
                                            pass
                                
                                self._clustering_history[position_key][current_step] = (qlabels_pos, klabels_pos)

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def _uniform_bucketing(self, norms, num_buckets):
        """
        """
        B, N = norms.shape
        device = norms.device
        
        sorted_indices = torch.argsort(norms, dim=-1)  # [B, N]
        
        bucket_size = N // num_buckets
        
        bucket_labels_sorted = torch.arange(N, device=device).unsqueeze(0).expand(B, -1) // bucket_size
        bucket_labels_sorted = torch.clamp(bucket_labels_sorted, 0, num_buckets - 1)
        
        bucket_labels = torch.zeros(B, N, dtype=torch.long, device=device)
        bucket_labels.scatter_(1, sorted_indices, bucket_labels_sorted)
        
        return bucket_labels
    
    def _bucket_clustering_parallel(self, tokens, bucket_labels, num_buckets, num_clusters_per_bucket, max_iters, init_centroids=None, sorted_indices=None, inverse_indices=None):
        """
        
        Args:
            
        Returns:
        """
        from ...co_clustering import batch_kmeans_Euclid
        
        B, N, D = tokens.shape
        device = tokens.device
        
        assert N % num_buckets == 0, f"Sequence length {N} must be divisible by num_buckets {num_buckets} for uniform bucketing"
        bucket_size = N // num_buckets
        
        if sorted_indices is not None:
            sorted_tokens = torch.gather(tokens, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
        else:
            sorted_indices = torch.argsort(bucket_labels, dim=-1)  # [B, N]
            sorted_tokens = torch.gather(tokens, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
        
        sorted_tokens_original = sorted_tokens.clone()
        
        bucket_tokens_flat = sorted_tokens.view(B * num_buckets, bucket_size, D)
        
        init_centroids_reshaped = init_centroids.view(B * num_buckets, num_clusters_per_bucket, D) if init_centroids is not None else None
        if init_centroids_reshaped is not None:
            init_centroids_reshaped = F.normalize(init_centroids_reshaped, p=2, dim=-1)
        
        all_bucket_labels, all_centroids, _, _ = batch_kmeans_Euclid(
            bucket_tokens_flat, n_clusters=num_clusters_per_bucket, max_iters=max_iters, init_centroids=init_centroids_reshaped
        )
        
        cluster_labels_sorted = all_bucket_labels.view(B, N)
        if inverse_indices is not None:
            cluster_labels = torch.gather(cluster_labels_sorted, 1, inverse_indices)
        else:
            inverse_indices = torch.argsort(sorted_indices, dim=-1)
            cluster_labels = torch.gather(cluster_labels_sorted, 1, inverse_indices)
        
        
        all_centroids = all_centroids.reshape(B, num_buckets * num_clusters_per_bucket, D)
        cluster_sizes = torch.zeros(B, num_buckets * num_clusters_per_bucket, device=device, dtype=torch.long)
        cluster_indices = bucket_labels * num_clusters_per_bucket + cluster_labels
        cluster_sizes.scatter_add_(1, cluster_indices, torch.ones_like(cluster_indices, dtype=torch.long))
        
        return cluster_labels, all_centroids, cluster_sizes

    def _get_min_kc_ratio_for_heads_from_step(self, inference_step, num_heads):
        """
        
        Args:
            num_heads: number of heads
        
        Returns:
        """
        if self.use_dynamic_min_kc_ratio and self.sparsity_lookup is not None and inference_step is not None:
            if not self.sparsity_lookup.has_data_for_step_layer(inference_step, self.layer_idx, num_heads):
                return self.min_kc_ratio
            
            sparsity_values = self.sparsity_lookup.get_sparsity_batch(inference_step, self.layer_idx, num_heads)
            
            final_values = []
            for csv_value in sparsity_values:
                value = csv_value
                if self.dynamic_min_kc_ratio_min is not None:
                    value = max(value, self.dynamic_min_kc_ratio_min)
                if self.dynamic_min_kc_ratio_max is not None:
                    value = min(value, self.dynamic_min_kc_ratio_max)
                final_values.append(value)
            
            return final_values
        
        return self.min_kc_ratio

    def semantic_aware_permutation(self, query, key, value, timestep=None):
        cfg, num_heads, seq_len, dim = query.size()

        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(
            query, key, self.layer_idx
        )
        
        min_kc_ratio = self._get_min_kc_ratio_for_heads_from_step(self.current_inference_step, num_heads)
        
        # 2. Identify dynamic map or indices
        if self.use_routing_transformer_strategy:
            import torch.nn.functional as F
            

            num_q_clusters = self.mq1 * self.mq2
            num_k_clusters = self.mk1 * self.mk2
            
            from ...co_clustering import identify_dynamic_map
            
            q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, num_q_clusters)
            k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, num_k_clusters)
            
            dynamic_map = identify_dynamic_map(
                qcentroids.view(cfg, num_heads, num_q_clusters, dim),
                kcentroids.view(cfg, num_heads, num_k_clusters, dim),
                q_cluster_sizes,
                k_cluster_sizes,
                self.top_p_kmeans,
                min_kc_ratio,
            )
          

            q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
            k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
            v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)
            
            return (q_permuted, k_permuted, v_permuted, dynamic_map, 
                    qcluster_sizes.view(cfg, num_heads, num_q_clusters),
                    kcluster_sizes.view(cfg, num_heads, num_k_clusters), 
                    q_sorted_indices)
                    
        elif self.use_global_constraints:
            from ...co_clustering import identify_dynamic_map_global
            
            if timestep is not None:
                if isinstance(timestep, torch.Tensor):
                    if timestep.numel() > 0:
                        current_timestep = timestep[0].item()
                    else:
                        current_timestep = 0
                else:
                    current_timestep = timestep
            else:
                current_timestep = 0
            q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
            k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)
            
            dynamic_map = identify_dynamic_map_global(
                qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
                kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
                q_cluster_sizes,
                k_cluster_sizes,
                self.top_p_kmeans,
                min_kc_ratio,
                key_tokens=key,
                k_labels=klabels,
                num_frame=self.num_frame,
                frame_size=self.frame_size,
                context_length=self.context_length,
                timestep=current_timestep,
                layer_idx=self.layer_idx,
                num_layers=self.num_layers,
                lambda_schedule=self.lambda_schedule,
                diverse_top_p_k=self.diverse_top_p_k,
            )
        else:
            from ...co_clustering import identify_dynamic_map
            
            q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
            k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)
            
            dynamic_map = identify_dynamic_map(
                qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
                kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
                q_cluster_sizes,
                k_cluster_sizes,
                self.top_p_kmeans,
                min_kc_ratio,
            )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(
            value, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        return (q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices)

    def flashinfer_attention(self, query, key, value):

        cfg, num_heads, seq_len, dim = query.size()

        query = query.flatten(0, 1).permute(1, 0, 2)
        key = key.flatten(0, 1).permute(1, 0, 2)
        value = value.flatten(0, 1).permute(1, 0, 2)

        o, o_lse = flashinfer.single_prefill_with_kv_cache(
            query,
            key,
            value,
            causal=False,
            return_lse=True,
        )

        o = o.permute(1, 0, 2).reshape(cfg, num_heads, seq_len, dim)

        return o

    def attention_core_logic(self, query, key, value, timestep):
        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"

        # Prefer an explicit step index provided by the pipeline callback (see wan_t2v_inference.py).
        # Fallback to deriving it from the scheduler timestep if callback isn't used.
        cb_step = getattr(self, "_step_from_callback", None)
        if cb_step is not None:
            self.current_inference_step = cb_step
        elif timestep is not None:
            self.current_inference_step = self._get_inference_step(timestep)
        else:
            self.current_inference_step = None

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep is not None:
            if isinstance(timestep, torch.Tensor):
                if timestep.numel() > 0:
                    timestep_value = timestep[0].item()
                else:
                    timestep_value = None
            else:
                timestep_value = timestep
            
            if timestep_value is not None and timestep_value > self.first_times_fp:
                full_attention_flag = True

        if full_attention_flag:
            if self.zero_step_kmeans_init:
                video_length = self.num_frame * self.frame_size
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                self.kmeans_clustering(query_video, key_video, self.layer_idx)

            output_hidden_states = self.flash_attention(query, key, value)
            output_hidden_states = output_hidden_states.transpose(1, 2).flatten(2, 3)
            return output_hidden_states

        else:
            perm_result = self.semantic_aware_permutation(query, key, value, timestep)
            
            if isinstance(perm_result, tuple) and len(perm_result) == 7:
                q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = perm_result
                if self.enable_mem_save:
                    del query, key, value

                # Backend selection for block-sparse attention.
                # If you hit CUDA illegal memory access with the flashinfer path on some sparsity patterns,
                # you can switch to the Triton implementation for robustness:
                sparse_backend = os.environ.get("SVG_SPARSE_ATTN_BACKEND", "flashinfer").lower()
                if sparse_backend == "triton":
                    from ...co_clustering import dynamic_block_sparse_fwd_triton

                    output_permuted = dynamic_block_sparse_fwd_triton(
                        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s
                    )
                else:
                    output_permuted = dynamic_block_sparse_fwd_flashinfer(
                        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
                    )
                if self.enable_mem_save:
                    del q_perm, k_perm, v_perm

                attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)
                if self.enable_mem_save:
                    del output_permuted

                # Save time, layer, density information to logging file
                if self.logging_file is not None:
                    # 4. Calculate density
                    densities = density_calculation(dyn_map, qc_sz_s, kc_sz_s)

                    avg_density = densities.mean().item()
                    if timestep is not None:
                        if isinstance(timestep, torch.Tensor):
                            if timestep.numel() > 0:
                                timestep_value = timestep[0].item()
                            else:
                                timestep_value = 0
                        else:
                            timestep_value = timestep
                    else:
                        timestep_value = 0
                        
                    log_entry = {
                        "timestep": timestep_value,
                        "layer": self.layer_idx,
                        "avg_density": avg_density,
                        "density": densities.tolist(),
                    }


                    with open(self.logging_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                attn_output = attn_output.transpose(1, 2).flatten(2, 3)
                return attn_output
