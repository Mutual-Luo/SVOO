import os

import torch

from .attention import (
    WanAttn_SAPAttn_Processor,
    WanAttn_SVGAttn_Processor2_0,
    prepare_flashinfer_attention,
    prepare_flexattention,
)
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width
from ...utils.triton_warmup import warmup_svoo_triton_kernels


def replace_wan_attention(
    pipe,
    height,
    width,
    num_frames,
    first_layers_fp,
    first_times_fp,
    num_inference_steps=None,
    attention_backend="flexattn",
    pattern="SAP",
    # SVG specific, but provide defaults for general call signature
    num_sampled_rows=64,
    sample_mse_max_row=10000,
    sparsity=0.25,
    # Pattern dispatcher and KMEANS_BLOCK specific args
    num_q_centroids=None,
    num_k_centroids=None,
    top_p_kmeans=None,
    min_kc_ratio=0,
    kmeans_iter_init=0,
    kmeans_iter_step=0,
    zero_step_kmeans_init=False,
    # Global constraints switch
    use_global_constraints=False,  # Default False, use original method
    lambda_schedule="linear",  # Weight schedule strategy
    diverse_top_p_k=0.0,  # Diversity selection ratio; if >0 use two-stage selection
    # Routing Transformer strategy (hierarchical clustering)
    use_routing_transformer_strategy=False,  # Whether to use hierarchical Routing Transformer strategy
    mq1=None,  # Hierarchical clustering: number of query buckets (level 1)
    mk1=None,  # Hierarchical clustering: number of key buckets (level 1)
    mq2=None,  # Hierarchical clustering: number of clusters per query bucket (level 2)
    mk2=None,  # Hierarchical clustering: number of clusters per key bucket (level 2)
    # SVOO switch
    use_svoo=True,
    # Clustering reuse parameters
    start_reuse_step=None,  # Step from which to start reusing cluster results (None = disabled)
    reuse_interval=1,  # Re-cluster every N steps after start_reuse_step
    # Attention sparsity measurement
    measure_attention_sparsity=False,  # Whether to measure attention sparsity
    sparsity_output_file="attention_sparsity.txt",  # Output file for sparsity stats (txt)
    sparsity_batch_size=0,  # 0 auto-selects the largest safe exact query chunk
    sparsity_query_samples=0,  # 0 uses all query rows; >0 estimates from sampled rows
    sparsity_threshold=0.95,  # Attention coverage threshold (0.95 = 95% coverage)
    sparsity_start_step=1,  # First inference step to compute and log sparsity
    # Dynamic min_kc_ratio parameters
    use_dynamic_min_kc_ratio=False,  # Use dynamic min_kc_ratio loaded from CSV
    sparsity_csv_path=None,  # Path to sparsity CSV for dynamic min_kc_ratio
    dynamic_min_kc_ratio_min=None,  # Lower clipping threshold for dynamic min_kc_ratio
    dynamic_min_kc_ratio_max=None,  # Upper clipping threshold for dynamic min_kc_ratio
    cpu_offload=False,
):
    if num_inference_steps is None:
        num_inference_steps = 50
    if pattern != "SAP" or not use_svoo:
        raise ValueError("Only SVOO is supported in this release.")

    context_length = 0  # This seems to be 0 for I2V in SVG
    num_frame_patches = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_patches_one_frame = int(height // mod_value) * int(width // mod_value)

    dtype = torch.bfloat16  # Or pipe.dtype
    device = pipe.device
    num_attention_heads = pipe.transformer.config.num_attention_heads
    attention_head_dim = pipe.transformer.config.attention_head_dim

    replace_sparse_forward(cpu_offload)  # Assuming this is a general patch; if not, it might need to be conditional.

    num_layers = len(pipe.transformer.blocks)

    if pattern == "SVG":
        AttnModule = WanAttn_SVGAttn_Processor2_0
        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.sparsity = sparsity

        masks = ["spatial", "temporal"]
        AttnModule.attention_masks = [
            get_attention_mask(
                mask_name, sample_mse_max_row, context_length, num_frame_patches, frame_patches_one_frame
            )
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        multiplier = diag_width = sparsity_to_width(
            sparsity, context_length, num_frame_patches, frame_patches_one_frame
        )

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        if attention_backend == "flexattn":
            block_mask = prepare_flexattention(
                1,
                num_attention_heads,
                attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.block_mask = block_mask


        elif attention_backend == "flashinfer":
            temporal_mask_metadata = prepare_flashinfer_attention(
                1,
                num_attention_heads,
                attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.temporal_mask_metadata = temporal_mask_metadata

        else:
            raise ValueError(f"Attention backend {attention_backend} not supported")

        # Helper to replace attention processors for Wan2.2
        def replace_blocks_attn(transformer_model, num_layers):
            if transformer_model is None:
                return
                
            for layer_idx, m in enumerate(transformer_model.blocks):
                # Wan models use attn1 (Self-Attention)
                if hasattr(m, "attn1") and hasattr(m.attn1, "processor"):
                    current_processor = AttnModule(layer_idx=layer_idx)
                    current_processor.num_layers = num_layers
                    m.attn1.set_processor(current_processor)

        # 1. Get total number of layers (assuming both transformers have the same depth)
        num_layers = len(pipe.transformer.blocks)

        # 2. Replace first Transformer (High-noise Expert)
        replace_blocks_attn(pipe.transformer, num_layers)

        # 3. Replace second Transformer (Low-noise Expert, if present)
        if hasattr(pipe, "transformer_2"):
            print("Detected Wan2.2 transformer_2 (Low-noise Expert), applying Sparse Attention...")
            replace_blocks_attn(pipe.transformer_2, num_layers)
    
    elif pattern == "SPG":
        AttnModule = WanAttn_SVGAttn_Processor2_0
        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.sparsity = sparsity
        AttnModule.use_spargeattn = True  # Use spargeattn method

        masks = ["spatial", "temporal"]
        AttnModule.attention_masks = [
            get_attention_mask(
                mask_name, sample_mse_max_row, context_length, num_frame_patches, frame_patches_one_frame
            )
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        multiplier = diag_width = sparsity_to_width(
            sparsity, context_length, num_frame_patches, frame_patches_one_frame
        )

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        if attention_backend == "flexattn":
            block_mask = prepare_flexattention(
                1,
                num_attention_heads,
                attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.block_mask = block_mask


        elif attention_backend == "flashinfer":
            temporal_mask_metadata = prepare_flashinfer_attention(
                1,
                num_attention_heads,
                attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.temporal_mask_metadata = temporal_mask_metadata

        else:
            raise ValueError(f"Attention backend {attention_backend} not supported")

        # Helper to replace attention processors for Wan2.2
        def replace_blocks_attn(transformer_model, num_layers):
            if transformer_model is None:
                return
                
            for layer_idx, m in enumerate(transformer_model.blocks):
                # Wan models use attn1 (Self-Attention)
                if hasattr(m, "attn1") and hasattr(m.attn1, "processor"):
                    current_processor = AttnModule(layer_idx=layer_idx)
                    current_processor.num_layers = num_layers
                    m.attn1.set_processor(current_processor)

        # 1. Get total number of layers (assuming both transformers have the same depth)
        num_layers = len(pipe.transformer.blocks)

        # 2. Replace first Transformer (High-noise Expert)
        replace_blocks_attn(pipe.transformer, num_layers)

        # 3. Replace second Transformer (Low-noise Expert, if present)
        if hasattr(pipe, "transformer_2"):
            print("Detected Wan2.2 transformer_2 (Low-noise Expert), applying Sparse Attention...")
            replace_blocks_attn(pipe.transformer_2, num_layers)
    

    elif pattern == "SAP":

        # Pass K-means specific parameters to the processor's constructor or set them as attributes
        # The processor itself will handle the K-means logic internally

        AttnModule = WanAttn_SAPAttn_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.num_layers = num_layers
        AttnModule.kmeans_iter_init = kmeans_iter_init
        AttnModule.kmeans_iter_step = kmeans_iter_step
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init
        
        # Global constraints switch
        AttnModule.use_global_constraints = use_global_constraints
        AttnModule.lambda_schedule = lambda_schedule
        AttnModule.diverse_top_p_k = diverse_top_p_k
        
        # Routing Transformer strategy switch (hierarchical clustering)
        AttnModule.use_routing_transformer_strategy = use_routing_transformer_strategy
        AttnModule.mq1 = mq1
        AttnModule.mk1 = mk1
        AttnModule.mq2 = mq2
        AttnModule.mk2 = mk2
        
        # SVOO switch
        AttnModule.use_svoo = use_svoo
        
        # Clustering reuse parameters
        AttnModule.start_reuse_step = start_reuse_step
        AttnModule.reuse_interval = reuse_interval
        
        # Dynamic min_kc_ratio switch
        if use_dynamic_min_kc_ratio and sparsity_csv_path:
            try:
                from ...sparsity import SparsityLookup
                AttnModule.sparsity_lookup = SparsityLookup(sparsity_csv_path)
                AttnModule.use_dynamic_min_kc_ratio = True
                AttnModule.dynamic_min_kc_ratio_min = dynamic_min_kc_ratio_min
                AttnModule.dynamic_min_kc_ratio_max = dynamic_min_kc_ratio_max
            except Exception as e:
                AttnModule.use_dynamic_min_kc_ratio = False
                AttnModule.sparsity_lookup = None
        else:
            AttnModule.use_dynamic_min_kc_ratio = False
            AttnModule.sparsity_lookup = None
        
        # Attention sparsity measurement parameters
        AttnModule.measure_attention_sparsity = measure_attention_sparsity
        AttnModule.sparsity_output_file = sparsity_output_file
        AttnModule.sparsity_batch_size = sparsity_batch_size
        AttnModule.sparsity_query_samples = sparsity_query_samples
        AttnModule.sparsity_threshold = sparsity_threshold
        AttnModule.sparsity_start_step = sparsity_start_step
        
        # If resume is enabled, keep existing raw sparsity entries so completed
        # layer/step records can be skipped by has_completed_sparsity_entry.
        if measure_attention_sparsity and sparsity_output_file:
            os.makedirs(os.path.dirname(sparsity_output_file) if os.path.dirname(sparsity_output_file) else ".", exist_ok=True)
            if os.environ.get("SVOO_SPARSITY_RESUME", "0") in ("0", "", "false", "False"):
                with open(sparsity_output_file, "w", encoding="utf-8") as f:
                    f.write("")
            # Reset inference step tracker
            AttnModule._inference_step_counter = 0
            AttnModule._seen_timesteps = set()
            AttnModule._current_internal_timestep = None
        
        # Create and register a processor for each attention module
        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                current_processor.num_layers = num_layers
                # Ensure each instance has sparsity measurement attributes set
                current_processor.measure_attention_sparsity = measure_attention_sparsity
                current_processor.sparsity_output_file = sparsity_output_file
                current_processor.sparsity_batch_size = sparsity_batch_size
                current_processor.sparsity_query_samples = sparsity_query_samples
                current_processor.sparsity_threshold = sparsity_threshold
                current_processor.sparsity_start_step = sparsity_start_step
                current_processor.transformer_stage = "high-noise transformer"
                m.attn1.set_processor(current_processor)
        if getattr(pipe, "transformer_2", None) is not None:
            for layer_idx, m in enumerate(pipe.transformer_2.blocks):
                if hasattr(m.attn1, "processor"):
                    current_processor = AttnModule(layer_idx=layer_idx)
                    current_processor.num_layers = num_layers
                    # Ensure each instance has sparsity measurement attributes set
                    current_processor.measure_attention_sparsity = measure_attention_sparsity
                    current_processor.sparsity_output_file = sparsity_output_file
                    current_processor.sparsity_batch_size = sparsity_batch_size
                    current_processor.sparsity_query_samples = sparsity_query_samples
                    current_processor.sparsity_threshold = sparsity_threshold
                    current_processor.sparsity_start_step = sparsity_start_step
                    current_processor.transformer_stage = "low-noise transformer"
                    m.attn1.set_processor(current_processor)

        # Precompile Triton kernels before inference progress starts.
        try:
            _attn1 = pipe.transformer.blocks[0].attn1
            norm_dim = (
                _attn1.norm_q.weight.shape[-1]
                if getattr(_attn1, "norm_q", None) is not None
                else num_attention_heads * attention_head_dim
            )
        except (AttributeError, IndexError):
            norm_dim = num_attention_heads * attention_head_dim

        warmup_svoo_triton_kernels(
            model_name="Wan",
            num_head=num_attention_heads,
            head_dim=attention_head_dim,
            hidden_dim=norm_dim,
            seq_len=num_frame_patches * frame_patches_one_frame,
            num_q_centroids=num_q_centroids,
            num_k_centroids=num_k_centroids,
            dtype=dtype,
            device=device,
            cfg_values=(1,),
            include_rmsnorm=True,
            include_wan_block_kernels=True,
            include_flashinfer_sparse=os.environ.get("SVG_SPARSE_ATTN_BACKEND", "flashinfer").lower() != "triton",
        )

    elif pattern == "dense":
        AttnModule = WanAttn_SVGAttn_Processor2_0
        AttnModule.attention_masks = None
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame
        AttnModule.measure_attention_sparsity = measure_attention_sparsity
        AttnModule.sparsity_output_file = sparsity_output_file
        AttnModule.sparsity_batch_size = sparsity_batch_size
        AttnModule.sparsity_query_samples = sparsity_query_samples
        AttnModule.sparsity_threshold = sparsity_threshold
        AttnModule.sparsity_start_step = sparsity_start_step
        
        if measure_attention_sparsity and sparsity_output_file:
            os.makedirs(os.path.dirname(sparsity_output_file) if os.path.dirname(sparsity_output_file) else ".", exist_ok=True)
            if os.environ.get("SVOO_SPARSITY_RESUME", "0") in ("0", "", "false", "False"):
                with open(sparsity_output_file, "w") as f:
                    f.write("")
            AttnModule._inference_step_counter = 0
            AttnModule._seen_timesteps = set()
            AttnModule._current_internal_timestep = None
        
        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                current_processor.num_layers = num_layers
                current_processor.measure_attention_sparsity = measure_attention_sparsity
                current_processor.sparsity_output_file = sparsity_output_file
                current_processor.sparsity_batch_size = sparsity_batch_size
                current_processor.sparsity_query_samples = sparsity_query_samples
                current_processor.sparsity_threshold = sparsity_threshold
                current_processor.sparsity_start_step = sparsity_start_step
                m.attn1.set_processor(current_processor)
        if getattr(pipe, "transformer_2", None) is not None:
            for layer_idx, m in enumerate(pipe.transformer_2.blocks):
                if hasattr(m, "attn1") and hasattr(m.attn1, "processor"):
                    current_processor = AttnModule(layer_idx=layer_idx)
                    current_processor.num_layers = num_layers
                    current_processor.measure_attention_sparsity = measure_attention_sparsity
                    current_processor.sparsity_output_file = sparsity_output_file
                    current_processor.sparsity_batch_size = sparsity_batch_size
                    current_processor.sparsity_query_samples = sparsity_query_samples
                    current_processor.sparsity_threshold = sparsity_threshold
                    current_processor.sparsity_start_step = sparsity_start_step
                    m.attn1.set_processor(current_processor)
    
    else:
        raise ValueError(f"Pattern '{pattern}' not supported")

    print(f"Attention processors replaced with {pattern} pattern.")
