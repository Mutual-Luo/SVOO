import torch

from .attention import (
    Hunyuan_SAPAttn_Processor2_0,
    Hunyuan_SVGAttn_Processor2_0,
    HunyuanVideoAttnProcessor2_0_FlashAttention,
    prepare_flexattention,
)
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width
from ...utils.triton_warmup import warmup_svoo_triton_kernels


def replace_hunyuan10_flashattention(pipe):
    """
    Replace the FSDP + masked attention with flash attention + varlen. Crucial for inference efficiency.
    """
    for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
        self_attn = m.attn
        self_attn.processor = HunyuanVideoAttnProcessor2_0_FlashAttention(layer_idx=layer_idx)
        print(f"Replaced FlashAttention implementation in double stream transformer block {layer_idx}")

    for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
        self_attn = m.attn
        self_attn.processor = HunyuanVideoAttnProcessor2_0_FlashAttention(
            layer_idx=layer_idx + len(pipe.transformer.transformer_blocks)
        )
        print(f"Replaced FlashAttention implementation in single stream transformer block {layer_idx}")


def replace_hunyuan10_attention(
    pipe,
    height,
    width,
    num_frames,
    prompt_length,
    first_layers_fp,
    first_times_fp,
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
    context_length=256,
    # SVOO / reuse / dynamic min_kc_ratio (SAP)
    use_svoo=True,
    start_reuse_step=None,
    reuse_interval=1,
    use_dynamic_min_kc_ratio=False,
    sparsity_csv_path=None,
    dynamic_min_kc_ratio_min=None,
    dynamic_min_kc_ratio_max=None,
    measure_attention_sparsity=False,
    sparsity_output_file="attention_sparsity.txt",
    sparsity_batch_size=0,
    sparsity_query_samples=0,
    sparsity_threshold=0.95,
    sparsity_start_step=1,
):
    if pattern != "SAP" or not use_svoo:
        raise ValueError("Only SVOO is supported in this release.")

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame = 256, 1 + num_frames // 4
    frame_size = height * width // 256

    if pattern == "SVG":
        masks = ["spatial", "temporal"]

        # Calculation
        spatial_width = temporal_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

        print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {sparsity}")

        AttnModule = Hunyuan_SVGAttn_Processor2_0

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.attention_masks = [
            get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.use_spargeattn = False

        block_mask = prepare_flexattention(
            cfg_size,
            num_head,
            head_dim,
            dtype,
            device,
            context_length,
            prompt_length,
            num_frame,
            frame_size,
            diag_width=spatial_width,
            multiplier=temporal_width,
        )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()


        AttnModule.measure_attention_sparsity = measure_attention_sparsity
        AttnModule.sparsity_output_file = sparsity_output_file
        AttnModule.sparsity_batch_size = sparsity_batch_size
        AttnModule.sparsity_query_samples = sparsity_query_samples
        AttnModule.sparsity_threshold = sparsity_threshold
        AttnModule.sparsity_start_step = sparsity_start_step
        
        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            proc = AttnModule(layer_idx=layer_idx)
            proc.measure_attention_sparsity = measure_attention_sparsity
            proc.sparsity_output_file = sparsity_output_file
            proc.sparsity_batch_size = sparsity_batch_size
            proc.sparsity_query_samples = sparsity_query_samples
            proc.sparsity_threshold = sparsity_threshold
            proc.sparsity_start_step = sparsity_start_step
            self_attn.processor = proc
            print(f"Replaced Sparse VideoGen block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            proc = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            proc.measure_attention_sparsity = measure_attention_sparsity
            proc.sparsity_output_file = sparsity_output_file
            proc.sparsity_batch_size = sparsity_batch_size
            proc.sparsity_query_samples = sparsity_query_samples
            proc.sparsity_threshold = sparsity_threshold
            proc.sparsity_start_step = sparsity_start_step
            self_attn.processor = proc
            print(
                f"Replaced Sparse VideoGen block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    elif pattern in ["SAP"]:

        # Pass K-means specific parameters to the processor's constructor or set them as attributes
        # The processor itself will handle the K-means logic internally
        AttnModule = Hunyuan_SAPAttn_Processor2_0

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.kmeans_iter_init = kmeans_iter_init
        AttnModule.kmeans_iter_step = kmeans_iter_step
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init

        # SVOO / reuse
        AttnModule.use_svoo = bool(use_svoo)
        AttnModule.start_reuse_step = start_reuse_step
        AttnModule.reuse_interval = max(1, int(reuse_interval)) if reuse_interval is not None else 1

        # Dynamic min_kc_ratio
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
                AttnModule.dynamic_min_kc_ratio_min = None
                AttnModule.dynamic_min_kc_ratio_max = None
        else:
            AttnModule.use_dynamic_min_kc_ratio = False
            AttnModule.sparsity_lookup = None
            AttnModule.dynamic_min_kc_ratio_min = None
            AttnModule.dynamic_min_kc_ratio_max = None

        replace_sparse_forward()
        
        AttnModule.measure_attention_sparsity = measure_attention_sparsity
        AttnModule.sparsity_output_file = sparsity_output_file
        AttnModule.sparsity_batch_size = sparsity_batch_size
        AttnModule.sparsity_query_samples = sparsity_query_samples
        AttnModule.sparsity_threshold = sparsity_threshold
        AttnModule.sparsity_start_step = sparsity_start_step

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            proc = AttnModule(layer_idx=layer_idx)
            proc.measure_attention_sparsity = measure_attention_sparsity
            proc.sparsity_output_file = sparsity_output_file
            proc.sparsity_batch_size = sparsity_batch_size
            proc.sparsity_query_samples = sparsity_query_samples
            proc.sparsity_threshold = sparsity_threshold
            proc.sparsity_start_step = sparsity_start_step
            self_attn.processor = proc
            print(f"Replaced Semantic Aware Permutation block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            proc = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            proc.measure_attention_sparsity = measure_attention_sparsity
            proc.sparsity_output_file = sparsity_output_file
            proc.sparsity_batch_size = sparsity_batch_size
            proc.sparsity_query_samples = sparsity_query_samples
            proc.sparsity_threshold = sparsity_threshold
            proc.sparsity_start_step = sparsity_start_step
            self_attn.processor = proc
            print(
                f"Replaced Semantic Aware Permutation block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

        warmup_svoo_triton_kernels(
            model_name="HunyuanVideo 1.0",
            num_head=num_head,
            head_dim=head_dim,
            hidden_dim=num_head * head_dim,
            seq_len=num_frame * frame_size,
            num_q_centroids=num_q_centroids,
            num_k_centroids=num_k_centroids,
            dtype=dtype,
            device=device,
            cfg_values=(1,),
            inverse_seq_len=context_length + num_frame * frame_size,
            include_flashinfer_sparse=True,
        )

    elif pattern == "SPG":
        AttnModule = Hunyuan_SVGAttn_Processor2_0
        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.sparsity = sparsity
        AttnModule.use_spargeattn = True

        masks = ["spatial", "temporal"]
        AttnModule.attention_masks = [
            get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        # Calculation
        spatial_width = temporal_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

        print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {sparsity}")

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        block_mask = prepare_flexattention(
            cfg_size,
            num_head,
            head_dim,
            dtype,
            device,
            context_length,
            prompt_length,
            num_frame,
            frame_size,
            diag_width=spatial_width,
            multiplier=temporal_width,
        )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()

        
        AttnModule.measure_attention_sparsity = measure_attention_sparsity
        AttnModule.sparsity_output_file = sparsity_output_file
        AttnModule.sparsity_batch_size = sparsity_batch_size
        AttnModule.sparsity_query_samples = sparsity_query_samples
        AttnModule.sparsity_threshold = sparsity_threshold
        AttnModule.sparsity_start_step = sparsity_start_step

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            proc = AttnModule(layer_idx=layer_idx)
            proc.measure_attention_sparsity = measure_attention_sparsity
            proc.sparsity_output_file = sparsity_output_file
            proc.sparsity_batch_size = sparsity_batch_size
            proc.sparsity_query_samples = sparsity_query_samples
            proc.sparsity_threshold = sparsity_threshold
            proc.sparsity_start_step = sparsity_start_step
            self_attn.processor = proc
            print(f"Replaced SPG (SpargeAttn) block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            proc = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            proc.measure_attention_sparsity = measure_attention_sparsity
            proc.sparsity_output_file = sparsity_output_file
            proc.sparsity_batch_size = sparsity_batch_size
            proc.sparsity_query_samples = sparsity_query_samples
            proc.sparsity_threshold = sparsity_threshold
            proc.sparsity_start_step = sparsity_start_step
            self_attn.processor = proc
            print(
                f"Replaced SPG (SpargeAttn) block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    else:
        assert pattern == "dense", f"Invalid pattern: {pattern}"
