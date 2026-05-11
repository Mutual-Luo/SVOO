import argparse
import os
import math
from copy import deepcopy

import torch
from diffusers import HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

from svoo.utils.seed import seed_everything
from svoo.utils.data import load_prompt_or_image
from svoo.utils.runtime import configure_cuda_linalg_backend
from svoo.models.hunyuan10.inference import replace_hunyuan10_flashattention, replace_hunyuan10_attention
from svoo.models.hunyuan10.utils import get_prompt_length
from svoo.models.hunyuan10.pipelines import Hunyuan10VideoPipelineWithCPUOffload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video with SVOO using HunyuanVideo 1.0")
    parser.add_argument("--model_id", type=str, default="tencent/HunyuanVideo", help="HunyuanVideo 1.0 model path or Hugging Face id")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Hunyuan10_Web", "T2V_Xingyang_Motion"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution of the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    parser.add_argument(
        "--pattern",
        type=str,
        default="SVOO",
        choices=["SVOO"],
        help="Run SVOO.",
    )
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")
    parser.add_argument("--enable_cpu_offload", action="store_true", help="Offload text encoder / VAE to CPU when possible (1.0 models only).")

    # SVOO specific
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for SVOO.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for SVOO.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in SVOO.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in SVOO.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for SVOO initialization.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in SVOO.")
    parser.add_argument("--zero_step_kmeans_init", action="store_true", help="Initialize the centroids for the first SVOO step, not after warmup.")
    parser.add_argument("--start_reuse_step", type=int, default=None, help="Start reusing clustering results from this 1-based step (None to disable).")
    parser.add_argument("--reuse_interval", type=int, default=1, help="Re-cluster every N steps after start_reuse_step.")
    parser.add_argument("--use_dynamic_min_kc_ratio", action="store_true", help="Use dynamic min_kc_ratio from CSV instead of fixed value.")
    parser.add_argument("--sparsity_csv_path", type=str, default="sparsity_profiles/sparsity_results.csv", help="Path to sparsity CSV file (Step,Layer,Head,Sparsity).")
    parser.add_argument("--dynamic_min_kc_ratio_min", type=float, default=None, help="Minimum threshold for dynamic min_kc_ratio (clips CSV values below this).")
    parser.add_argument("--dynamic_min_kc_ratio_max", type=float, default=None, help="Maximum threshold for dynamic min_kc_ratio (clips CSV values above this).")

    # Attention sparsity measurement
    parser.add_argument("--measure_attention_sparsity", type=int, default=0, help="Enable attention sparsity measurement (1=enabled, 0=disabled)")
    parser.add_argument("--sparsity_output_file", type=str, default="attention_sparsity.txt", help="Output file for attention sparsity statistics (txt format)")
    parser.add_argument("--sparsity_batch_size", type=int, default=0, help="Exact query chunk size for sparsity; 0 auto-selects the largest safe chunk")
    parser.add_argument("--sparsity_query_samples", type=int, default=0, help="Number of query rows to sample for sparsity estimation; 0 uses all queries")
    parser.add_argument("--sparsity_threshold", type=float, default=0.95, help="Attention coverage threshold for sparsity calculation (default: 0.95 means 95% coverage)")
    parser.add_argument("--sparsity_start_step", type=int, default=1, help="Start computing and printing sparsity from this inference step (default: 1 means from step 1)")

    args = parser.parse_args()

    seed_everything(args.seed)

    configure_cuda_linalg_backend()
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            exit(0)

    # Load model.
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.model_id == "tencent/HunyuanVideo":
        model_kwargs["revision"] = "refs/pr/18"

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", **model_kwargs
    )

    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

    pipe = Hunyuan10VideoPipelineWithCPUOffload.from_pretrained(
        args.model_id, transformer=transformer, scheduler=scheduler, **model_kwargs
    )
        
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    config = pipe.transformer.config

    # Resolve warmup thresholds.
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    total_layers = config.num_layers + config.num_single_layers
    
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * total_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001  # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    
    
    # Load prompt/image input.
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    if args.prompt is None:
        args.prompt = "A cat walks on the grass, realistic"

    if args.negative_prompt is None:
        args.negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    
    prompt_length = get_prompt_length(pipe, args.prompt)
    print(f"Prompt length: {prompt_length}")

    # Install SVOO attention.
    
    common_attn_kwargs = dict(
        first_layers_fp=args.first_layers_fp,
        first_times_fp=args.first_times_fp,
    )
    
    replace_hunyuan10_flashattention(pipe)
    replace_hunyuan10_attention(
        pipe,
        args.height,
        args.width,
        args.num_frames,
        prompt_length,
        **common_attn_kwargs,
        pattern="SAP",
        num_q_centroids=args.num_q_centroids,
        num_k_centroids=args.num_k_centroids,
        top_p_kmeans=args.top_p_kmeans,
        min_kc_ratio=args.min_kc_ratio,
        logging_file=args.logging_file,
        kmeans_iter_init=args.kmeans_iter_init,
        kmeans_iter_step=args.kmeans_iter_step,
        zero_step_kmeans_init=args.zero_step_kmeans_init,
        use_svoo=True,
        start_reuse_step=args.start_reuse_step,
        reuse_interval=args.reuse_interval,
        use_dynamic_min_kc_ratio=bool(args.use_dynamic_min_kc_ratio),
        sparsity_csv_path=args.sparsity_csv_path if args.use_dynamic_min_kc_ratio else None,
        dynamic_min_kc_ratio_min=args.dynamic_min_kc_ratio_min,
        dynamic_min_kc_ratio_max=args.dynamic_min_kc_ratio_max,
        measure_attention_sparsity=bool(args.measure_attention_sparsity),
        sparsity_output_file=args.sparsity_output_file,
        sparsity_batch_size=args.sparsity_batch_size,
        sparsity_query_samples=args.sparsity_query_samples,
        sparsity_threshold=args.sparsity_threshold,
        sparsity_start_step=args.sparsity_start_step,
    )
        

    # Generate video.
    # Pipeline callbacks expose completed steps; attention needs the next step index.
    def _set_attention_step(step_1_based: int):
        for blk in pipe.transformer.transformer_blocks:
            attn = getattr(blk, "attn", None)
            proc = getattr(attn, "processor", None) if attn is not None else None
            if proc is not None:
                setattr(proc, "_step_from_callback", step_1_based)
        for blk in pipe.transformer.single_transformer_blocks:
            attn = getattr(blk, "attn", None)
            proc = getattr(attn, "processor", None) if attn is not None else None
            if proc is not None:
                setattr(proc, "_step_from_callback", step_1_based)

    # Start from the first denoising step.
    _set_attention_step(1)

    def _on_step_end(pipe_obj, step_index: int, timestep: int, callback_kwargs: dict):
        # Set the step used by the next denoising iteration.
        _set_attention_step(step_index + 2)
        return callback_kwargs

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        callback_on_step_end=_on_step_end,
        cpu_offload=args.enable_cpu_offload,
    ).frames[0]

    # Ensure output directory exists.
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=24)
