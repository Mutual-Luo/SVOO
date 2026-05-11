import argparse
import math
import os
from copy import deepcopy

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from svoo.models.wan.inference import replace_wan_attention
from svoo.utils.data import load_prompt_or_image
from svoo.utils.runtime import configure_cuda_linalg_backend
from svoo.utils.seed import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image-to-video with SVOO using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--image_path", type=str, default=None, help="Path of image")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "I2V_Wan_Web"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution preset used when auto-computing height/width")
    parser.add_argument("--height", type=int, default=None, help="Target video height (overrides auto-computed height when set together with --width)")
    parser.add_argument("--width", type=int, default=None, help="Target video width (overrides auto-computed width when set together with --height)")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    parser.add_argument(
        "--pattern",
        type=str,
        default="SVOO",
        choices=["SVOO"],
        help="Run SVOO.",
    )
    parser.add_argument("--first_layers_fp", type=float, default=0.3, help="The percentage of timesteps to leave in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.03, help="The percentage of layers to leave in FP")

    # SVOO related
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for KMEANS_BLOCK.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for KMEANS_BLOCK.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in KMEANS_BLOCK.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in KMEANS_BLOCK.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for initialization in KMEANS_BLOCK.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in KMEANS_BLOCK.")
    parser.add_argument("--zero_step_kmeans_init", action="store_true", help="Initialize the centroids for the first SVOO step, not after warmup.")
    
    # Clustering reuse
    parser.add_argument("--start_reuse_step", type=int, default=None, help="Start reusing clustering results from this step (None to disable)")
    parser.add_argument("--reuse_interval", type=int, default=1, help="Re-cluster every N steps after start_reuse_step")
    
    # Dynamic min_kc_ratio
    parser.add_argument("--use_dynamic_min_kc_ratio", action="store_true", help="Use dynamic min_kc_ratio from CSV file instead of fixed value")
    parser.add_argument("--sparsity_csv_path", type=str, default="sparsity_profiles/sparsity_results.csv", help="Path to sparsity CSV file for dynamic min_kc_ratio")
    parser.add_argument("--dynamic_min_kc_ratio_min", type=float, default=None, help="Minimum threshold for dynamic min_kc_ratio (clips CSV values below this)")
    parser.add_argument("--dynamic_min_kc_ratio_max", type=float, default=None, help="Maximum threshold for dynamic min_kc_ratio (clips CSV values above this)")
    
    parser.add_argument("--cpu_offload", type=int, default=0, help="Offload text encoder / VAE to CPU when possible.")

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
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    pipe.to("cuda")
    
    config = pipe.transformer.config
    
    # Resolve warmup thresholds.
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)

    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * config.num_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001  # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    

    # Load prompt/image input.
    args.prompt, args.image_path = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, args.image_path)

    if args.prompt is not None:
        assert args.image_path is not None, "Image path must be provided"
        image = load_image(args.image_path)

        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

        def _round_to_multiple(value: int, step: int) -> int:
            if value <= 0:
                raise ValueError("height/width must be positive integers")
            return max(step, round(value / step) * step)

        custom_height = args.height
        custom_width = args.width
        if (custom_height is None) ^ (custom_width is None):
            raise ValueError("Both --height and --width must be provided together.")

        if custom_height is not None and custom_width is not None:
            args.height = _round_to_multiple(custom_height, mod_value)
            args.width = _round_to_multiple(custom_width, mod_value)
        else:
            max_area = 720 * 1280 if args.resolution == "720p" else 480 * 832
            aspect_ratio = image.height / image.width
            raw_height = round(np.sqrt(max_area * aspect_ratio))
            raw_width = round(np.sqrt(max_area / aspect_ratio))
            args.height = max(mod_value, raw_height // mod_value * mod_value)
            args.width = max(mod_value, raw_width // mod_value * mod_value)

        image = image.resize((args.width, args.height))
    else:
        raise ValueError("Prompt must be provided")

    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


    # Install SVOO attention.
    replace_wan_attention(
        pipe,
        args.height,
        args.width,
        args.num_frames,
        num_inference_steps=args.num_inference_steps,
        first_layers_fp=args.first_layers_fp,
        first_times_fp=args.first_times_fp,
        pattern="SAP",
        num_q_centroids=args.num_q_centroids,
        num_k_centroids=args.num_k_centroids,
        top_p_kmeans=args.top_p_kmeans,
        min_kc_ratio=args.min_kc_ratio,
        kmeans_iter_init=args.kmeans_iter_init,
        kmeans_iter_step=args.kmeans_iter_step,
        zero_step_kmeans_init=args.zero_step_kmeans_init,
        use_svoo=True,
        start_reuse_step=args.start_reuse_step,
        reuse_interval=args.reuse_interval,
        use_dynamic_min_kc_ratio=args.use_dynamic_min_kc_ratio,
        sparsity_csv_path=args.sparsity_csv_path if args.use_dynamic_min_kc_ratio else None,
        dynamic_min_kc_ratio_min=args.dynamic_min_kc_ratio_min,
        dynamic_min_kc_ratio_max=args.dynamic_min_kc_ratio_max,
        measure_attention_sparsity=bool(args.measure_attention_sparsity),
        sparsity_output_file=args.sparsity_output_file,
        sparsity_batch_size=args.sparsity_batch_size,
        sparsity_query_samples=args.sparsity_query_samples,
        sparsity_threshold=args.sparsity_threshold,
        sparsity_start_step=args.sparsity_start_step,
        cpu_offload=bool(args.cpu_offload),
    )

    # Generate video.
    # Pipeline callbacks expose completed steps; attention needs the next step index.
    def _set_attention_step(step_1_based: int):
        def _apply_blocks(blocks):
            for blk in blocks:
                setattr(blk, "_step_from_callback", step_1_based)
                for attn_name in ("attn1", "attn2"):
                    attn = getattr(blk, attn_name, None)
                    proc = getattr(attn, "processor", None) if attn is not None else None
                    if proc is not None:
                        setattr(proc, "_step_from_callback", step_1_based)

        _apply_blocks(pipe.transformer.blocks)
        if hasattr(pipe, "transformer_2"):
            if pipe.transformer_2 is not None:
                _apply_blocks(pipe.transformer_2.blocks)

    # Start from the first denoising step.
    _set_attention_step(1)

    def _on_step_end(pipe_obj, step_index: int, timestep: int, callback_kwargs: dict):
        # Set the step used by the next denoising iteration.
        _set_attention_step(step_index + 2)
        return callback_kwargs

    if "2.2" in args.model_id:
        output = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=3.5,
            guidance_scale_2=3.0,
            num_inference_steps=args.num_inference_steps,
            callback_on_step_end=_on_step_end,
        ).frames[0]
    else:
        output = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=5.0,
            num_inference_steps=args.num_inference_steps,
            callback_on_step_end=_on_step_end,
        ).frames[0]

    # Ensure output directory exists.
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=16)
