from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.hunyuan_video import pipeline_hunyuan_video as hv_base
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import (
    HunyuanVideoPipeline,
    HunyuanVideoPipelineOutput,
    retrieve_timesteps,
)

XLA_AVAILABLE = getattr(hv_base, "XLA_AVAILABLE", False)
xm = getattr(hv_base, "xm", None)


class Hunyuan10VideoPipelineWithCPUOffload(HunyuanVideoPipeline):
    _callback_tensor_inputs: Tuple[str, ...] = ("latents", "prompt_embeds")

    def _text_modules(self):
        modules = []
        if getattr(self, "text_encoder", None) is not None:
            modules.append(self.text_encoder)
        if getattr(self, "text_encoder_2", None) is not None:
            modules.append(self.text_encoder_2)
        return modules

    def _move_text_modules(self, device: Union[str, torch.device]):
        for module in self._text_modules():
            module.to(device)

    def _move_vae(self, device: Union[str, torch.device]):
        if getattr(self, "vae", None) is not None:
            self.vae.to(device)

    def set_cpu_offload(self):
        self._text_vae_offload = True

    def enable_text_vae_offload(self):
        self._move_text_modules("cpu")
        self._move_vae("cpu")

    def disable_text_vae_offload(self):
        self._text_vae_offload = False
        device = getattr(self, "_execution_device", "cuda")
        self._move_text_modules(device)
        self._move_vae(device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = hv_base.DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        cpu_offload: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` will
                be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            height (`int`, defaults to `720`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `129`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            guidance_scale (`float`, defaults to `6.0`):
                Embedded guidance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifier-free guidance for `guidance_scale` > 1.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A `torch.Generator` to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. If not provided, pooled text embeddings will be generated from
                `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. If not provided, negative_prompt_embeds will be generated from
                `negative_prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. If not provided, pooled negative_prompt_embeds will be
                generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `HunyuanVideoPipelineOutput` instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or callback called at the end of each denoising step during the inference.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            prompt_template (`Dict`, *optional*):
                Template dict used for prompt formatting.
            max_sequence_length (`int`, *optional*):
                Maximum text length fed into the text encoder.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if cpu_offload:
            self._text_vae_offload = True
        offload_enabled = getattr(self, "_text_vae_offload", False)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.transformer.device
        if offload_enabled:
            self._move_text_modules(device)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        transformer_dtype = self.transformer.dtype
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(transformer_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)

        if offload_enabled:
            self._move_text_modules("cpu")
            self._move_vae("cpu")

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_attention_mask=negative_prompt_attention_mask,
                            pooled_projections=negative_pooled_prompt_embeds,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            if offload_enabled:
                self._move_vae(device)
                self.transformer.to("cpu")
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            if offload_enabled:
                self._move_vae("cpu")
                self.transformer.to(device)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
