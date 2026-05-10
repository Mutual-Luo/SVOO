from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import PIL.Image
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.hunyuan_video import pipeline_hunyuan_video_image2video as hv_i2v_base
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video_image2video import (
    DEFAULT_PROMPT_TEMPLATE,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoPipelineOutput,
    retrieve_timesteps,
)
from diffusers.utils import replace_example_docstring

XLA_AVAILABLE = getattr(hv_i2v_base, "XLA_AVAILABLE", False)
xm = getattr(hv_i2v_base, "xm", None)


class Hunyuan10VideoImageToVideoPipelineWithCPUOffload(HunyuanVideoImageToVideoPipeline):
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
    @replace_example_docstring(hv_i2v_base.EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PIL.Image.Image,
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
        guidance_scale: float = 1.0,
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
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        image_embed_interleave: Optional[int] = None,
        cpu_offload: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
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
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            guidance_scale (`float`, defaults to `1.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. Note that the only available
                HunyuanVideo model is CFG-distilled, which means that traditional guidance between unconditional and
                conditional latent is not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            cpu_offload (`bool`, *optional*, defaults to `False`):
                Whether to keep text encoders and VAE on CPU outside their usage to reduce peak GPU memory.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
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
            true_cfg_scale,
            guidance_scale,
        )

        image_condition_type = self.transformer.config.image_condition_type
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        image_embed_interleave = (
            image_embed_interleave
            if image_embed_interleave is not None
            else (
                2 if image_condition_type == "latent_concat" else 4 if image_condition_type == "token_replace" else 1
            )
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Prepare latent variables
        vae_dtype = self.vae.dtype
        image_tensor = self.video_processor.preprocess(image, height, width).to(device, vae_dtype)

        if image_condition_type == "latent_concat":
            num_channels_latents = (self.transformer.config.in_channels - 1) // 2
        elif image_condition_type == "token_replace":
            num_channels_latents = self.transformer.config.in_channels

        if offload_enabled:
            self._move_vae(device)
        latents, image_latents = self.prepare_latents(
            image_tensor,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            image_condition_type,
        )
        if image_condition_type == "latent_concat":
            image_latents[:, :, 1:] = 0
            mask = image_latents.new_ones(image_latents.shape[0], 1, *image_latents.shape[2:])
            mask[:, :, 1:] = 0
        if offload_enabled:
            self._move_vae("cpu")

        # 4. Encode input prompt
        transformer_dtype = self.transformer.dtype
        if offload_enabled:
            self._move_text_modules(device)
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            image=image,
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
            image_embed_interleave=image_embed_interleave,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        if do_true_cfg:
            black_image = PIL.Image.new("RGB", (width, height), 0)
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                image=black_image,
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

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 6. Prepare guidance condition
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = (
                torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0
            )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if image_condition_type == "latent_concat":
                    latent_model_input = torch.cat([latents, image_latents, mask], dim=1).to(transformer_dtype)
                elif image_condition_type == "token_replace":
                    latent_model_input = torch.cat([image_latents, latents[:, :, 1:]], dim=2).to(transformer_dtype)

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
                if image_condition_type == "latent_concat":
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                elif image_condition_type == "token_replace":
                    latents = latents = self.scheduler.step(
                        noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False
                    )[0]
                    latents = torch.cat([image_latents, latents], dim=2)

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
            latents = latents.to(self.vae.dtype) / self.vae_scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            if image_condition_type == "latent_concat":
                video = video[:, :, 4:, :, :]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            if offload_enabled:
                self._move_vae("cpu")
                self.transformer.to(device)
        else:
            if image_condition_type == "latent_concat":
                video = latents[:, :, 1:, :, :]
            else:
                video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
