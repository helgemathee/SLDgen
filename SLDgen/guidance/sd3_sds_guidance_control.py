# Inspired by https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderTiny,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3ControlNetPipeline,
)
from easydict import EasyDict
from jaxtyping import Float
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from .sd3_controlnet import create_condition, get_sd35_medium_controlnet


class SD3GuidanceControl(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.cfg = EasyDict(
            {
                "pretrained_model_name_or_path": args.diffusion_model,
                "pretrained_vae": args.vae_path,
                "enable_sequential_cpu_offload": False,  # same as below
                "enable_attention_slicing": False,  # reduces memory usage and performance if true
                "enable_channels_last_format": True,
                "guidance_scale": args.diffusion_guidance_scale,
                "half_precision_weights": True,  # float16  if True else float32 for model weights
                "min_step_percent": 0.02,
                "max_step_percent": 0.98,
                "weighting_strategy": "uniform",
                "multisteps": args.multisteps,
            }
        )
        self.configure()

    def configure(self) -> None:
        """Build the diffusion pipeline, encoders, scheduler, and text embeddings."""
        # Load the ControlNet model for the selected condition.
        controlnet = get_sd35_medium_controlnet(self.args.condition, self.device)
        print(f"\t\tUsing {self.args.condition} ControlNet for condition.", flush=True)

        self.weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32

        # Create model (pipeline)
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.args.diffusion_model,
            controlnet=controlnet,
            torch_dtype=self.weights_dtype,
            use_safetensors=True,
        )

        if self.cfg.pretrained_vae == "madebyollin/taesd3":
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                self.cfg.pretrained_vae, torch_dtype=self.weights_dtype
            ).to(self.device)
            self.pipe.vae.config.shift_factor = 0.0
            self.pipe.vae.decoder.to("cpu")

        # Load LoRA model if provided
        if self.args.lora_model is not None and self.args.lora_weight > 0.0:
            print(
                f"\t\tLoading LoRA model from {self.args.lora_model} "
                f"with weight {self.args.lora_weight}.",
                flush=True,
            )
            self.pipe.load_lora_weights(self.args.lora_model, adapter_name="sld")
            self.pipe.fuse_lora(lora_scale=self.args.lora_weight)

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.transformer.to(memory_format=torch.channels_last)

        # Handle the transformer model
        self.transformer = self.pipe.transformer.to(self.device)
        self.transformer = self.transformer.eval()

        for p in self.transformer.parameters():
            p.requires_grad_(False)

        # Precompute the conditioning image once since it is reused at every step.
        self.conditioning_scale = self.args.conditioning_scale
        self.control_image = self.create_control_image(self.args.condition)

        # Cache the scheduler and a timestep range used for SDS sampling.
        self.scheduler: FlowMatchEulerDiscreteScheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.intervals = torch.tensor(
            [
                self.min_step + (i * (self.max_step - self.min_step) // self.cfg.multisteps)
                for i in range(self.cfg.multisteps + 1)
            ],
            device=self.device,
        )
        self.sigmas = self.scheduler.sigmas.to(device=self.device, dtype=self.weights_dtype)
        self.timesteps = self.scheduler.timesteps.to(device=self.device)

        # Keep the tokenizer and text encoder available for prompt encoding.
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        if self.args.caption == "":
            self.create_caption()
        print(f"\t\tText prompt for the SDS loss: {self.args.caption}.", flush=True)

        # Pre-encode the text prompt and negative prompt for classifier-free guidance.
        self.transformer = self.transformer.to("cpu")
        self.pipe.text_encoder.to(self.device)
        self.pipe.text_encoder_2.to(self.device)
        self.pipe.text_encoder_3.to(self.device)
        with torch.no_grad():
            prompt_embeds_list = self.pipe.encode_prompt(
                prompt=self.args.caption,
                prompt_2=self.args.caption,
                prompt_3=self.args.caption,
                negative_prompt=self.args.negative_caption,
                negative_prompt_2=self.args.negative_caption,
                negative_prompt_3=self.args.negative_caption,
                do_classifier_free_guidance=True,
            )
            self.prompt_embeds = prompt_embeds_list[0]
            self.negative_prompt_embeds = prompt_embeds_list[1]
            self.pooled_prompt_embeds = prompt_embeds_list[2]
            self.negative_pooled_prompt_embeds = prompt_embeds_list[3]

        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self.pipe.text_encoder_3.to("cpu")
        self.transformer = self.transformer.to(self.device)
        torch.cuda.empty_cache()

        print("\t\tLoaded Stable Diffusion 3.5 ControlNet pipeline and encoders.", flush=True)

    def create_control_image(self, condition_name):
        """Create and encode the ControlNet conditioning image."""
        condition = create_condition(self.args.input_image, condition_name)
        condition = self.create_masked_condition(condition)
        condition.save(f"{self.args.output_dir}/{condition_name}_condition.png")

        # Persist the conditioning image under a stable, documented name for reuse
        # as a partition label map (sld_partition.py --strategy labelmap). It is in
        # CANVAS SPACE at --render-size: create_condition runs on args.input_image,
        # which targets.py sets AFTER the --object-size-ratio rescale-and-center and
        # the resize to render_size (see targets.py: rescale_obj -> args.input_image),
        # and create_masked_condition re-applies args.mask (same canvas-space mask
        # used for TSP init) and resizes to render_size. It is therefore pixel-aligned
        # with the exported master SVG, so a partition script can sample it at each
        # master point's coordinates with no transform.
        condition.save(f"{self.args.output_dir}/condition_{condition_name}.png")
        print(
            f"\t\tSaved conditioning image -> condition_{condition_name}.png "
            f"(canvas space at {self.args.render_size}px, aligned with the master SVG).",
            flush=True,
        )
        # Prepare the image in the format expected by the pipeline.
        final_condition = self.pipe.prepare_image(
            image=condition,
            width=self.args.render_size,
            height=self.args.render_size,
            batch_size=1,
            num_images_per_prompt=1,
            device=self.device,
            dtype=self.weights_dtype,
            do_classifier_free_guidance=True,
        )
        encoded_control_image = self.pipe.vae.encode(final_condition, return_dict=False)[
            0
        ]  # TODO: use .latent_dist.sample() to introduce stochasticity.
        encoded_control_image = (
            encoded_control_image - (self.pipe.vae.config.shift_factor or 0.0)
        ) * self.pipe.vae.config.scaling_factor

        return encoded_control_image

    def create_masked_condition(self, condition):
        """Apply the input mask to the conditioning image."""
        im_np = np.array(condition)
        im_np = im_np / im_np.max()
        im_np = np.expand_dims(self.args.mask, axis=-1) * im_np
        im_np[self.args.mask < self.args.mask.mean()] = 0
        im_final = (im_np / im_np.max() * 255).astype(np.uint8)
        masked_im = Image.fromarray(im_final).resize((self.args.render_size, self.args.render_size))
        return masked_im

    def create_caption(self):
        """Generate a caption for the input image with BLIP-2."""
        blip2processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
        blip2model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, resume_download=True
        ).to(self.device)
        with torch.no_grad():
            inputs = blip2processor(self.args.input_image, return_tensors="pt").to(
                self.device, torch.float16
            )
            generated_ids = blip2model.generate(**inputs, max_new_tokens=20)
            generated_text = blip2processor.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ].strip()
        caption = f"{generated_text}"
        self.args.caption = f"a single line drawing of {caption}"

        del blip2model
        del blip2processor
        torch.cuda.empty_cache()

    def encode(self, image):
        """Encode an image into VAE latents."""
        input_dtype = image.dtype
        image = self.pipe.image_processor.preprocess(image, resize_mode="default")
        image = image.to(self.weights_dtype)
        latents = self.pipe.vae.encode(image, return_dict=False)[
            0
        ]  # TODO: use .latent_dist.sample() to introduce stochasticity.
        latents = (
            latents - (self.pipe.vae.config.shift_factor or 0.0)
        ) * self.pipe.vae.config.scaling_factor
        latents = latents.to(input_dtype)
        return latents

    def forward(self, x: Float[torch.Tensor, "B C H W"]):
        """Compute the SDS loss for a batch of input images."""
        # Set up the call parameters.
        controlnet_config = self.pipe.controlnet.config
        batch_size = x.shape[0]

        # Build classifier-free guidance embeddings.
        prompt_embeds = torch.cat([self.prompt_embeds, self.negative_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [self.pooled_prompt_embeds, self.negative_pooled_prompt_embeds], dim=0
        )

        # Sample timesteps from a trimmed range to avoid extreme noise levels.
        t = torch.randint(
            self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device
        )
        timestep = self.timesteps[t]

        # Encode the input image into latents.
        latents = self.encode(x)

        # Configure the ControlNet inputs.
        if controlnet_config.force_zeros_for_pooled_projection:
            # Some ControlNet variants expect zero pooled projections.
            controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = pooled_prompt_embeds

        if controlnet_config.joint_attention_dim is not None:
            controlnet_encoder_hidden_states = prompt_embeds
        else:
            # The official SD3.5 ControlNet does not use encoder hidden states.
            controlnet_encoder_hidden_states = None

        # Run the ControlNet and transformer under no-grad inference.
        with torch.no_grad():
            noise = torch.randn_like(latents)
            noised_latents = self.scheduler.scale_noise(
                sample=latents, timestep=timestep, noise=noise
            )
            noised_latent_input = torch.cat(
                [noised_latents] * 2
            )  # Duplicate latents for classifier-free guidance.

            # Broadcast timesteps to the doubled batch dimension.
            timesteps = self.timesteps[torch.cat([t] * 2)]
            if self.conditioning_scale == 0.0:
                # Skip ControlNet when conditioning is disabled.
                control_block_samples = None
            else:
                control_block_samples = self.pipe.controlnet(
                    hidden_states=noised_latent_input.to(self.weights_dtype),
                    timestep=timestep.to(self.weights_dtype),
                    encoder_hidden_states=controlnet_encoder_hidden_states,
                    pooled_projections=controlnet_pooled_projections,
                    controlnet_cond=self.control_image,
                    conditioning_scale=self.conditioning_scale,
                    return_dict=False,
                )[0]

            noise_pred = self.transformer(
                hidden_states=noised_latent_input.to(self.weights_dtype),
                timestep=timesteps.to(self.weights_dtype),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
            ).sample

        # Apply classifier-free guidance.
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        sigma = self.sigmas[t]
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")
        w = sigma**2
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # Clip gradients for stability.
        grad_clip_val = 1
        if grad_clip_val is not None:
            grad = grad.clamp(-grad_clip_val, grad_clip_val)

        target = (latents - grad).detach()
        sds_loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        sds_loss /= self.cfg.multisteps
        return sds_loss  # , t
