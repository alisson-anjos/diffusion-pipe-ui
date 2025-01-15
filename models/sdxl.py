import diffusers
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.utils.logging import logger
import peft
import safetensors

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE


class SDXLPipeline(BasePipeline):
    # Unique name, used to make the cache_dir path.
    name = 'sdxl'

    # layers that will participate in activation checkpointing
    checkpointable_layers = [
        'InitialLayer',
        'UnetDownBlockLayer',
        'UnetMidBlockLayer',
        'UnetUpBlockLayer',
        'FinalLayer',
    ]

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.diffusers_pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
            self.model_config['checkpoint_path'],
            torch_dtype=self.model_config['dtype'],
            add_watermarker=False,
        )
        self.diffusers_pipeline.scheduler = diffusers.DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        # Probably good to always do this for SDXL.
        self.diffusers_pipeline.upcast_vae()
        self.unet.train()
        self.text_encoder.train()
        self.text_encoder_2.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for module in (self.unet, self.text_encoder, self.text_encoder_2):
            for name, p in module.named_parameters():
                p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        # TODO: support training with cached text embeddings.
        return []

    def configure_adapter(self, adapter_config):
        # Target all linear layers in the main blocks.
        self._add_adapter(adapter_config, self.unet, [self.unet.down_blocks, self.unet.mid_block, self.unet.up_blocks], state_dict_key_prefix='unet.')
        # Target all linear layers in the text encoder.
        self._add_adapter(adapter_config, self.text_encoder, [self.text_encoder], state_dict_key_prefix='text_encoder.')
        self._add_adapter(adapter_config, self.text_encoder_2, [self.text_encoder_2], state_dict_key_prefix='text_encoder_2.')

    def _add_adapter(self, adapter_config, top_level_module, target_modules, state_dict_key_prefix=''):
        adapter_type = adapter_config['type']
        target_linear_modules = []
        for target_module in target_modules:
            for name, submodule in target_module.named_modules():
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.append(name)

        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        top_level_module.add_adapter(peft_config)
        for name, p in top_level_module.named_parameters():
            p.original_name = state_dict_key_prefix + name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def save_adapter(self, save_dir, peft_state_dict):
        adapter_type = self.config['adapter']['type']
        if adapter_type == 'lora':
            # TODO: should we do any additional checks here? This helpful function appears to completely convert
            # the PEFT format state_dict to kohya format. Every key in the lora is correctly loaded by Forge.
            # But all these different formats are a mess and I hardly understand it. This seems to work though.
            kohya_sd = diffusers.utils.state_dict_utils.convert_state_dict_to_kohya(peft_state_dict)
            safetensors.torch.save_file(kohya_sd, save_dir / 'lora.safetensors', metadata={'format': 'pt'})
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents']
        caption = inputs['caption']
        input_ids = self._get_input_ids(caption, self.tokenizer)
        input_ids_2 = self._get_input_ids(caption, self.tokenizer_2)

        bs = latents.shape[0]
        device = latents.device
        noise = torch.randn_like(latents, device=device)
        min_timestep = 0
        max_timestep = self.scheduler.config.num_train_timesteps
        if timestep_quantile is not None:
            t = int(timestep_quantile*max_timestep)
            timesteps = torch.full((bs,), t, device=device)
        else:
            timesteps = torch.randint(min_timestep, max_timestep, (bs,), device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        target = noise

        pixel_height = latents.shape[-2] * self.vae_scale_factor
        pixel_width = latents.shape[-1] * self.vae_scale_factor
        # TODO: set original size based on actual source image? Not sure what other trainers do here.
        original_size = target_size = (pixel_height, pixel_width)
        add_time_ids = self._get_add_time_ids(
            original_size,
            (0, 0),
            target_size,
            dtype=torch.float32,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        ).expand(bs, -1)

        return noisy_latents, timesteps, input_ids, input_ids_2, add_time_ids, target

    def _get_input_ids(self, prompt, tokenizer):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # TODO: can we support >77 tokens like various inference programs do?
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )
        return text_input_ids

    def to_layers(self):
        layers = [InitialLayer(self.diffusers_pipeline)]
        unet = self.diffusers_pipeline.unet
        for block in unet.down_blocks:
            layers.append(UnetDownBlockLayer(block))
        if unet.mid_block is not None:
            layers.append(UnetMidBlockLayer(unet.mid_block))
        for i, block in enumerate(unet.up_blocks):
            is_final_block = i == len(unet.up_blocks) - 1
            layers.append(UnetUpBlockLayer(block, is_final_block))
        layers.append(FinalLayer(unet))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, diffusers_pipeline):
        super().__init__()
        self.clip_skip = None
        self.diffusers_pipeline = diffusers_pipeline
        self.text_encoder = self.diffusers_pipeline.text_encoder
        self.text_encoder_2 = self.diffusers_pipeline.text_encoder_2
        # Unet modules we need to register on the nn.Module
        self.time_proj = self.unet.time_proj
        self.time_embedding = self.unet.time_embedding
        self.add_embedding = self.unet.add_embedding
        self.time_embed_act = self.unet.time_embed_act
        self.encoder_hid_proj = self.unet.encoder_hid_proj
        self.conv_in = self.unet.conv_in

    @property
    def unet(self):
        return self.diffusers_pipeline.unet

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for tensor in inputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        sample, timestep, input_ids, input_ids_2, add_time_ids, target = inputs

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break
        forward_upsample_size = torch.tensor(forward_upsample_size)

        prompt_embeds_list = []
        for text_input_ids, text_encoder in [(input_ids, self.text_encoder), (input_ids_2, self.text_encoder_2)]:
            prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
            # We are only ALWAYS interested in the pooled output of the final text encoder
            if prompt_embeds[0].ndim == 2:
                pooled_prompt_embeds = prompt_embeds[0]

            if self.clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(self.clip_skip + 2)]
            prompt_embeds_list.append(prompt_embeds)
        encoder_hidden_states = torch.concat(prompt_embeds_list, dim=-1)

        add_time_ids = add_time_ids.to(prompt_embeds.dtype)
        add_text_embeds = pooled_prompt_embeds
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        # TODO: need timestep cond? Might be optional / inference only.

        t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
        emb = self.unet.time_embedding(t_emb, None)

        aug_emb = self.unet.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.unet.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)

        return make_contiguous(sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target)


class UnetDownBlockLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target = inputs

        if hasattr(self.block, "has_cross_attention") and self.block.has_cross_attention:
            sample, res_samples = self.block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = self.block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples
        return make_contiguous(sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target)


class UnetMidBlockLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target = inputs

        if hasattr(self.block, "has_cross_attention") and self.block.has_cross_attention:
            sample = self.block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = self.mid_block(sample, emb)

        return make_contiguous(sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target)


class UnetUpBlockLayer(nn.Module):
    def __init__(self, block, is_final_block):
        super().__init__()
        self.block = block
        self.is_final_block = is_final_block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target = inputs

        res_samples = down_block_res_samples[-len(self.block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(self.block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not self.is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
        else:
            upsample_size = None

        if hasattr(self.block, "has_cross_attention") and self.block.has_cross_attention:
            sample = self.block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                upsample_size=upsample_size,
            )
        else:
            sample = self.block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

        return make_contiguous(sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target)


class FinalLayer(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.conv_norm_out = unet.conv_norm_out
        self.conv_act = unet.conv_act
        self.conv_out = unet.conv_out

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size, target = inputs

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        sample = sample.to(torch.float32)
        target = target.to(torch.float32)
        loss = F.mse_loss(sample, target)
        return loss
