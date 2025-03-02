from pathlib import Path
import sys
import argparse
import os.path
sys.path.insert(0, os.path.abspath('submodules/Cosmos'))

import torch
import torchvision
from torch import nn
import accelerate

from models.base import PreprocessMediaFile
from utils.common import load_state_dict
from cosmos1.models.diffusion.inference.inference_utils import load_model_by_config
from cosmos1.utils.lazy_config import instantiate as lazy_instantiate
from cosmos1.models.autoregressive.tokenizer.modules import EncoderFactorized, DecoderFactorized, CausalConv3d

torch.set_grad_enabled(False)


OFFICIAL_VIDEO_VAE_PATH = '/data2/imagegen_models/cosmos/Cosmos-1.0-Tokenizer-CV8x8x8'
COMFYUI_VIDEO_VAE_WEIGHTS = '/data2/imagegen_models/cosmos/cosmos_cv8x8x8_1.0.safetensors'


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)

args = parser.parse_args()
assert args.input.is_file()


class CausalContinuousVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalContinuousVideoTokenizer")
        self.embedding_dim = embedding_dim
        self.sigma_data = 0.5
        self.encoder = EncoderFactorized(z_channels=z_factor * z_channels, **kwargs)
        self.decoder = DecoderFactorized(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        latent_temporal_chunk = 16
        self.latent_mean = nn.Parameter(torch.zeros([self.embedding_dim * latent_temporal_chunk], dtype=torch.float32))
        self.latent_std = nn.Parameter(torch.ones([self.embedding_dim * latent_temporal_chunk], dtype=torch.float32))


    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        latent_ch = z.shape[1]
        latent_t = z.shape[2]
        dtype = z.dtype
        mean = self.latent_mean.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=dtype, device=z.device)
        std = self.latent_std.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=dtype, device=z.device)
        return ((z - mean) / std) * self.sigma_data

    def decode(self, z):
        in_dtype = z.dtype
        latent_ch = z.shape[1]
        latent_t = z.shape[2]
        mean = self.latent_mean.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        std = self.latent_std.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        z = z / self.sigma_data
        z = z * std + mean
        z = self.post_quant_conv(z)
        return self.decoder(z)


def load_official_video_vae():
    model = load_model_by_config(
        config_job_name='Cosmos_1_0_Diffusion_Text2World_7B',
        config_file='submodules/Cosmos/cosmos1/models/diffusion/config/config.py',
    )
    vae = lazy_instantiate(model.config.tokenizer)
    vae.load_weights(OFFICIAL_VIDEO_VAE_PATH)
    vae.sigma_data = model.sigma_data
    return vae


def load_custom_video_vae():
    with accelerate.init_empty_weights():
        vae = CausalContinuousVideoTokenizer(
            attn_resolutions=[32],
            channels=128,
            channels_mult=[2, 4, 4],
            dropout=0.0,
            in_channels=3,
            num_res_blocks=2,
            out_channels=3,
            resolution=1024,
            patch_size=4,
            patch_method="haar",
            z_channels=16,
            z_factor=1,
            num_groups=1,
            legacy_mode=False,
            spatial_compression=8,
            temporal_compression=8,
            embedding_dim=16,
        )
    missing_keys, unexpected_keys = vae.load_state_dict(load_state_dict(COMFYUI_VIDEO_VAE_WEIGHTS), assign=True, strict=False)
    assert len(missing_keys) == 0
    vae.eval()
    return vae


if __name__ == '__main__':
    vae = load_custom_video_vae().to('cuda')
    preprocessor = PreprocessMediaFile({}, support_video=True, framerate=24, round_height=8, round_width=8, round_frames=8)

    target_frames = 33 if args.input.suffix == '.mp4' else 1
    tensor = preprocessor(args.input, size_bucket=(720, 720, target_frames))[0].unsqueeze(0)

    p = next(vae.encoder.parameters())
    device, dtype = p.device, p.dtype
    print(f'Input shape: {tensor.shape}')
    latents = vae.encode(tensor.to(device, dtype))
    print(f'Latents shape: {latents.shape}')
    decoded = vae.decode(latents).to('cpu', torch.float32)
    print(f'Decoded shape: {decoded.shape}')

    decoded = decoded.squeeze(0)
    decoded = ((decoded + 1) / 2).clamp(0, 1)

    if decoded.shape[1] == 1:
        img = decoded.squeeze(1)
        pil_img = torchvision.transforms.functional.to_pil_image(img)
        output_path = args.input.with_stem(args.input.stem + '_decoded')
        pil_img.save(output_path)
