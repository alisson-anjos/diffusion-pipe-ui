from pathlib import Path
import sys
import argparse
import os.path
sys.path.insert(0, os.path.abspath('submodules/Wan2_1'))

import torch
import torchvision

from models.base import PreprocessMediaFile
from wan import configs as wan_configs
from wan.modules.vae import WanVAE

CKPT_DIR = '/data2/imagegen_models/Wan2.1-T2V-1.3B'

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)

args = parser.parse_args()
assert args.input.is_file()


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


def vae_decode(tensor, vae):
    return vae.model.decode(tensor, vae.scale)


def write_image(decoded, name):
    assert decoded.ndim == 5 and decoded.shape[2] == 1, decoded.shape
    decoded = decoded.squeeze(0)
    decoded = ((decoded + 1) / 2).clamp(0, 1)

    img = decoded.squeeze(1)
    pil_img = torchvision.transforms.functional.to_pil_image(img)
    output_path = args.input.with_name(name + '.jpg')
    pil_img.save(output_path)


if __name__ == '__main__':
    wan_config = wan_configs.t2v_1_3B
    vae = WanVAE(
        vae_pth=os.path.join(CKPT_DIR, wan_config.vae_checkpoint),
        device='cuda',
    )

    preprocessor = PreprocessMediaFile({}, support_video=True, framerate=16, round_height=8, round_width=8, round_frames=8)

    target_frames = 33 if args.input.suffix == '.mp4' else 1
    tensor = preprocessor(args.input, None, size_bucket=(624, 624, target_frames))[0][0].unsqueeze(0)

    p = next(vae.model.parameters())
    device, dtype = p.device, p.dtype

    print(f'Input shape: {tensor.shape}')
    latents = vae_encode(tensor.to(device, dtype), vae)
    print(f'Latents shape: {latents.shape}')
    first_frame = latents[:, :, 0:1, ...]
    decoded = vae_decode(first_frame, vae)
    print(f'Decoded shape: {decoded.shape}')
    write_image(decoded, args.input.stem + '_decoded')

    tensor[:, :, 1:, ...] = 0
    latents = vae_encode(tensor.to(device, dtype), vae)
    print(latents[:, :, -1, :, :])
    first_frame = latents[:, :, 0:1, ...]
    decoded = vae_decode(first_frame, vae)
    write_image(decoded, args.input.stem + '_decoded2')
