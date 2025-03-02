import argparse
from pathlib import Path

from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)

args = parser.parse_args()
assert args.input.is_dir()


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    pil_img = pil_img.filter(ImageFilter.GaussianBlur(2))
    return ImageOps.fit(pil_img, width_and_height)


if __name__ == '__main__':
    for path in tqdm(list(args.input.glob('*'))):
        if path.suffix == '.txt':
            continue
        if '_' in path.stem:
            continue
        try:
            img = Image.open(path)
        except Exception:
            print(f'Image {path.name} could not be opened. Skipping.')
            continue
        scaled_image = convert_crop_and_resize(img, (512, 512))
        output_path = path.with_stem(path.stem + '_scaled1').with_suffix('.png')
        scaled_image.save(output_path)
