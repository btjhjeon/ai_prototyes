import os
import argparse
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything

from ai_prototypes.inpainting.huggingface.stable_diffusion import (
    build_stable_diffusion,
    inpaint
)
from ai_prototypes.utils.PIL_image import apply_exif_orientation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='parlance/dreamlike-diffusion-1.0-inpainting')
    parser.add_argument('-i', '--image', type=str, default='examples/mask_off/MMU_team/input')
    parser.add_argument('-m', '--mask', type=str, default='examples/mask_off/MMU_team/mask')
    parser.add_argument('-o', '--output', type=str, default='examples/mask_off/MMU_team/input_inpainted')
    parser.add_argument('-p', '--prompt', type=str, default='talking face')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--dilate', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed:
        seed_everything(args.seed)

    image_path = args.image
    mask_path = args.mask
    name = os.path.splitext(image_path)[0]
    prompt = args.prompt


    if os.path.isdir(args.image):
        image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image)]
        mask_paths = [os.path.join(args.mask, f) for f in os.listdir(args.mask)]
        output_paths = [os.path.join(args.output, f) for f in os.listdir(args.image)]
        image_paths.sort()
        mask_paths.sort()
        output_paths.sort()
        os.makedirs(args.output, exist_ok=True)
    else:
        image_paths = [args.image]
        mask_paths = [args.mask]
        output_paths = [args.output]
        if args.output and os.path.split(args.output)[0]:
            os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    for image_path, mask_path, output_path in zip(image_paths, mask_paths, output_paths):
        image = Image.open(image_path).convert('RGB')
        image = apply_exif_orientation(image)
        
        mask = Image.open(mask_path).convert('L')
        mask = apply_exif_orientation(mask)

        pipe = build_stable_diffusion(args.model)
        # pipe = build_stable_diffusion1()
        # pipe = build_stable_diffusion2()

        output = inpaint(pipe, image, mask, prompt=prompt)
        output.save(output_path)
