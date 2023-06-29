import argparse
import numpy as np
from PIL import Image

from ai_prototypes.utils.PIL_image import load_image, alpha_blend, merge_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default='examples/mask_off/ray_by_finn.jpg')
    parser.add_argument("-m", "--mask", type=str, default='examples/mask_off/ray_by_finn.png')
    parser.add_argument("-o", "--output", type=str, default='result.jpg')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image = load_image(args.image)
    mask = load_image(args.mask, image.size, "L")

    assert image.size == mask.size

    white_mask = np.asarray(mask)[:,:,np.newaxis]
    black_mask = np.zeros_like(white_mask)
    red_mask = np.concatenate([white_mask, black_mask, black_mask], axis=2)
    red_mask = Image.fromarray(red_mask)

    mask_only = (np.asarray(image) * (np.asarray(mask)[:,:,np.newaxis] / 255)).astype(np.uint8)
    mask_only = Image.fromarray(mask_only)

    except_mask = (np.asarray(image) * (1. - np.asarray(mask)[:,:,np.newaxis] / 255)).astype(np.uint8)
    except_mask = Image.fromarray(except_mask)

    blending_image = alpha_blend(image, mask, red_mask, 0.5)

    result = merge_images([image, mask_only, except_mask, blending_image])
    result.save(args.output)
