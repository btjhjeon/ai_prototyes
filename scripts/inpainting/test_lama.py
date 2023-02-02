import argparse
import os
import cv2
import numpy as np
import torch

from ai_prototypes.inpainting.lama import setup
from ai_prototypes.inpainting.lama.config import load_config
from ai_prototypes.inpainting.lama.model import load_model, infer
from ai_prototypes.inpainting.lama.evaluation.data import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/inpainting/lama/lama-celeba-hq/lama-fourier/')
    parser.add_argument('-i', '--image', type=str, default='examples/mask_off/stargan-v2/result/stargan-v2__reference_elon__input_elon__blended.jpg')
    parser.add_argument('-m', '--mask', type=str, default='examples/mask_off/mask_aligned/mask_elon.png')
    parser.add_argument('-o', '--output', type=str, default='result.jpg')
    parser.add_argument('-d', '--dilate', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    setup()
    args = parse_args()
    predict_config = load_config(args.model)

    dilation_kernel_size = args.dilate

    device = torch.device(predict_config.device)
    model = load_model(predict_config, device)

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
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    for image_path, mask_path, output_path in zip(image_paths, mask_paths, output_paths):
        image = load_image(image_path, mode='RGB')
        _, mask = load_image(mask_path, mode='L', return_orig=True)
        if dilation_kernel_size > 0:
            mask = cv2.dilate(mask, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8))
            mask = mask.astype(np.float32) / 255

        cur_res = infer(predict_config, model, image, mask, device)
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cur_res)
