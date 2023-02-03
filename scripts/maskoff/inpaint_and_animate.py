import argparse
import os
import cv2
import numpy as np
from skimage import img_as_ubyte
import torch

from ai_prototypes.inpainting.lama import setup
from ai_prototypes.inpainting.lama.config import load_config
from ai_prototypes.inpainting.lama.model import load_model as load_lama
from ai_prototypes.inpainting.lama.model import infer as lama_infer
from ai_prototypes.inpainting.lama.evaluation.data import load_image
from ai_prototypes.animation.tpsmm.model import load_model as load_tpsmm
from ai_prototypes.utils.PIL_image import save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lama_model', type=str, default='checkpoints/inpainting/lama/lama-celeba-hq/lama-fourier/')
    parser.add_argument("--tpsmm_config", default='configs/animation/tpsmm/vox-256.yaml', help="path to config")
    parser.add_argument("--tpsmm_checkpoint", default='checkpoints/animation/tpsmm/vox.pth.tar', help="path to checkpoint to restore")
    parser.add_argument('-i', '--image', type=str, default='examples/mask_off/MMU_team/input')
    parser.add_argument('-m', '--mask', type=str, default='examples/mask_off/MMU_team/mask')
    parser.add_argument('-r', '--reference', type=str, default='examples/mask_off/MMU_team/reference')
    parser.add_argument('-o', '--output', type=str, default='examples/mask_off/MMU_team/reference_animated')
    parser.add_argument('-d', '--dilate', type=int, default=0)
    return parser.parse_args()


def inpaint_and_animate(lama_config, lama, tpsmm, image_path, mask_path, reference_path, output_path, dilation_kernel_size, device):
    size = (256, 256)
    image = load_image(image_path, size, mode='RGB')
    _, mask = load_image(mask_path, size, mode='L', return_orig=True)
    reference = load_image(reference_path, size, mode='RGB')
    if dilation_kernel_size > 0:
        mask = cv2.dilate(mask, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8))
        mask = mask.astype(np.float32) / 255

    inpainted = lama_infer(lama_config, lama, image, mask, device)
    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, inpainted)

    source = (np.transpose(reference, (1, 2, 0)) * 255).astype(np.uint8)
    driving = [inpainted]

    source = torch.tensor(source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255
    source = source.to(device)
    driving = torch.tensor(np.array(driving)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3) / 255
    driving = driving.to(device)

    predictions = tpsmm(source, driving, mode='standard')
    save_image(img_as_ubyte(predictions[0]), output_path)


if __name__ == '__main__':
    setup()
    args = parse_args()
    lama_config = load_config(args.lama_model)

    dilation_kernel_size = args.dilate

    device = torch.device(lama_config.device)
    lama = load_lama(lama_config, device)
    tpsmm = load_tpsmm(args.tpsmm_config, args.tpsmm_checkpoint, device)

    if os.path.isdir(args.image):
        image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image)]
        mask_paths = [os.path.join(args.mask, f) for f in os.listdir(args.mask)]
        reference_paths = [os.path.join(args.reference, f) for f in os.listdir(args.reference)]
        output_paths = [os.path.join(args.output, f) for f in os.listdir(args.reference)]
        image_paths.sort()
        mask_paths.sort()
        reference_paths.sort()
        output_paths.sort()
        os.makedirs(args.output, exist_ok=True)
    else:
        image_paths = [args.image]
        mask_paths = [args.mask]
        reference_paths = [args.reference]
        output_paths = [args.output]
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    for image_path, mask_path, reference_path, output_path in zip(image_paths, mask_paths, reference_paths, output_paths):
        inpaint_and_animate(lama_config, lama, tpsmm, image_path, mask_path, reference_path, output_path, dilation_kernel_size, device)
