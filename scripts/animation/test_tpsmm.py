from time import time
import matplotlib
matplotlib.use('Agg')
import os
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from ai_prototypes.animation.tpsmm.model import load_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default='configs/animation/tpsmm/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/animation/tpsmm/vox.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='examples/mask_off/reference_aligned/reference_jin.png', help="path to source image")
    parser.add_argument("--driving_image", default='examples/mask_off/input_inpainted/input_jin.png', help="path to driving video")
    parser.add_argument("--result_image", default='result.jpg', help="path to output")
    
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    
    parser.add_argument("--mode", default='standard', choices=['standard', 'relative', 'avd'], help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")
    
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    if opt.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    model = load_model(opt.config, opt.checkpoint, device)
    
    if os.path.isdir(opt.source_image):
        source_paths = [os.path.join(opt.source_image, f) for f in os.listdir(opt.source_image)]
        driving_paths = [os.path.join(opt.driving_image, f) for f in os.listdir(opt.driving_image)]
        result_paths = [os.path.join(opt.result_image, f) for f in os.listdir(opt.source_image)]
        source_paths.sort()
        driving_paths.sort()
        result_paths.sort()
    else:
        source_paths = [opt.source_image]
        driving_paths = [opt.driving_image]
        result_paths = [opt.result_image]

    for source_path, driving_path, result_path in zip(source_paths, driving_paths, result_paths):
        source_name = os.path.splitext(os.path.split(source_path)[-1])[0]
        driving_name = os.path.splitext(os.path.split(driving_path)[-1])[0]

        source_image = imageio.v3.imread(source_path)
        driving_image = imageio.v3.imread(driving_path)

        source_image = resize(source_image, opt.img_shape)[..., :3]
        driving_image = [resize(driving_image, opt.img_shape)[..., :3]]

        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_image)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)

        predictions = model(source, driving, mode=opt.mode)

        if os.path.split(result_path)[0]:
            os.makedirs(os.path.split(result_path)[0], exist_ok=True)
        imageio.imsave(result_path, img_as_ubyte(predictions[0]))
 