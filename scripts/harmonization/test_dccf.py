import argparse
import numpy as np
import cv2
import torch

from ai_prototypes.harmonization.dccf.harmonizer import load_model
from ai_prototypes.harmonization.dccf.mconfigs import ALL_MCONFIGS
from ai_prototypes.utils.opencv import save_image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--use-flip', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')

    parser.add_argument('--eval-prefix', type=str, default='')
    parser.add_argument('--version', type=str, default='hsl', help='[v1, hsl]')

    parser.add_argument("-i", "--image", type=str, default='examples/mask_off/ray_by_finn.jpg')
    parser.add_argument("-m", "--mask", type=str, default='examples/mask_off/ray_by_finn.png')
    parser.add_argument("-o", "--output", type=str, default='result.jpg')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    model = load_model(args.model_type, args.checkpoint, device, args.version, args.use_flip)

    # load the example
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(args.mask)
    mask = mask[:, :, 0].astype(np.float32) / 255.

    pred = model(image, mask)

    save_image(pred[:, :, ::-1], args.output)


if __name__ == '__main__':
    main()
