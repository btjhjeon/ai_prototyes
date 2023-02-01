import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from ai_prototypes.inpainting.lama import setup
from ai_prototypes.inpainting.lama.config import load_config
from ai_prototypes.inpainting.lama.model import load_model
from ai_prototypes.inpainting.lama.evaluation.data import load_image, pad_img_to_modulo
from ai_prototypes.inpainting.lama.evaluation.utils import move_to_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ai_prototypes/inpainting/lama/big-lama/')
    parser.add_argument('-i', '--image', type=str, default='examples/mask_off/stargan-v2/result/stargan-v2__reference_elon__input_elon__blended.jpg')
    parser.add_argument('-m', '--mask', type=str, default='examples/mask_off/mask_aligned/mask_elon.png')
    parser.add_argument('-o', '--output', type=str, default='result.jpg')
    return parser.parse_args()


if __name__ == '__main__':
    setup()
    args = parse_args()
    predict_config = load_config(args.model)

    dilation_kernel_size = 13

    device = torch.device(predict_config.device)
    model = load_model(predict_config, device)

    image = load_image(args.image, mode='RGB')
    _, mask = load_image(args.mask, mode='L', return_orig=True)
    mask = cv2.dilate(cv2.Canny(mask, 128, 255), np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8))
    cv2.imwrite(os.path.splitext(args.output)[0]+'_mask.png', mask)
    mask = mask.astype(np.float32) / 255
    sample = dict(image=image, mask=mask[None, ...])

    sample['unpad_to_size'] = sample['image'].shape[1:]
    sample['image'] = pad_img_to_modulo(sample['image'], 8)
    sample['mask'] = pad_img_to_modulo(sample['mask'], 8)
    batch = default_collate([sample])

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = model(batch)                    
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, cur_res)
