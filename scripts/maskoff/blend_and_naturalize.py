import os
import argparse
import numpy as np
import cv2
import torch

from ai_prototypes.inpainting.lama import setup
from ai_prototypes.inpainting.lama.config import load_config
from ai_prototypes.inpainting.lama.model import load_model, infer
from ai_prototypes.utils.PIL_image import load_image, blend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default='examples/mask_off/lama/source')
    parser.add_argument("-m", "--mask", type=str, default='examples/mask_off/lama/mask')
    parser.add_argument("-t", "--target", type=str, default='examples/mask_off/lama/target')
    parser.add_argument("-b", "--blend", type=str, default='examples/mask_off/lama/blend')
    parser.add_argument("-o", "--output", type=str, default='examples/mask_off/lama/blend_and_inpaint')
    return parser.parse_args()


if __name__ == "__main__":
    size = (256, 256)
    scale = 1.0
    y_offset = 0
    dilation_kernel_size = 11

    args = parse_args()

    setup()
    # predict_config = load_config('ai_prototypes/inpainting/lama/big-lama/')
    predict_config = load_config('ai_prototypes/inpainting/lama/lama-celeba-hq/lama-fourier/')

    device = torch.device(predict_config.device)
    model = load_model(predict_config, device)

    target_files = [f for f in os.listdir(args.target)]
    for target_file in target_files:
        target_name = os.path.splitext(target_file)[0]
        source_name = target_name.split('_')[-1]
        source_file = mask_file = f'{source_name}.png'
        source_path = os.path.join(args.source, source_file)
        target_path = os.path.join(args.target, target_file)
        mask_path = os.path.join(args.mask, mask_file)
        blend_path = os.path.join(args.blend, target_file)
        output_path = os.path.join(args.output, target_file)

        src_image = load_image(source_path, size)
        mask_image = load_image(mask_path, size, "L")
        trg_image = load_image(target_path, size)

        if scale != 1.0:
            size = (int(trg_image.width * scale), int(trg_image.height * scale))
            scaled_image = trg_image.resize(size)
            box = [
                int((trg_image.width - size[0])/2),
                int((trg_image.height - size[1])/2) + y_offset,
                int((trg_image.width - size[0])/2) + size[0],
                int((trg_image.height - size[1])/2) + size[1] + y_offset]
            trg_image.paste(scaled_image, box)

        assert src_image.size == mask_image.size == trg_image.size

        blend_image = blend(src_image, mask_image, trg_image)
        blend_image.save(blend_path)

        image = np.transpose(np.array(blend_image, dtype=np.float32), (2, 0, 1)) / 255
        mask = cv2.dilate(cv2.Canny(np.array(mask_image), 128, 255), np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8))
        mask = mask.astype(np.float32) / 255

        cur_res = infer(predict_config, model, image, mask, device)
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cur_res)
