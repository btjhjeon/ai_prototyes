import os
import cv2
import numpy as np

import torch
from torch.utils.data._utils.collate import default_collate

from .evaluation.data import pad_img_to_modulo
from .evaluation.utils import move_to_device
from .training.trainers import load_checkpoint


def load_model(predict_config, device):
    checkpoint_path = os.path.join(predict_config.model.path, 
                                   'models', 
                                   predict_config.model.checkpoint)
    model = load_checkpoint(predict_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    return model


def infer(predict_config, model, image, mask, device):
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
    return cur_res
