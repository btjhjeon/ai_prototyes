import os
import glob
import tqdm
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline


MODEL_CAPABLE_BATCH = 2


def build_stable_diffusion(model_name, torch_dtype=torch.float16):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to('cuda')
    return pipe


def generate_and_save(pipe, output_path, prompt, negative_prompt='', num_samples=4):
    outputs = []
    for i in range(0, num_samples, MODEL_CAPABLE_BATCH):
        num = min(MODEL_CAPABLE_BATCH, num_samples - i)
        outputs += generate(pipe, prompt, negative_prompt, num_samples=num)

    for i, output in enumerate(outputs):
        file, ext = os.path.splitext(output_path)
        path = f'{file}_{i:03d}{ext}' if len(outputs) > 0 else output_path
        output.save(path)
        print(f'successfully save result to "{path}"!')


def generate(pipe, prompt, negative_prompt='', guidance_scale=7, num_samples=4):
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples,
    ).images
    return out
