import os
import glob
import tqdm
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

from ai_prototypes.utils.PIL_image import dilate, filter_gaussian


def build_stable_diffusion():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to('cuda')
    return pipe


def build_stable_diffusion2():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to('cuda')
    return pipe


def outpaint_from_path(pipe, image_path, mask_path, output_path):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    output = outpaint(pipe, image, mask)

    output.save(output_path)
    print(f'successfully save result to "{output_path}"!')


def outpaint(pipe, image, mask, prompt=''):
    # `height` and `width` have to be divisible by 8
    scale = 1024 / max(image.width, image.height)
    if scale > 1:
        scale = 1.0
    width = int(image.width * scale // 8 * 8)
    height = int(image.height * scale // 8 * 8)
    image = image.resize((width, height))
    mask = mask.resize((width, height))

    image_np = np.asarray(image)
    mask_np = np.asarray(mask)

    gen_image = outpaint_with_numpy(pipe, image_np, mask_np, prompt=prompt)

    mask = filter_gaussian(dilate(mask, 13), 3)

    # omask = (mask > 0).astype('uint8')[..., None]
    omask = (np.asarray(mask) / 255)[..., None]
    out = (1 - omask) * image + omask * np.clip(gen_image * 255, 0, 255).astype('uint8')
    out = out.astype('uint8')

    return Image.fromarray(out)


def outpaint_with_numpy(pipe, image, mask, prompt=''):
    height, width = image.shape[:2]
    negative_prompt = "person, any object, duplicate artifacts"
    guidance_scale = 3
    num_samples = 1

    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    gen_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples,
        height=height,
        width=width,
        output_type=np.ndarray,
    ).images[-1]

    omask = (mask > 0).astype('uint8')[..., None]
    # omask = (omask / 255)[..., None]
    out = (1 - omask) * image + omask * np.clip(gen_image * 255, 0, 255).astype('uint8')
    out = out.astype('uint8')

    return Image.fromarray(out)


if __name__ == "__main__":
    image_dir = './samples/'
    pipe = build_stable_diffusion()

    for mask_path in glob.glob(os.path.join(image_dir, '*_mask.png')):
        image_path = mask_path.replace('_mask.png', '.png')
        output_path = mask_path.replace('_mask.png', '_output.png')

        outpaint_from_path(pipe, image_path, mask_path, output_path)
