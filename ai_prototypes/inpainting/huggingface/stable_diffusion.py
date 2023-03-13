import os
import glob
import tqdm
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

from ai_prototypes.utils.PIL_image import smoothe_blend, blend


def build_stable_diffusion(model_name, revision=None, torch_dtype=torch.float16):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
    ).to('cuda')
    return pipe


def build_stable_diffusion1():
    return build_stable_diffusion(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )


def build_stable_diffusion2():
    return build_stable_diffusion(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )


def inpaint_from_path(pipe, image_path, mask_path, output_path):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    output = inpaint(pipe, image, mask)

    output.save(output_path)
    print(f'successfully save result to "{output_path}"!')


def inpaint(pipe, image, mask, prompt='', blend_smooth=0):
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

    gen_image = inpaint_with_numpy(pipe, image_np, mask_np, prompt=prompt, blend_smooth=blend_smooth)
    return gen_image


def inpaint_with_numpy(pipe, image, mask, prompt='', blend_smooth=0):
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

    if blend_smooth > 0:
        out = smoothe_blend(image, mask, np.clip(np.asarray(gen_image) * 255, 0, 255).astype('uint8'))
    else:
        out = blend(image, mask, np.clip(np.asarray(gen_image) * 255, 0, 255).astype('uint8'))
    return Image.fromarray(out)


if __name__ == "__main__":
    image_dir = './samples/'
    pipe = build_stable_diffusion()

    for mask_path in glob.glob(os.path.join(image_dir, '*_mask.png')):
        image_path = mask_path.replace('_mask.png', '.png')
        output_path = mask_path.replace('_mask.png', '_output.png')

        inpaint_from_path(pipe, image_path, mask_path, output_path)
