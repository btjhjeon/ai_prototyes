import os
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from detectron2.data.detection_utils import _apply_exif_orientation

from ai_prototypes.inpainting.stable_diffusion.huggingface import build_stable_diffusion, build_stable_diffusion2, outpaint


if __name__ == "__main__":
    # seed_everything(123)

    image_path = 'ray_by_finn.jpg'
    mask_path = 'ray_by_finn.png'
    name = os.path.splitext(image_path)[0]
    output_path = f'{name}_output.png'
    prompt = 'talking face'

    image = Image.open(image_path).convert('RGB')
    image = _apply_exif_orientation(image)
    
    mask = Image.open(mask_path).convert('L')
    mask = _apply_exif_orientation(mask)

    # pipe = build_stable_diffusion()
    pipe = build_stable_diffusion2()
    output = outpaint(pipe, image, mask, prompt=prompt)
    output.save(output_path)
