import os
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from detectron2.data.detection_utils import _apply_exif_orientation

from ai_prototypes.inpainting.huggingface.stable_diffusion import build_stable_diffusion, build_stable_diffusion2, inpaint


def make_center_ingredients(image, bbox, scale):
    width, height = image.size
    bbox_width, bbox_height = int((bbox[2] - bbox[0]) * scale), int((bbox[3] - bbox[1]) * scale)
    relocated_bbox = [
        int((width-bbox_width)/2),
        int((height-bbox_height)/2),
        int((width-bbox_width)/2) + bbox_width,
        int((height-bbox_height)/2) + bbox_height,
    ]
    crop_image = image.crop(bbox).resize((bbox_width, bbox_height))
    new_image = Image.new(mode="RGB", size=(width, height), color="white")
    new_image.paste(crop_image, relocated_bbox)

    mask = Image.new(mode="L", size=(width, height), color="white")
    mask_draw = ImageDraw.Draw(mask)
    mask_bbox = [
        relocated_bbox[0] + 1,
        relocated_bbox[1] + 1,
        relocated_bbox[2] - 1,
        relocated_bbox[3] - 1
    ]
    mask_draw.rectangle(mask_bbox, fill="black")

    return new_image, mask


if __name__ == "__main__":
    # seed_everything(23)

    # image_path = 'Image_Inpainting_Private/STG_admin/temp/imgs_v2/0e586634-197e-4363-8707-5fdeeedbcb1a_input.jpg'
    # bbox = [100, 100, 1000, 960]
    # image_path = 'Image_Inpainting_Private/STG_admin/temp/imgs_v2/3cda7053-88df-4c39-984a-b5cb897c67fe_input.jpg'
    # bbox = [100, 200, 1000, 1000]
    # image_path = 'Image_Inpainting_Private/STG_admin/temp/imgs_v2/fe2b229c-92a5-4853-bd6e-10eaa73911bd_input.jpg'
    # bbox = [0, 300, 1247, 1660]
    # image_path = 'Image_Inpainting_Private/STG_admin/temp/imgs/017c11bd-60ad-4226-924c-26d5215fa0f1_input.jpg'
    # bbox = [420, 500, 870, 1440]
    # image_path = 'Image_Inpainting_Private/test_instagram_for_Object_Removal/zae_zer0_2730318555986539987.jpg'
    # bbox = [300, 200, 900, 1000]
    # image_path = 't-kendall-jenner-enraged-models.jpg'
    # bbox = [0, 0, 1280, 720]
    image_path = 'istockphoto-1181447477-170667a.jpeg'
    bbox = [150, 40, 270, 300]
    name = os.path.splitext(image_path)[0]
    output_path = f'{name}_output.png'
    scale = 1.0
    prompt = 'in crowd'

    image = Image.open(image_path).convert('RGB')
    image = _apply_exif_orientation(image)
    new_image, mask = make_center_ingredients(image, bbox, scale)

    # pipe = build_stable_diffusion()
    pipe = build_stable_diffusion2()
    output = inpaint(pipe, new_image, mask)
    # output.save('temp.png')

    width, height = output.size
    result_board = Image.new(mode="RGB", size=(width*3, height), color="white")
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle(bbox, outline='red')
    result_board.paste(image.resize((width, height)), [0, 0, width, height])
    result_board.paste(new_image.resize((width, height)), [width, 0, width*2, height])
    result_board.paste(output, [width*2, 0, width*3, height])

    result_board.save('temp.png')
