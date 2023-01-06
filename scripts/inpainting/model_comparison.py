import os
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from detectron2.data.detection_utils import _apply_exif_orientation

from ai_prototypes.inpainting.stable_diffusion.huggingface import build_stable_diffusion, build_stable_diffusion2, outpaint


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
    # bbox = [0, 0, 1280, 680]
    # image_path = 'natalie-portman-orgasm-1500543917.jpeg'
    # bbox = [300, 0, 1700, 1000]
    # image_path = 'sample_jerome.jpg'
    # bbox = [0, 0, 996, 1440]
    image_path = 'istockphoto-1181447477-170667a.jpeg'
    bbox = [150, 40, 270, 300]
    name = os.path.splitext(image_path)[0]
    output_path = f'{name}_output.png'
    scale = 1.0
    prompt = 'standing in front of a park with Manhattan feeled medieval style.'

    image = Image.open(image_path).convert('RGB')
    image = _apply_exif_orientation(image)
    new_image, mask = make_center_ingredients(image, bbox, scale)

    pipe_1_5 = build_stable_diffusion()
    output_1_5 = outpaint(pipe_1_5, new_image, mask, prompt)

    pipe_2_0 = build_stable_diffusion2()
    output_2_0 = outpaint(pipe_2_0, new_image, mask, prompt)
    # output.save('temp.png')

    width, height = output_1_5.size
    result_board = Image.new(mode="RGB", size=(width*4, height), color="white")
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle(bbox, outline='red')
    result_board.paste(image.resize((width, height)), [0, 0, width, height])
    result_board.paste(new_image.resize((width, height)), [width, 0, width*2, height])
    result_board.paste(output_1_5, [width*2, 0, width*3, height])
    result_board.paste(output_2_0, [width*3, 0, width*4, height])

    result_board.save('temp.png')
