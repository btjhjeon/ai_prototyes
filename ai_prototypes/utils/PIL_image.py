import numpy as np
from PIL import Image, ImageFilter


def load_image(path, size=None, mode="RGB"):
    image = Image.open(path).convert(mode)
    image = apply_exif_orientation(image)
    if size is not None and image.size != size:
        image = image.resize(size)
    return image


def load_images(paths, size=None, resize=True, mode="RGB"):
    images = []
    for path in paths:
        if size is None:
            images.append(load_image(path, mode=mode))
            if resize:
                size = images[-1].size
        else:
            images.append(load_image(path, size, mode))
    return images


# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.
    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`
    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527
    Args:
        image (PIL.Image): a PIL image
    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def save_image(image, path):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(path, quality=100, subsampling=0)


def merge_images(images, size=None, margin=0, grid=None, mode="RGB", padding=True, background='white'):
    for i, image in enumerate(images):
        if not isinstance(image, Image.Image):
            images[i] = Image.fromarray(image, mode=mode)

    if size is None:
        width, height = images[0].size
    elif isinstance(size, int):
        width, height = size, size
    elif isinstance(size, (list, tuple)):
        width, height = size

    if grid is None:
        grid = (len(images), 1)

    board_size = (
        width*grid[0] + margin*(grid[0]-1),
        height*grid[1] + margin*(grid[1]-1)
    )
    result = Image.new(mode=mode, size=board_size, color=background)
    for i, image in enumerate(images):
        loc = (i%grid[0], i//grid[0])

        if padding:
            scale = min(width / image.width, height / image.height)
            new_size = (int(scale * image.width), int(scale * image.height))
            offset = (int((width - new_size[0]) / 2), int((height - new_size[1]) / 2))
            box = [
                loc[0] * width + offset[0],
                loc[1] * height + offset[1],
                loc[0] * width + offset[0] + new_size[0],
                loc[1] * height + offset[1] + new_size[1]
            ]
            image = image.resize(new_size)
        else:
            box = [
                loc[0] * width,
                loc[1] * height,
                (loc[0] + 1) * width,
                (loc[1] + 1) * height
            ]
            image = image.resize(size)
        result.paste(image, box)
    return result


def erode(image, filter_size=3, cycles=1):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    for _ in range(cycles):
        image = image.filter(ImageFilter.MinFilter(filter_size))
    return image


def dilate(image, filter_size=3, cycles=1):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    for _ in range(cycles):
        image = image.filter(ImageFilter.MaxFilter(filter_size))
    return image


def filter_gaussian(image, filter_size=3):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    return image.filter(ImageFilter.GaussianBlur(filter_size))


def blend(img1, mask, img2):
    omask = (np.asarray(mask) / 255)[..., None]
    out = (1 - omask) * img1 + omask * img2
    return Image.fromarray(out.astype('uint8'))


def alpha_blend(img1, mask, img2, alpha):
    omask = (np.asarray(mask) / 255)[..., None]
    out = (1 - omask) * img1 + alpha * omask * img2 + (1 - alpha) * omask * img1
    return Image.fromarray(out.astype('uint8'))


def smoothe_blend(img1, mask, img2, kernel_size=3):
    mask = filter_gaussian(dilate(mask, 13), kernel_size)

    return blend(img1, mask, img2)
