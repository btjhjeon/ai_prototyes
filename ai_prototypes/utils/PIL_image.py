import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from detectron2.data.detection_utils import _apply_exif_orientation


def load_image(path, size=None, mode="RGB"):
    image = Image.open(path).convert(mode)
    image = _apply_exif_orientation(image)
    if size is not None and image.size != size:
        image = image.resize(size)
    return image


def load_images(paths, size=None, mode="RGB"):
    images = []
    for path in paths:
        if size is None:
            images.append(load_image(path, mode=mode))
            size = images[-1].size
        else:
            images.append(load_image(path, size, mode))
    return images


def save_image(image, path):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(path, quality=100, subsampling=0)


def merge_images(images, margin=0, grid=None, mode="RGB"):
    for i, image in enumerate(images):
        if not isinstance(image, Image.Image):
            images[i] = Image.fromarray(image, mode=mode)

    width, height = images[0].size
    if grid is None:
        grid = (len(images), 1)

    board_size = (
        width*grid[0] + margin*(grid[0]-1),
        height*grid[1] + margin*(grid[1]-1)
    )
    result = Image.new(mode=mode, size=board_size, color="white")
    for i, image in enumerate(images):
        loc = (i%grid[0], i//grid[0])
        box = [loc[0]*width, loc[1]*height, (loc[0]+1)*width, (loc[1]+1)*height]
        result.paste(image.resize((width, height)), box)
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


def smoothe_blend(img1, mask, img2):
    mask = filter_gaussian(dilate(mask, 13), 3)

    return blend(img1, mask, img2)
