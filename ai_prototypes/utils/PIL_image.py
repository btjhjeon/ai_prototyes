import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def load_image(path, size=None, mode="RGB"):
    image = Image.open(path).convert(mode)
    if size is not None and image.size != size:
        image = image.resize(size)
    return image


def load_images(paths, size=None, mode="RGB"):
    images = []
    for path in paths:
        if size is None:
            images.append(load_image(path, mode))
            size = images[-1].size
        else:
            images.append(load_image(path, size, mode))
    return images


def merge_images(images, margin=0):
    width, height = images[0].size

    result = Image.new(mode="RGB", size=(width*len(images) + margin*(len(images)-1), height), color="white")
    for i, image in enumerate(images):
        result.paste(image.resize((width, height)), [i*width, 0, (i+1)*width, height])
    return result


def erode(image, filter_size=3, cycles=1):
    for _ in range(cycles):
        image = image.filter(ImageFilter.MinFilter(filter_size))
    return image


def dilate(image, filter_size=3, cycles=1):
    for _ in range(cycles):
        image = image.filter(ImageFilter.MaxFilter(filter_size))
    return image


def filter_gaussian(image, filter_size=3):
    return image.filter(ImageFilter.GaussianBlur(filter_size))


def blend(img1, mask, img2):
    omask = (np.asarray(mask) / 255)[..., None]
    out = (1 - omask) * img1 + omask * img2
    return Image.fromarray(out.astype('uint8'))


def smoothe_blend(img1, mask, img2):
    mask = filter_gaussian(dilate(mask, 13), 3)

    return blend(img1, mask, img2)
