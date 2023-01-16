from PIL import ImageFilter


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
