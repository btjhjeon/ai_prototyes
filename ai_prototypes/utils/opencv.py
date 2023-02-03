import os
import cv2


def save_image(image, path):
    if os.path.splitext(path)[-1].lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif os.path.splitext(path)[-1].lower() == '.png':
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
