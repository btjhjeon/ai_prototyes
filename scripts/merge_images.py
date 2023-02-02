import argparse
import math

from ai_prototypes.utils.PIL_image import load_images, merge_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, nargs='+', required=True)
    parser.add_argument("-s", "--size", type=int, default=256)
    parser.add_argument("-r", "--rows", type=int, default=1)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_images = len(args.images)
    num_row = args.rows
    num_col = math.ceil(num_images/num_row)

    images = load_images(args.images, (args.size, args.size))
    merged = merge_images(images, grid=(num_col, num_row))
    merged.save(args.output)
