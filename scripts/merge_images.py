import os
import argparse
import math

from ai_prototypes.utils.PIL_image import load_images, merge_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, nargs='+', required=True)
    parser.add_argument("-s", "--size", type=int, default=None)
    parser.add_argument("-c", "--board_color", type=str, default='white')
    parser.add_argument("-r", "--rows", type=int, default=1)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_images = len(args.images)
    num_row = args.rows
    num_col = math.ceil(num_images/num_row)
    size = (args.size, args.size) if args.size else None

    output_dir = os.path.split(args.output)[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    images = load_images(args.images, resize=False)
    merged = merge_images(images, size=size, grid=(num_col, num_row), background=args.board_color)
    merged.save(args.output)
