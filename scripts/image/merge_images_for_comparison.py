import os
import argparse
import math

from ai_prototypes.utils.PIL_image import load_images, merge_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--candidates", type=str, nargs='+', required=True)
    parser.add_argument("-s", "--size", type=int, default=None)
    parser.add_argument("--board_color", type=str, default='white')
    parser.add_argument("-r", "--rows", type=int, default=1)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_candidates = len(args.candidates)
    num_row = args.rows

    candidate_files = []
    num_images = None
    for candidate in args.candidates:
        image_files = [os.path.join(candidate, f) for f in os.listdir(candidate)]
        image_files.sort()
        candidate_files.append(image_files)
        if num_images is None:
            num_images = len(image_files)
        assert len(image_files) == num_images

    output_dir = os.path.split(args.output)[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, image_paths in enumerate(zip(*candidate_files)):
        if os.path.isdir(args.output):
            file = os.path.split(image_paths[0])[-1]
            output_path = os.path.join(args.output, file)
        else:
            file, ext = os.path.splitext(args.output)
            output_path = f'{file}_{i:03d}{ext}' if num_images > 1 else args.output

        images = load_images(image_paths)
        num_plot = len(images)

        num_col = math.ceil(num_plot/num_row)
        size = (args.size, args.size) if args.size else None

        merged = merge_images(images, size=size, grid=(num_col, num_row), background=args.board_color)
        merged.save(output_path)
