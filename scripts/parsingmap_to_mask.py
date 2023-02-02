import os
import argparse
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--map', type=str, default='examples/mask_off/ai_hub/input/parsingmap/')
    parser.add_argument('-o', '--output', type=str, default='examples/mask_off/ai_hub/input/mask/')
    parser.add_argument('-i', '--index', type=int, default=19)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.isdir(args.map):
        map_paths = [os.path.join(args.map, f) for f in os.listdir(args.map)]
        output_paths = [os.path.join(args.output, f) for f in os.listdir(args.map)]
        map_paths.sort()
        output_paths.sort()
        os.makedirs(args.output, exist_ok=True)
    else:
        map_paths = [args.map]
        output_paths = [args.output]
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    for map_path, output_path in zip(map_paths, output_paths):
        map = Image.open(map_path)
        map_np = np.asarray(map)
        mask_np = (map_np == args.index).astype(np.uint8) * 255
        mask = Image.fromarray(mask_np)
        mask.save(output_path)
