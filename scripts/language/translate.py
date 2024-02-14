import os
import argparse
import tqdm
import numpy as np

from ai_prototypes.language.translation import translate
from ai_prototypes.utils.file import load_data, write_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="data file path")
    parser.add_argument("-k", "--key", type=str, required=True, help="key with the splitter \".\" (e.g. data[i]['a']['b'][j]['c'] -> a.b.c)")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file path")
    parser.add_argument("-w", "--keep_str", type=str, default=None)
    parser.add_argument("-a", "--agent", type=str, default="openai")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    agent = "openai"
    backup_path = "translation.json"

    data = load_data(args.data)
    keys = args.key.split('.')

    if os.path.exists(backup_path):
        backup_data = load_data(backup_path)
        backup_idx = len(backup_data)
        data = backup_data + data[backup_idx:]
    
    def _iter_data(data, keys, idx=None):
        if isinstance(data, (list, tuple)):
            for i, d in enumerate(data):
                if idx is None:
                    idx = i
                yield from _iter_data(d, keys, idx)
        elif isinstance(data, dict):
            value = data[keys[0]]
            if isinstance(value, str):
                yield value, data, keys[0], idx
            else:
                yield from _iter_data(value, keys[1:], idx)

    try:
        for value, d, k, idx in tqdm.tqdm(_iter_data(data, keys)):
            if idx < backup_idx:
                continue

            translated, model = translate(value, keep_str=args.keep_str, agent=args.agent)

            if translated:
                d[f"{k}_en"] = value
                d[k] = translated
                d["translator"] = model
    except:
        write_data(data[:idx], backup_path)

    write_data(data, args.output)
