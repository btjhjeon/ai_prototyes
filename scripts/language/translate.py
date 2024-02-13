import os
import argparse
import tqdm
import numpy as np

from ai_prototypes.language.api import get_response
from ai_prototypes.utils.file import load_data, write_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="data file path")
    parser.add_argument("-k", "--key", type=str, required=True, help="key with the splitter \".\" (e.g. data[i]['a']['b'][j]['c'] -> a.b.c)")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file path")
    parser.add_argument("-w", "--keep_str", type=str, default=None)
    return parser.parse_args()


def translate(input, keep_str=None):
    if len(input) <= 1:
        return None, None

    system_prompt = "I want you to act as an Korean translator, spelling corrector and improver." \
                    "I will speak to you in English and you will translate it and answer in the corrected and improved version of my text, in Korean." \
                    "I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level Korean words and sentences." \
                    "Keep the meaning same, but make them more literary." \
                    "I want you to only reply the correction, the improvements and nothing else, do not write explanations."
    user_prefix_prompt = "Please translate the following into Korean:\n"
    if keep_str:
        inputs = input.split(keep_str)
    else:
        inputs = [input]
    outputs = [get_response(user_prefix_prompt + p, system_prompt=system_prompt, agent="openai") if p else ("", "") for p in inputs]
    model = [output[1] for output in outputs if output[1]][0]
    outputs = [output[0] for output in outputs]
    if keep_str:
        outputs = keep_str.join(outputs)
    else:
        outputs = outputs[0]
    return outputs, model


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

            translated, model = translate(value, keep_str=args.keep_str)

            if translated:
                d[f"{k}_en"] = value
                d[k] = translated
                d["translator"] = model
    except:
        write_data(data[:idx], backup_path)

    write_data(data, args.output)
