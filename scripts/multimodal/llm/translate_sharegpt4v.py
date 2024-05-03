import os
import argparse
import tqdm
import traceback
from multiprocessing import Pool

from ai_prototypes.language.translation import translate
from ai_prototypes.utils.file import load_data, write_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="data file path")
    parser.add_argument("-k", "--key", type=str, required=True, help="key with the splitter \".\" (e.g. data[i]['a']['b'][j]['c'] -> a.b.c)")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file path")
    parser.add_argument("-a", "--agent", type=str, default="openai")
    return parser.parse_args()


IMAGE_TOKEN = "<image>"


if __name__=="__main__":
    args = parse_args()
    agent = "openai"

    if os.path.exists(args.output):
        org_data_size = len(load_data(args.data))
        data = load_data(args.output)
        assert len(data) == org_data_size, f"{len(data)} != {org_data_size}"
    else:
        data = load_data(args.data)
    keys = args.key.split('.')

    
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


    inputs = []
    for value, d, k, idx in _iter_data(data, keys):
        if f"{k}_en" in d:
            continue
        
        if value.endswith("\n" + IMAGE_TOKEN):
            keep_str = "\n" + IMAGE_TOKEN
        elif value.startswith(IMAGE_TOKEN + "\n"):
            keep_str = IMAGE_TOKEN + "\n"
        else:
            keep_str = IMAGE_TOKEN
        
        inputs.append((value, d, k, data[idx]['id'], keep_str))

    def _translate(input):
        value, d, k, id, keep_str = input
        try:
            translated, model = translate(value, keep_str, agent=agent)

            if len(translated) > len(value) * 4 or len(translated) < len(value) // 4:
                print(f"[ID] {id}")
                print(f"[EN] {value}")
                print(f"[KR] {translated}")

                translated, model = translate(value, keep_str, agent=agent, model="gpt-4-turbo")
        except Exception:
            traceback.print_exc()
            return d, k, id, None, None

        return d, k, id, translated, model


    with Pool(128) as p:
        outputs = list(tqdm.tqdm(p.imap(_translate, inputs), total=len(inputs)))
        outputs = p.map(_translate, inputs)
    print("Translated successfully!")

    num_error = 0
    for input, output in tqdm.tqdm(zip(inputs, outputs), total=len(inputs)):
        d, k, output_id, translated, model = output
        value, d, k, input_id, keep_str = input
        assert input_id == output_id, f"{input_id} != {output_id}"

        if translated:
            d[f"{k}_en"] = d[k]
            d[k] = translated
            d["translator"] = model
        else:
            num_error += 1

    print(f"{num_error} / {len(data)}")
    write_data(data, args.output)
    print(f"Save \"{args.output}\" successfully!")
