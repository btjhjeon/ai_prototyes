import os
import fire
import tqdm
import easydict

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk


PROMPT = """
{text}
{summary}"""


def main(**args):
    args = easydict.EasyDict(args)
    args.tokenizers = args.tokenizers.split(',')
    tokenizers = [AutoTokenizer.from_pretrained(tokenizer) for tokenizer in args.tokenizers]
    # dataset = load_dataset(args.dataset, split=args.split)
    dataset = load_from_disk(os.path.join(args.dataset, args.split))

    num_tokens = []
    for i in tqdm.tqdm(range(dataset.num_rows)):
        ins = dataset[i]
        for j, tokenizer in enumerate(tokenizers):
            if len(num_tokens) <= j:
                num_tokens.append(0)
            prompt = PROMPT.format(
                text=ins["text"],
                summary=ins["label"],
                bos_token=tokenizer.bos_token,
                eos_token=tokenizer.eos_token
            )
            num_tokens[j] += len(tokenizer.tokenize(prompt))
    
    for tokenizer_name, num_token in zip(args.tokenizers, num_tokens):
        print(f'{tokenizer_name}:\t{num_token}')


if __name__ == "__main__":
    fire.Fire(main)
