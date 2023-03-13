import argparse
from pytorch_lightning import seed_everything

from ai_prototypes.text_to_image.huggingface.stable_diffusion import build_stable_diffusion, generate_and_save


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-n', '--num_samples', type=int, default=4)
    parser.add_argument('-m', '--model_path',
                        type=str,
                        default='/t2meta/dataset/stable-diffusion-models/custom_models/SD_1_5__gwanghwamun_1000')
    parser.add_argument('-p', '--prompt', type=str, default='<s1>')
    parser.add_argument('-np', '--neg_prompt', type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    num_samples = args.num_samples

    sd_model = args.model_path
    prompt = args.prompt
    neg_prompt = args.neg_prompt

    print(f'[MODEL]      {sd_model}')
    print(f'[PROMPT]     {prompt}')
    print(f'[NEG_PROMPT] {neg_prompt}')

    pipe = build_stable_diffusion(sd_model)
    generate_and_save(pipe, 'temp.jpg', prompt, neg_prompt, num_samples)
