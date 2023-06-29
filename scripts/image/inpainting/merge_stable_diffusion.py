import os
import argparse
from diffusers import StableDiffusionInpaintPipeline
import torch


def load_inpainting_model(model_name):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    return pipe


def merge_parameters(module_1, module_2, alpha, out_module):
    state_1 = module_1.state_dict()
    state_2 = module_2.state_dict()
    assert len(state_1) == len(state_2)

    new_state = {}
    for k, v1 in state_1.items():
        assert k in state_2
        v2 = state_2[k]
        new_v = alpha * v1 + (1 - alpha) * v2
        new_state[k] = new_v
    out_module.load_state_dict(new_state)


def merge(
    model_1: StableDiffusionInpaintPipeline,
    model_2: StableDiffusionInpaintPipeline,
    model_out: StableDiffusionInpaintPipeline,
    alpha: float = 0.25
):
    merge_parameters(model_1.text_encoder, model_2.text_encoder, alpha, model_out.text_encoder)
    merge_parameters(model_1.unet, model_2.unet, alpha, model_out.unet)
    merge_parameters(model_1.vae, model_2.vae, alpha, model_out.vae)
    merge_parameters(model_1.safety_checker, model_2.safety_checker, alpha, model_out.safety_checker)


def calculate_weight_distance(model_1, model_2):
    def _sum_module_diff(module_1, module_2):
        state_1 = module_1.state_dict()
        state_2 = module_2.state_dict()
        assert len(state_1) == len(state_2)

        abs_diff = torch.tensor(0.0, dtype=torch.float32)
        square_std = torch.tensor(0.0, dtype=torch.float32)
        num_elem = torch.tensor(0, dtype=torch.int64)
        for k, v1 in state_1.items():
            assert k in state_2
            v2 = state_2[k]
            diff = (v1 - v2).to(torch.float32)  # std operation only support float32
            abs_diff += torch.abs(diff).sum()
            square_std += torch.std(diff).square() * v1.numel()
            num_elem += v1.numel()
        return abs_diff, square_std, num_elem
    
    abs_diff, square_std, num_elem = torch.tensor(0.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32), 0
    temp_abs_diff, temp_square_std, temp_num_elem = _sum_module_diff(model_1.text_encoder, model_2.text_encoder)
    abs_diff += temp_abs_diff
    square_std += temp_square_std
    num_elem += temp_num_elem
    temp_abs_diff, temp_square_std, temp_num_elem = _sum_module_diff(model_1.unet, model_2.unet)
    abs_diff += temp_abs_diff
    square_std += temp_square_std
    num_elem += temp_num_elem
    temp_abs_diff, temp_square_std, temp_num_elem = _sum_module_diff(model_1.vae, model_2.vae)
    abs_diff += temp_abs_diff
    square_std += temp_square_std
    num_elem += temp_num_elem
    temp_abs_diff, temp_square_std, temp_num_elem = _sum_module_diff(model_1.safety_checker, model_2.safety_checker)
    abs_diff += temp_abs_diff
    square_std += temp_square_std
    num_elem += temp_num_elem
    return abs_diff / num_elem, torch.sqrt(square_std / num_elem)


def validate(model_1, model_2, model_out):
    dist_12, _ = calculate_weight_distance(model_1, model_2)
    dist_1out, _ = calculate_weight_distance(model_1, model_out)
    dist_2out, _ = calculate_weight_distance(model_2, model_out)
    assert dist_1out < dist_12 and dist_2out < dist_12
    print(f'[DISTANCE] 1 - 2 : {dist_12}, 1 - merged : {dist_1out}, merged - 2 : {dist_2out}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model1', type=str, default='parlance/dreamlike-diffusion-1.0-inpainting')
    parser.add_argument('-m2', '--model2', type=str, default='runwayml/stable-diffusion-inpainting')
    parser.add_argument('-a', '--alpha', type=float, default=0.5)
    parser.add_argument('-o', '--output_dir', type=str, default='huggingface_caches/diffusers/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    alpha = args.alpha
    model_1_id = args.model1
    model_2_id = args.model2
    out_path = os.path.join(args.output_dir, f'models--merged-{alpha}-diffusion-inpainting')
    print(f'Alpha: {alpha}')
    print(f'Model 1 path: {model_1_id}')
    print(f'Model 2 path: {model_2_id}')
    print(f'Output path: {out_path}')

    pipe_1 = load_inpainting_model(model_1_id)
    pipe_2 = load_inpainting_model(model_2_id)
    pipe_out = load_inpainting_model(model_1_id)

    merge(pipe_1, pipe_2, pipe_out, alpha)

    # pipe_out.save_pretrained(out_path)
    pipe_load = load_inpainting_model(out_path)
    load_diff, _ = calculate_weight_distance(pipe_out, pipe_load)
    assert load_diff == 0.0

    validate(pipe_1, pipe_2, pipe_load)
