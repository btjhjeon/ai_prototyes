from typing import Optional, List, Tuple
import os
import json
import fire
import tqdm
import torch

import transformers
from transformers import StoppingCriteriaList
from peft import PeftModel

from ai_prototypes.language.huggingface import get_model_and_tokenizer
from ai_prototypes.language import openai_api, anthropic_api
from scripts.language.korean_sat.prompt import get_prompt


def predict(
    data_path: str,
    model: str="hf",                # openai, anthropic, hf, etc.
    model_name: str="gpt-3.5-turbo",    # required for openai, anthropic
    model_path: str=None,               # required for hf
    tokenizer_path: str=None,           # required for hf
    adapter_path: str=None,             # required for hf
    fp16: bool=False,                   # required for hf
    bf16: bool=False,                   # required for hf
    model_max_length: int=2048,         # required for hf
    peft_type: str="none",              # required for hf
    num_shot: int=-1,
):
    mode = "CoT" if num_shot < 0 else f"{num_shot}-shot"
    data_name = os.path.splitext(os.path.basename(data_path))[0]

    if model.lower() == "openai":
        assert model_name
        need_integrated_prompt = False

        def _infer(prompt):
            result = openai_api.request_inference(prompt['user_prompt'], prompt['system_prompt'], model_name)
            first_value = result[0] if len(result) > 0 else ""
            return first_value, result

    elif model.lower() == "anthropic":
        assert model_name
        need_integrated_prompt = True
        agent = anthropic_api.get_agent()

        def _infer(prompt):
            result = anthropic_api.request_inference(prompt, agent, model_name)
            first_value = result[0] if len(result) > 0 else ""
            return first_value, result

    elif model.lower() == "hf":
        assert model_path
        tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        model_path = os.path.abspath(model_path)
        if model_path.lower().endswith("hf"):
            model_name = os.path.split(os.path.split(model_path)[0])[1]
        else:
            model_name = os.path.split(model_path)[1]
        need_integrated_prompt = True

        model, tokenizer = get_model_and_tokenizer(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            adapter_path=adapter_path,
            model_max_length=model_max_length,
            fp16=fp16,
            bf16=bf16,
            peft_type=peft_type,
            training=False,
        )
        model.eval()

        def _infer(prompt, kind="greedy"):
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                if kind == "greedy":
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=model_max_length,
                        # stopping_criteria=stopping_criteria,
                    )
                elif kind == "beam":
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=model_max_length,
                        num_beams=5,
                        early_stopping=True
                    )
                elif kind == "sampling":
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=model_max_length,
                        do_sample=True,
                        top_k=0,
                        temperature=0.7
                    )
                elif kind == "nucleus":
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=model_max_length,
                        do_sample=True,
                        top_k=0,
                        top_p=0.92
                    )

                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = result[len(prompt):].strip()
                first_value = result[0] if len(result) > 0 else ""
                return first_value, result
    else:
        need_integrated_prompt = True
        print(f"Only evaluation for {model_name} with result file.")

    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []
    intermediate_result_path = f"outputs/results_{data_name}_{model_name}_{mode}.json"
    if os.path.exists(intermediate_result_path):
        with open(intermediate_result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

    try:
        total_score = 0
        earned_score = 0
        earned_score_common = 0
        earned_score_select = 0
        missing_score = 0
        num_total = 0
        num_correct = 0
        num_missing = 0
        for i, d in tqdm.tqdm(enumerate(data), total=len(data)):

            for j, p in enumerate(d['problems']):
                prompt = get_prompt(data, i, j, need_integrated_prompt, num_shot)
                score = p['score']
                answer = str(p['answer'])

                if num_total >= len(results):
                    pred, text = _infer(prompt)
                    pred = pred.replace('①', '1')
                    pred = pred.replace('②', '2')
                    pred = pred.replace('③', '3')
                    pred = pred.replace('④', '4')
                    pred = pred.replace('⑤', '5')
                    results.append({
                        "no": num_total+1,
                        "text": text,
                        "prediction": pred,
                        "answer": answer,
                        "score": score
                    })
                else:
                    result = results[num_total]
                    pred = result['prediction']

                total_score += score
                num_total += 1
                if pred == answer:
                    earned_score += score
                    num_correct += 1
                    if num_total < 35:
                        earned_score_common += score
                    else:
                        earned_score_select += score
                elif not pred.isdigit():
                    num_missing += 1
                    missing_score += score

                # print(result)
                # print(f"### GT: {answer}")

    except Exception as e:
        import traceback
        traceback.print_exc()

    with open(intermediate_result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    
    print(f"[SCORE] {earned_score}/{total_score}")
    print(f"[COMMON] {earned_score_common}, [SELECT]{earned_score_select}")
    print(f"[ACCURACY] {num_correct/num_total:.4f} ({num_correct}/{num_total})")
    print(f"[# MISSING]: {num_missing}, [MISSING SCORE] {missing_score}")


if __name__ == '__main__':
    fire.Fire(predict)
