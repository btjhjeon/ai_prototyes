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


def predict(
    model_path: str,
    tokenizer_path: str,
    adapter_path: Optional[str]=None,
    fp16: bool=False,
    bf16: bool=False,
    model_max_length: int=2048,
    peft_type: str="none",
):
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
        # FIXME: just for toolken!!
        # stopping_criteria = StoppingCriteriaList([model.get_stopping_criteria()])

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

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    while True:
        print()
        try:
            prompt = input("Input: ")

            result = _infer(prompt)
            print(result)
        except:
            pass


if __name__ == '__main__':
    fire.Fire(predict)
