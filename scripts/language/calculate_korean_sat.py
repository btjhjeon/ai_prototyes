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
    data_path: str,
    model_path: str,
    tokenizer_path: str,
    adapter_path: Optional[str]=None,
    fp16: bool=False,
    bf16: bool=False,
    model_max_length: int=2048,
    peft_type: str="none",
    num_shot: int=0
):
    prediction_arguments = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "adapter_path": adapter_path,
        "fp16": fp16,
        "bf16": bf16,
        "model_max_length": model_max_length,
        "peft_type": peft_type,
        "data_path": data_path,
    }

    environment_infos = {
        "world_size": os.environ.get("WORLD_SIZE", None),
        "master_address": os.environ.get("MASTER_ADDR", None),
        "master_port": os.environ.get("MASTER_PORT", None),
        "slurm": {
            "jobid": os.environ.get("SLURM_JOBID", None),
            "nodelist": os.environ.get("SLURM_JOB_NODELIST", None),
        },
    }

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

    with open(data_path, 'r') as f:
        data = json.load(f)

    if num_shot == 0:
        example = ""
    elif num_shot == 1:
        example = """### 지문:
" 다음 글을 읽고 물음에 답하시오. 어떤 독서 이론도 이 한 장의 사진만큼 독서의 위대함을 분명하게 말해 주지 못할 것이다. 사진은 제2차 세계 대전 당시 처참하게 무너져 내린 런던의 한 건물 모습이다. ㉠(폐허 속에서도 사람들이 책을 찾아 서가 앞에 선 이유는 무엇일까?) 이들은 갑작스레 닥친 상황에서 독서를 통해 무언가를 구하고자 했을 것이다.독서는 자신을 살피고 돌아볼 계기를 제공함으로써 어떻게 살 것인가의 문제를 생각하게 한다. 책은 인류의 지혜와 경험이 담겨 있는 문화유산이며, 독서는 인류와의 만남이자 끝없는 대화이다. 독자의 경험과 책에 담긴 수많은 경험들의 만남은 성찰의 기회를 제공함으로써 독자의 내면을 성장시켜 삶을 바꾼다. 이런 의미에서 독서는 자기 성찰의 행위이며, 성찰의 시간은 깊이 사색하고 스스로에게 질문을 던지는 시간이어야 한다. 이들이 책을 찾은 것도 혼란스러운 현실을 외면하려 한 것이 아니라 자신의 삶에 대한 숙고의 시간이 필요했기 때문이다.또한 ㉡(독서는 자신을 둘러싼 현실을 올바로 인식하고 당면한 문제를 해결할 논리와 힘을 지니게 한다.) 책은 세상에 대한 안목을 키우는 데 필요한 지식을 담고 있으며, 독서는 그 지식을 얻는 과정이다. 독자의 생각과 오랜 세월 축적된 지식의 만남은 독자에게 올바른 식견을 갖추고 당면한 문제를 해결할 방법을 모색하도록 함으로써 세상을 바꾼다. 세상을 변화시킬 동력을 얻는 이 시간은 책에 있는 정보를 이해하는 데 그치는 것이 아니라 그 정보가 자신의 관점에서 문제를 해결할 수 있는 타당한 정보인지를 판단하고 분석하는 시간이어야 한다. 서가 앞에 선 사람들도 시대적 과제를 해결할 실마리를 책에서 찾으려 했던 것이다.독서는 자기 내면으로의 여행이며 외부 세계로의 확장이다. 폐허 속에서도 책을 찾은 사람들은 독서가 지닌 힘을 알고, 자신과 현실에 대한 이해를 구하고자 책과의 대화를 시도하고 있었던 것이다."
위 지문을 읽고 다음 문제를 푸시오.
### 문제: 윗글을 바탕으로 할 때, ㉠의 답으로 적절하지 않은 것은?
### 선택지:
1. 인류의 지혜와 경험을 배우기 위해
2. 현실로부터 도피할 방법을 구하기 위해
3. 시대적 과제를 해결할 실마리를 찾기 위해
4. 자신의 삶에 대해 숙고할 시간을 갖기 위해
5. 세상에 대한 안목을 키우는 지식을 얻기 위해
### 정답: 2. 현실로부터 도피할 방법을 구하기 위해

"""

    prompt = """{example}### 지문:
"{paragraph}"
위 지문을 읽고 다음 문제를 푸시오.
### 문제: {question}
### 선택지
{choices}
### 정답: """
    prompt_w_ref = """{example}### 지문:
"{paragraph}"
### 보기:
"{question_plus}"
위 지문과 보기을 읽고 다음 문제를 푸시오.
### 문제: {question}
### 선택지:
{choices}
### 정답: """

    total_score = 0
    earned_score = 0
    num_total = 0
    num_correct = 0
    for d in tqdm.tqdm(data):
        paragraph = d['paragraph']

        for p in d['problems']:
            question = p['question']
            choices = []
            for i, c in enumerate(p['choices']):
                choices.append(f'{i+1}. {c}')
            choices = '\n'.join(choices)
            score = p['score']
            answer = str(p['answer'])

            if 'question_plus' in p:
                text = prompt_w_ref.format(
                    example=example,
                    paragraph=paragraph,
                    question=question,
                    question_plus=p['question_plus'],
                    choices=choices 
                )
            else:
                text = prompt.format(
                    example=example,
                    paragraph=paragraph,
                    question=question,
                    choices=choices 
                )

            result = _infer(text)
            pred = result[len(text)]

            total_score += score
            num_total += 1
            if pred == answer:
                earned_score += score
                num_correct += 1

            # print(result)
            # print(f"### GT: {answer}")
    
    print(f"[SCORE] {earned_score}/{total_score}")
    print(f"[ACCURACY] {num_correct}/{num_total}")


if __name__ == '__main__':
    fire.Fire(predict)
