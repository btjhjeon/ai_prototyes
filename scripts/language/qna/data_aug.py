import os
import tqdm
import time
import json
import random
import tiktoken
import traceback
import argparse
import numpy as np
from collections import Counter
from datasets import load_dataset

from ai_prototypes.language.api import get_response
from scripts.language.qna.retrieve_sentence import (
    get_embedding_model,
    get_embeddings,
    calculate_score,
    preprocess_data
)


DATA_DIR = "data/qna/"
DATA_PATH = {
    "input": os.path.join(DATA_DIR, "complaint/v0.2/trainval/complaint_v0.2_trainval.jsonl"),
    "dist": os.path.join(DATA_DIR, "complaint/v0.1/trainval/complaint_v0.1_trainval.jsonl"),
    "law": os.path.join(DATA_DIR, "law/law_v3.json"),
    "output": os.path.join(DATA_DIR, "complaint/v0.3/complaint_v0.3_aug.{ext}")
}


def calculate_dist(dataset, ids=None):
    law_ids = []
    law_counts = {}
    for d in dataset:
        for doc_id in d["doc_id"]:
            if doc_id not in law_counts:
                law_ids.append(doc_id)
                law_counts[doc_id] = 1
            else:
                law_counts[doc_id] += 1
    total_count = sum(law_counts.values())

    if ids is not None:
        law_ids = ids
    counts = np.array([law_counts[law_id] for law_id in law_ids], dtype=float)
    dist = counts / total_count
    return dist, counts, total_count, law_ids


def write_output(data, path):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path.format(ext="json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    with open(path.format(ext="jsonl"), 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False)+"\n")


if __name__=="__main__":
    aug_data_count = 500
    agent = "openai"
    enc = tiktoken.get_encoding("cl100k_base")

    data_path = DATA_PATH["input"]
    with open(data_path, 'r', encoding="utf-8") as jsonls:
        dataset = [json.loads(j) for j in jsonls if j]
    
    data_path = DATA_PATH["dist"]
    with open(data_path, 'r', encoding="utf-8") as jsonls:
        dataset_dist = [json.loads(j) for j in jsonls if j]

    input_dist, input_count, input_total, law_ids = calculate_dist(dataset)
    dist, dist_count, dist_total, _ = calculate_dist(dataset_dist, law_ids)
    new_exp = np.exp(np.log(dist) / 1.5)
    new_dist = new_exp / new_exp.sum()

    aug_count = np.around(new_dist / new_dist.max() * input_count.max() - input_count)
    aug_count[aug_count<0] = 0
    aug_count = np.around(aug_count * aug_data_count / aug_count.sum()).astype(int)
    # doc_id 당 몇개씩 증강할건지 계산한 dictionary
    doc_id_cnts = {law_ids[i]: count for i, count in enumerate(aug_count)}

    law_path = DATA_PATH["law"]
    with open(law_path, 'r', encoding="utf-8") as f:
        laws = json.load(f)
    law_dict = {law["doc_id"]:law for i, law in enumerate(laws)}

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    def get_qnp_by_doc_id(doc_id, dataset):
        queries = []
        for entry in dataset:
            if doc_id in entry["doc_id"]:
                queries.append(f"{entry['title']} {entry['summary']}")
        passage = law_dict[doc_id]["passage"]
        return queries, passage

    def get_prompt(doc_id, count, max_num_q=5, num_few_shot=10, max_token=4000, seed=777):
        base_prompt = f"""다음의 예시들은 ###관련법령과 관련된 ###질문을 활용하여 ###추가질문 {count}개를 생성한 예시입니다.\n\n"""
        queries, passage = get_qnp_by_doc_id(doc_id, dataset_dist)
        random.seed(seed)
        random.shuffle(queries)
        input_q = " @@@eos@@@\n".join(queries[:max_num_q]) + " @@@eos@@@"
        input_prompt = f"""앞의 예시들을 참고하여 아래의 ###관련법령과 ###질문을 활용하여 ###추가질문 {count}개를 생성해주세요. ###추가질문은 반복되지 않고, 최대한 아래 ###관련법령과 아래 ###질문을 잘 활용하여 간결하게 만들어 주시오. 아래 ###관련법령에 있는 용어를 그대로 사용하기 보다 유사 단어로 변경하여 생성해 주세요.

###관련법령
{passage}

###질문
{input_q}

###추가질문"""
        few_shot_prompt=""
        common_few_shot = [law_ids[i] for i in dist_count.argsort()[::-1][:num_few_shot]]
        for fs_id in common_few_shot:
            questions_fs, passage_fs = get_qnp_by_doc_id(fs_id, dataset_dist)
            random.seed(seed)
            random.shuffle(questions_fs)
            q = " @@@eos@@@\n".join(questions_fs[:max_num_q]) + " @@@eos@@@\n"
            qq = " @@@eos@@@\n".join(questions_fs[-max_num_q:]) + " @@@eos@@@\n"
            _few_shot_prompt = f"""###관련법령
{passage_fs}

###질문
{q}

###추가질문
{qq}\n\n"""
            if len(enc.encode(base_prompt + few_shot_prompt + _few_shot_prompt + input_prompt)) > max_token:
                break
            few_shot_prompt += _few_shot_prompt
        return base_prompt + few_shot_prompt + input_prompt, passage

    save_path = DATA_PATH["output"]
    if os.path.exists(save_path.format(ext="json")):
        with open(save_path.format(ext="json"), 'r', encoding="utf-8") as f:
            aug_dataset = json.load(f)
    else:
        aug_dataset = []
    for i, (doc_id, count) in tqdm.tqdm(enumerate(doc_id_cnts.items(), total=len(doc_id_cnts))):
        if count < 1 or i < 140:
            continue

        try:
            prompt, passage = get_prompt(doc_id, count)
            response, model = get_response(
                prompt,
                system_prompt="당신은 법령을 보고 예상되는 질문 제목과 요약을 생성하는 AI입니다.",
                agent=agent
            )
            aug_results = response.split("\n")
            for aug_result in aug_results:
                aug_dict = {
                    "title": aug_result,
                    "summary": "",
                    "law": passage,
                    "doc_id": [doc_id],
                    "meta": {
                        "is_aug": True,
                        "t-gen_model": model,
                        "s-gen_model": model
                    }
                }
                aug_dataset.append(aug_dict)
        except:
            traceback.print_exc()
            break
    
    write_output(aug_dataset, DATA_PATH["output"])
