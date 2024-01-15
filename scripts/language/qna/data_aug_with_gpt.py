import os
import tqdm
import time
import json
import random
import tiktoken
import traceback
import argparse
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
    "train": {
        "FAQ": os.path.join(DATA_DIR, "complaint_FAQ_total_v5/train/complaint_FAQ_total_v5_train.jsonl"),
        "total": os.path.join(DATA_DIR, "complaint_total/complaint_total_v2_train.jsonl"),
        "output": os.path.join(DATA_DIR, "complaint_total_with_FAQ/train/complaint_FAQ_total_v5_q-aug_v0.2_train.{ext}"),
        "output_total": os.path.join(DATA_DIR, "complaint_total_with_FAQ/train/complaint_total_v2_with_FAQ_total_v5_train.{ext}")
    },
    "val": {
        "FAQ": os.path.join(DATA_DIR, "complaint_FAQ_total_v5/val/complaint_FAQ_total_v5_val.jsonl"),
        "total": None,
        "output": os.path.join(DATA_DIR, "complaint_total_with_FAQ/val/complaint_FAQ_total_v5_q-aug_v0.2_val.{ext}"),
        "output_total": None
    },
    "trainval": {
        "FAQ": os.path.join(DATA_DIR, "complaint_FAQ_total_v5/trainval/complaint_FAQ_total_v5_trainval.jsonl"),
        "total": os.path.join(DATA_DIR, "complaint_total/complaint_total_v2_train.jsonl"),
        "output": os.path.join(DATA_DIR, "complaint_total_with_FAQ/trainval/complaint_FAQ_total_v5_q-aug_v0.2_trainval.{ext}"),
        "output_total": os.path.join(DATA_DIR, "complaint_total_with_FAQ/trainval/complaint_total_v2_with_FAQ_total_v5_trainval.{ext}")
    },
    "test": {
        "FAQ": os.path.join(DATA_DIR, "complaint_FAQ_total_v5/test/complaint_FAQ_total_v5_test.jsonl"),
        "total": os.path.join(DATA_DIR, "complaint_total/complaint_total_v2_test.jsonl"),
        "output": os.path.join(DATA_DIR, "complaint_total_with_FAQ/test/complaint_FAQ_total_v5_q-aug_v0.2_test.{ext}"),
        "output_total": os.path.join(DATA_DIR, "complaint_total_with_FAQ/test/complaint_total_v2_with_FAQ_total_v5_test.{ext}")
    },
    "example": os.path.join(DATA_DIR, "complaint_total/complaint_total_v2.json"),
    "law": os.path.join(DATA_DIR, "law/law_v3.json")
}


if __name__=="__main__":
    split = "trainval"
    agent = "openai"
    data_path = DATA_PATH[split]["FAQ"]
    data_source = os.path.basename(data_path)
    with open(data_path, 'r', encoding="utf-8") as jsonls:
        dataset = [json.loads(j) for j in jsonls]
    dataset = [d for d in dataset if "answer" in d]
    
    example_data_path = DATA_PATH["example"]
    with open(example_data_path, 'r', encoding="utf-8") as f:
        examples = json.load(f)
    examples = [e for e in examples if "answer" in e]

    law_path = DATA_PATH["law"]
    with open(law_path, 'r', encoding="utf-8") as f:
        laws = json.load(f)

    target_keys = ["title", "summary"]
    model_path = "BM-K/KoDiffCSE-RoBERTa"
    model_max_length = 512
    model, tokenizer = get_embedding_model(model_path, model_max_length)

    data_src = preprocess_data(dataset, target_keys)
    data_trg = preprocess_data(examples, target_keys)

    q_embeds_src = get_embeddings(model, tokenizer, data_src["target"])
    q_embeds_trg = get_embeddings(model, tokenizer, data_trg["target"])

    scores = calculate_score(q_embeds_src, q_embeds_trg)
    best_index = scores.argmax(axis=1)
    examples = [examples[i] for i in best_index]

    law_id_to_idx = {law["doc_id"]:i for i, law in enumerate(laws)}
 
    def get_prompt(data, example):
        example_passages = "\n".join([laws[law_id_to_idx[doc_id]]["passage"] for doc_id in example["doc_id"]])
        example_prompt = f"""다음은 제목, 질문요약, 관련법령, 답변을 보고 예상되는 질문을 생성한 예시입니다.
###제목
{example["title"]}

###질문요약
{example["summary"]}

###관련법령
{example_passages}

###답변
{example["answer"]}

###질문
{example["question"]}


"""

        input_passages = "\n".join(data["law"])
        input_prompt = f"""앞의 예시를 참고하여 아래의 제목, 질문요약, 관련법령, 답변을 보고 예상되는 질문을 생성해주세요. 생성된 질문이 아래 제목과 요약의 내용을 잘 반영해주세요. 생성된 질문에 관련법령 내용을 반영하여 응답하면 아래 답변이 나오도록 질문을 생성해 주세요.
###제목
{data["title"]}

###질문요약
{data["summary"]}

###관련법령
{input_passages}

###답변
{data["answer"]}

###질문
"""
        return example_prompt + input_prompt

    save_path = DATA_PATH[split]["output"]
    if os.path.exists(save_path.format(ext="json")):
        with open(save_path.format(ext="json"), 'r', encoding="utf-8") as f:
            resume_dataset = json.load(f)
        dataset = resume_dataset + dataset[len(resume_dataset):]

    is_failed = False
    for i, (data, example) in tqdm.tqdm(enumerate(zip(dataset, examples)), total=len(dataset)):
        if "question" in data:
            continue

        try:
            prompt = get_prompt(data, example)
            response, model = get_response(
                prompt,
                system_prompt='당신은 제목, 질문요약, 관련법령, 답변을 보고 예상되는 질문을 생성하는 AI입니다.',
                agent=agent
            )
            data["question"] = response
            data["meta"] = {
                "q-gen_model": model,
                "data_source": data_source,
            }
        except:
            traceback.print_exc()
            dataset = dataset[:i]
            is_failed = True
            break
    
    save_dir = os.path.split(save_path)[0]
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path.format(ext="json"), 'w', encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    with open(save_path.format(ext="jsonl"), 'w', encoding="utf-8") as f:
        for j in dataset:
            f.write(json.dumps(j, ensure_ascii=False)+"\n")

    if not is_failed:
        total_data_path = DATA_PATH[split]["total"]
        if total_data_path:
            total_data_source = os.path.basename(total_data_path)
            with open(total_data_path, 'r', encoding="utf-8") as f:
                dataset_total = [json.loads(j) for j in f.readlines() if j]

            for data in dataset_total:
                data["meta"] = {
                    "data_source": total_data_source,
                }
                data["law"] = [laws[law_id_to_idx[doc_id]]["passage"] for doc_id in data["doc_id"]]

            full_dataset = dataset + dataset_total

            full_data_path = DATA_PATH[split]["output_total"]
            with open(full_data_path.format(ext="json"), 'w', encoding="utf-8") as f:
                json.dump(full_dataset, f, ensure_ascii=False)

            with open(full_data_path.format(ext="jsonl"), 'w', encoding="utf-8") as f:
                for j in full_dataset:
                    f.write(json.dumps(j, ensure_ascii=False)+"\n")
