import fire
import json
import csv
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.roberta import RobertaModel

from ai_prototypes.language.huggingface import get_model_and_tokenizer


def get_embedding_model(
    model_path:str,
    model_max_length:int=512
):
    model = RobertaModel.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_embeddings(model, tokenizer, data, batch_size=128, return_tensors="np"):
    if not isinstance(data, list):
        data = [data]

    embeds = []
    # model.to('cpu')
    for i in range(0, len(data), batch_size):
        d = data[i:i+batch_size]
        inputs = tokenizer(d, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        embeddings = model(**inputs, return_dict=False)[0]
        if return_tensors == "np":
            embeddings = embeddings.detach().cpu().numpy()
        embeds.append(embeddings[:,0])
    if return_tensors == "np":
        return np.concatenate(embeds)
    elif return_tensors == "pt":
        return torch.cat(embeds)


def calculate_score(a, b):
    if isinstance(a, torch.Tensor):
        if len(a.shape) == 1: a = a.unsqueeze(0)
        if len(b.shape) == 1: b = b.unsqueeze(0)

        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))   
    elif isinstance(a, np.ndarray):
        if len(a.shape) == 1: a = a[np.newaxis,:]
        if len(b.shape) == 1: b = b[np.newaxis,:]

        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.matmul(a_norm, b_norm.T)  


def preprocess_data(data, target_key):
    if isinstance(target_key, str):
        target_key = [target_key]

    data_pair = []
    for d in data:
        pair = {"answer": d['answer']}
        target = [d[key] for key in target_key if key in d and d[key]]
        pair["target"] = "\n".join(target)
        for key in target_key:
            if key in d:
                pair[key] = d[key]
        data_pair.append(pair)

    data_dict = {"answer": [], "target":[]}
    for key in target_key:
        data_dict[key] = []

    if 'doc_id' in data[0]:
        for pair, d in zip(data_pair, data):
            pair["doc_id"] = d['doc_id']
        data_dict["doc_id"] = []
    if 'question' in data[0] and "question" not in data_dict:
        for pair, d in zip(data_pair, data):
            pair["question"] = d['question']
        data_dict["question"] = []
    for d in data_pair:
        for k, v in d.items():
            data_dict[k].append(v)
    return data_dict


def retrieve(
    data_src_path:str,
    data_trg_path:str,
    output_path:str,
    model_path:str,
    model_max_length:int=512,
    target_key:str="question"
):

    with open(data_src_path, 'r', encoding='utf-8') as f:
        data_src = json.load(f)

    with open(data_trg_path, 'r', encoding='utf-8') as f:
        data_trg = json.load(f)

    def _preprocess(data):
        data_pair = [{
            target_key: d[target_key],
            "answer": d['answer']
        } for d in data]
        data_dict = {
            target_key: [],
            "answer": []
        }
        if 'doc_id' in data[0]:
            for pair, d in zip(data_pair, data):
                pair["doc_id"] = d['doc_id']
            data_dict["doc_id"] = []
        if 'question' in data[0] and "question" not in data_dict:
            for pair, d in zip(data_pair, data):
                pair["question"] = d['question']
            data_dict["question"] = []
        for d in data_pair:
            for k, v in d.items():
                data_dict[k].append(v)
        return data_dict

    data_src = preprocess_data(data_src, target_key)
    data_trg = preprocess_data(data_trg, target_key)

    model, tokenizer = get_embedding_model(model_path, model_max_length)

    q_embeds_src = get_embeddings(data_src[target_key])
    q_embeds_trg = get_embeddings(data_trg[target_key])

    scores = calculate_score(q_embeds_src, q_embeds_trg)
    best_index = scores.argmax(axis=1)

    header = ["no", "type", target_key, "answer", "score"]
    if "question" not in header:
        header.insert(2, "question")
    with open(output_path, 'w', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for i, (t, a) in enumerate(zip(data_src[target_key], data_src['answer'])):
            idx = best_index[i]

            print(f"[SOURCE_{i}]")
            print(f"### {target_key.capitalize()}:")
            print(t)
            print("### Answer:")
            print(a)
            row = {
                "no": i+1,
                "type": "source",
                target_key: t,
                "answer": a,
                "score": scores[i][idx]
            }
            if "question" not in row and "question" in data_src:
                row["question"] = data_src["question"][i]
            writer.writerow(row)

            print(f"[TARGET_{i}]")
            print(f"### {target_key.capitalize()}:")
            print(data_trg[target_key][idx])
            print("### Answer:")
            print(data_trg['answer'][idx])
            row = {
                "no": i+1,
                "type": "target",
                target_key: data_trg[target_key][idx],
                "answer": data_trg['answer'][idx],
                "score": scores[i][idx]
            }
            if "question" not in row and "question" in data_trg:
                row["question"] = data_trg["question"][idx]
            writer.writerow(row)
            print()


if __name__ == "__main__":
    fire.Fire(retrieve)
