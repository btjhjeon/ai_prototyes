import os
import json
import random
import fire


def write_jsonl(data, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        f.writelines(data)


def split(
    data_path,
    start_idx=0,
    end_idx=-1,
    test_size=50
):
    random.seed(23)

    data_dir, data_file = os.path.split(os.path.abspath(data_path))
    data_name = os.path.splitext(data_file)[0]
    output_train_path = os.path.join(data_dir, f"{data_name}_train.jsonl")
    output_test_path = os.path.join(data_dir, f"{data_name}_test.jsonl")

    with open(data_path, "r", encoding='utf-8') as f:
        data = f.readlines()

    end_idx = end_idx if end_idx >= 0 else len(data)
    target_indices = list(range(start_idx, end_idx))
    random.shuffle(target_indices)
    test_indices = target_indices[-test_size:]

    train_data, test_data = [], []
    for i, d in enumerate(data):
        if i in test_indices:
            test_data.append(d)
        else:
            train_data.append(d)
    
    write_jsonl(train_data, output_train_path)
    write_jsonl(test_data, output_test_path)


if __name__ == "__main__":
    fire.Fire(split)
