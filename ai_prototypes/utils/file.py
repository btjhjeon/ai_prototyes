import os
import csv
import json
import warnings


def load_data(path, **kwargs):
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    elif path.endswith(".json"):
        return load_json(path)
    elif path.endswith(".csv"):
        return load_csv(path, **kwargs)
    elif path.endswith(".txt"):
        return load_txt(path)


def load_json(path):
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf-8") as jsonls:
        return [json.loads(j) for j in jsonls if j]


def load_csv(path, return_fieldnames=False):
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if return_fieldnames:
            field_names = []
            for name in reader.fieldnames:
                if name not in field_names:
                    field_names.append(name)
            return [d for d in reader], field_names
        else:
            return [d for d in reader]


def load_txt(path):
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf-8") as f:
        return f.readlines()


def write_data(data, path):
    ext = os.path.splitext(path)[1]
    if ext == ".json":
        write_json(data, path)
    elif ext == ".jsonl":
        write_jsonl(data, path)
    elif ext == ".csv":
        write_csv(data, path)
    elif ext == ".txt":
        write_txt(data, path)


def write_json(data, path):
    make_dirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(data, path):
    make_dirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False)+"\n")


def write_csv(data, path, header=None):
    make_dirs(os.path.split(path)[0], exist_ok=True)
    if header is None:
        header = list(data[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, header)
        writer.writeheader()
        writer.writerows(data)


def write_txt(data, path):
    make_dirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if isinstance(data, str):
            f.write(data)
        elif isinstance(data, (list, tuple)):
            data = [d+"\n" for d in data]
            f.writelines(data)


def make_dirs(dirs, exist_ok=True):
    if dirs:
        os.makedirs(os.path.split(dirs)[0], exist_ok=exist_ok)
