import os
import argparse
import csv
import json
import random
from typing import List
from collections import OrderedDict

from prompt import CONTROL_TEMPLATE_CLASSES, USER_PREFIX, AGENT_PREFIX, build_prompt_template


HEADER_SESS = "세션 번호"
HEADER_USER_NAME = "페르소나 이름(P0)"
HEADER_TONE_BOT = "Bot 어투"
HEADER_TONE_USER = "User 어투"
HEADER_DATETIME = "datetime"
HEADER_ID = "ID"
HEADER_CONTENT = "대화문"
HEADER_CONTENT_EDIT = "수정 후"
HEADER_STEP = "ADD/STEP"
HEADER_RP_ID = "RP ID"
HEADER_RP = "페르소나 인용"
BOT_NAME = "<bot_name>"
USER_ID = "화자1"
BOT_ID = "화자2"


TEST_SET_INFO = [
    {
        "size": 50,
        "start_idx": 101,
        "end_idx": 250
    },
    {
        "size": 100,
        "start_idx": 350,
        "end_idx": 1000
    }
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", nargs='+')
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def convert(
    data_path:List[str],
    output_path:str # require the jsonl file extension (ex. temp.jsonl)
):
    random.seed(23)

    data_dir, data_file = os.path.split(os.path.abspath(output_path))
    data_name, ext = os.path.splitext(data_file)
    output_train_path = os.path.join(data_dir, f"{data_name}_train{ext}")
    output_test_path = os.path.join(data_dir, f"{data_name}_test{ext}")

    datas = []
    for path in data_path:
        with open(path, "r", encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            datas.append([d for d in reader])
    
    test_indices = []
    for test_info in TEST_SET_INFO:
        test_start_idx = test_info["start_idx"]
        test_end_idx = test_info["end_idx"]
        test_size = test_info["size"]
        target_indices = list(range(test_start_idx, test_end_idx))
        random.shuffle(target_indices)
        test_indices += target_indices[-test_size:]

    content_id = None
    session_id = None
    count = 0
    jsonl_data_all = []
    jsonl_data_train = []
    jsonl_data_test = []
    json_data = None
    for data_no, data in enumerate(datas):
        for i, row in enumerate(data):
            content_id_prev = content_id
            content_id = row[HEADER_ID].strip() 
            assert content_id, f"Empty row in line no {i+2} of \"{data_path[data_no]}\""
            content = row[HEADER_CONTENT].strip() if HEADER_CONTENT in row else row[HEADER_CONTENT_EDIT].strip()

            if row[HEADER_SESS]:
                if json_data is not None:
                    if count in test_indices:
                        jsonl_data_test.append(json_data)
                    else:
                        jsonl_data_train.append(json_data)
                    jsonl_data_all.append(json_data)
                    count += 1

                session_id_prev = session_id
                session_id = f"{data_no+1}_{row[HEADER_SESS]}"

                user_name = row[HEADER_USER_NAME].strip().replace('[', '').replace(']이에요', '')
                user = OrderedDict()
                user["id"] = USER_ID
                user["name"] = user_name
                user["is_bot"] = False
                bot = OrderedDict()
                bot["id"] = BOT_ID
                bot["name"] = BOT_NAME
                bot["is_bot"] = True

                json_data = OrderedDict()
                json_data["source"] = "selectstar"
                json_data["file_name"] = output_path
                json_data["doc_id"] = session_id
                json_data["meta"] = OrderedDict()
                json_data["meta"]["negative_data"] = False
                json_data["meta"]["bot_tone"] = row[HEADER_TONE_BOT].strip()
                json_data["meta"]["user_tone"] = row[HEADER_TONE_USER].strip()
                json_data["meta"]["datetime"] = row[HEADER_DATETIME].strip()
                json_data["meta"]["personas"] = [row[HEADER_USER_NAME].strip(), content]
                json_data["speakers"] = OrderedDict()
                json_data["speakers"][USER_ID] = user
                json_data["speakers"][BOT_ID] = bot
                json_data["talks"] = []
                json_data["target"] = 1

                assert content_id == "P1", f"the ID of start row should be \"P1\", but {content_id}"
                ups_sum = content
                conv_sum = ""

            elif content_id.startswith("P"):
                persona_no = int(content_id[1:])
                if persona_no <= 3:
                    ups_sum += f" {content}"
                else:
                    conv_sum += f" {content}"
                json_data["meta"]["personas"].append(content)

                if persona_no == 5:
                    if count in test_indices:
                        template_no = 2
                    else:
                        template_no = random.randint(1, len(CONTROL_TEMPLATE_CLASSES))
                    template = build_prompt_template(template_no)
                    json_data["meta"]["template_no"] = template_no
                    initial_script = template.generate(user_name, BOT_NAME, ups_sum, conv_sum)

                    for p, u in initial_script:
                        if p == USER_PREFIX:
                            json_data["talks"].append({
                                "speaker": USER_ID,
                                "msg": u,
                                "ctrl": [],
                                "not_trainable": 32
                            })
                        elif p == AGENT_PREFIX:
                            json_data["talks"].append({
                                "speaker": BOT_ID,
                                "msg": u,
                                "ctrl": [],
                                "not_trainable": 32
                            })
                        else:
                            raise NotImplementedError(f"Unknown user type: {p}")
                    assert json_data["talks"][-1]["speaker"] == BOT_ID
            
            elif content_id.startswith("s"):
                if "u" in content_id:
                    json_data["talks"].append({
                        "speaker": USER_ID,
                        "msg": content,
                        "ctrl": [],
                        "not_trainable": 32,
                        "meta": {"ID": content_id}
                    })
                elif "b" in content_id:
                    if content_id != content_id_prev:
                        json_data["talks"].append({
                            "speaker": BOT_ID,
                            "msg": content,
                            "ctrl": [],
                            "not_trainable": 0,
                            "meta": {"ID": content_id}
                        })
                        if row[HEADER_RP_ID].strip():
                            json_data["talks"][-1]['meta']["RP_ID"] = row[HEADER_RP_ID].strip()
                    else:
                        pre_content = json_data["talks"][-1]["msg"]
                        pre_last_sent = pre_content.split("\n")[-1].strip()
                        cur_first_sent = content.split("\n")[0].strip()
                        if pre_last_sent.startswith("-") or pre_last_sent[0].isdigit() or \
                           cur_first_sent.startswith("-") or cur_first_sent[0].isdigit():
                            json_data["talks"][-1]["msg"] += "\n"
                        json_data["talks"][-1]["msg"] += f"\n{content}"
                else:
                    raise NotImplementedError(f"Invalid ID: {content_id}")
    if count in test_indices:
        jsonl_data_test.append(json_data)
    else:
        jsonl_data_train.append(json_data)
    jsonl_data_all.append(json_data)

    with open(output_train_path, "w", encoding="utf-8") as f:
        for json_data in jsonl_data_train:
            json.dump(json_data, f, ensure_ascii=False)
            f.write("\n")

    with open(output_test_path, "w", encoding="utf-8") as f:
        for json_data in jsonl_data_test:
            json.dump(json_data, f, ensure_ascii=False)
            f.write("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for json_data in jsonl_data_all:
            json.dump(json_data, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    convert(args.data_path, args.output_path)
