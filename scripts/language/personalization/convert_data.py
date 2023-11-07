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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", nargs='+')
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def convert(
    data_path:List[str],
    output_path:str # require the jsonl file extension (ex. temp.jsonl)
):
    data = []
    for path in data_path:
        with open(path, "r", encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            data.extend([d for d in reader])
    
    content_id = None
    jsonl_data = []
    json_data = None
    for i, d in enumerate(data):
        content_id_prev = content_id
        content_id = d[HEADER_ID].strip()
        assert content_id, f"Empty row in line no {i+2}"
        content = d[HEADER_CONTENT].strip() if HEADER_CONTENT in d else d[HEADER_CONTENT_EDIT].strip()

        if d[HEADER_SESS]:
            if json_data is not None:
                jsonl_data.append(json_data)
            user_name = d[HEADER_USER_NAME].strip().replace('[', '').replace(']이에요', '')
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
            json_data["doc_id"] = str(d[HEADER_SESS])
            json_data["meta"] = OrderedDict()
            json_data["meta"]["negative_data"] = False
            json_data["meta"]["bot_tone"] = d[HEADER_TONE_BOT].strip()
            json_data["meta"]["user_tone"] = d[HEADER_TONE_USER].strip()
            json_data["meta"]["datetime"] = d[HEADER_DATETIME].strip()
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

            if persona_no == 5:
                template = build_prompt_template(random.randint(1, len(CONTROL_TEMPLATE_CLASSES)))
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
                    "not_trainable": 32
                })
            elif "b" in content_id:
                if content_id != content_id_prev:
                    json_data["talks"].append({
                        "speaker": BOT_ID,
                        "msg": content,
                        "ctrl": [],
                        "not_trainable": 0
                    })
                else:
                    json_data["talks"][-1]["msg"] += f"\n{content}"
            else:
                raise NotImplementedError(f"Invalid ID: {content_id}")
    jsonl_data.append(json_data)

    with open(output_path, "w", encoding="utf-8") as f:
        for json_data in jsonl_data:
            json.dump(json_data, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    convert(args.data_path, args.output_path)
