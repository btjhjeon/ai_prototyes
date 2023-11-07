import os
import fire

from ai_prototypes.utils.file import load_data, write_csv


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


def _get_result(results, doc_id):
    for r in results:
        if int(r["doc_id"]) == int(doc_id)+300:
            return r
    return None


def merge(
    data_path:str,
    result_path:str,
    output_path:str
):
    data, field_names = load_data(data_path, return_fieldnames=True)
    results = load_data(result_path)

    result_name = os.path.splitext(os.path.split(result_path)[1])[0]
    field_names.append(result_name)

    count = 0
    doc_id = None
    content_id = None
    found_result = False
    output = []
    for row in data:
        content_id_prev = content_id
        content_id = row[HEADER_ID].strip() 
        # assert content_id, f"Empty row in line no {i+2}"

        if row[HEADER_SESS]:
            doc_id = row[HEADER_SESS]
            result = _get_result(results, doc_id)
            found_result = result is not None
            count += found_result
            if found_result:
                output.append(row)

        elif not found_result:
            continue

        elif content_id.startswith("s"):
            output.append(row)
            if "b" in content_id:
                if content_id == content_id_prev:
                    continue

                no = int(content_id.split('_')[1][1:])/2
                talks = result["talks"]

                t_no = 0
                response = ""
                for talk in talks:
                    t_no += talk["not_trainable"] == 0
                    if t_no == no:
                        response = talk["model_output"]
                        break
                
                row[result_name] = response
    
    write_csv(output, output_path, field_names)


if __name__ == "__main__":
    fire.Fire(merge)
