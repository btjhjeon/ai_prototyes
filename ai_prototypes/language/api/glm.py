from typing import List, Dict, Optional
import requests
import json

from ai_prototypes.utils.language import get_random_string


GLM_URL = 'http://172.27.30.115/v1/glm'
GLM_HEADERS = {'Content-type': 'application/json;charset=utf-8'}


class AdotAgent:
    def __init__(self, host: str, user_name: str, character_name: str = "아폴로", device_id: Optional[str] = None, ):
        self.host = host if host else GLM_URL
        self.user_name = user_name
        self.character_name = character_name

        self.device_id = get_random_string(20) if device_id == None else device_id
        self.user_id = "ZZALDEVBLZPMAZ05341CD5"
        self.transaction_id = "zz0f640f1399017601861333e23543e788"

    def conversation(self, utterance: str, history: List[Dict]):
        def inference():
            params = {
                "text": utterance,
                "origin_text": utterance,
                "user_history": history,
                "meta": {
                    "transaction_id": self.transaction_id,
                    "user_id": self.user_id,
                    "device_id": self.device_id,
                    "has_character_name": False,
                    "character_name" : self.character_name,
                    "user_name" : self.user_name,
                    "is_test": True,
                    "is_glm_only": True,
                }
            }
            res = requests.post(f"{self.host}/v1/glm/inference", headers=GLM_HEADERS, json=params)
            res.raise_for_status()

            contents = json.loads(res.content)
            # print(contents['BODY']['debugInfo'])
            return contents['BODY']['result']['system_utterance']['utterance']
        
        system_utterance = inference()
        new_history = history + [{
            "user": utterance,
            "agent": system_utterance,
            "source": "GLM"
        }]
        return {
            "answer": system_utterance,
            "history": new_history
        }
    
    def base_memory_script_len(self):
        return 2
    
    def get_memory_script(self, ups_summary, conv_summary):
        scripts = []
        scripts.append({
            "user": f"안녕? 나에 대해 소개해줄께. 내 이름은 {self.user_name}이야. {ups_summary}",
            "agent": f"나는 {self.character_name}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요.",
            "source": "INIT"
        })
        scripts.append({
            "user": f"{conv_summary}",
            "agent": "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.",
            "source": "INIT"
        })
        return scripts


def request_inference(prompt):
#     prompt = \
# f"""다음은 나(사용자)와 AI 간의 친근한 대화입니다. AI는 수다스럽고 맥락에서 많은 구체적인 세부 정보를 제공합니다. AI가 질문에 대한 답을 모른다면 솔직히 모른다고 말합니다.

# 현재 대화:
# 나는 어제 저녁에 버스를 타고 백화점에 다녀옴. 백화점에서 간식과 빵, 고기를 구매함.

# 나: {input}
# AI:
# """
    device_id = get_random_string()
    params = {
      "text": prompt,
      "origin_text": prompt,
      "meta": {
        "transaction_id": "zz0f640f1399017601861333e23543e788",
        "user_id": "ZZALDEVBLZPMAZ05341CD5",
        # "device_id": "ZZALDFK5ZNV81EC833295F",
        "device_id": device_id,
        "has_character_name": False,
        "character_name" : "아폴로",
        "user_name" : "홍길동",
        "birthdate" : "1990-01-23"
      }
    }
    res = requests.post(GLM_URL + "/inference", headers=GLM_HEADERS, json=params)
    if res.ok:
        contents = json.loads(res.content)
        print("[GLM]  : {}".format(contents['BODY']['result']['system_utterance']['utterance']))

