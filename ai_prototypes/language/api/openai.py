import os
import random
import time
from typing import List, Dict

import openai
from openai import OpenAI


def get_client(key_path="openai.key"):
    assert os.path.exists(key_path)

    with open(key_path, 'r') as f:
        api_key = f.readline().strip()
    return OpenAI(api_key=api_key)


client = get_client()


class ChatGPTAgent:
    def __init__(self, user_name: str, character_name: str = "아폴로"):
        self.model = "gpt-3.5-turbo"
        self.user_name = user_name
        self.character_name = character_name

    def conversation(self, utterance: str, history: List[Dict]):
        new_history = history + [{"role": "user", "content": utterance}]
        completion = client.chat.completions.create(
            model=self.model,
            messages=new_history
        )
        system_utterance = completion.choices[0]["message"]["content"]
        new_history = new_history + [{"role": "assistant", "content": system_utterance}]
        return {
            "answer": system_utterance,
            "history": new_history
        }
    
    def base_memory_script_len(self):
        return 4
    
    def get_memory_script(self, ups_summary, conv_summary):
        scripts = []
        scripts.append({
            "role": "user",
            "content": f"안녕? 나에 대해 소개해줄께. 내 이름은 {self.user_name}이야. {ups_summary}",
        })
        scripts.append({
            "role": "assistant",
            "content": f"나는 {self.character_name}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요.",
        })
        scripts.append({
            "role": "user",
            "content": f"{conv_summary}",
        })
        scripts.append({
            "role": "assistant",
            "content": "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.",
        })
        return scripts


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 1.1,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (
        openai.RateLimitError,
        openai.Timeout
    ),
    num_trials: int = 10
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        for _ in range(num_trials):
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
        return ""

    return wrapper


CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
]


@retry_with_exponential_backoff
def request_inference(prompt, system_prompt="", model="gpt-3.5-turbo", verbos=False):
    if model in CHAT_MODELS:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=0
        )
        result = response.choices[0].message.content.strip()
    else:
        response = client.completion.create(
            model=model,
            prompt=prompt,
            top_p=0
        )
        result = response.choices[0].text.strip()

    if verbos:
        print(f"[CHAT] : {result}")
    return result
