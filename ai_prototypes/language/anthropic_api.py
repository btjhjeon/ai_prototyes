import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


def get_agent(key_path="anthropic.key"):
    assert os.path.exists(key_path)

    with open(key_path, 'r') as f:
        api_key = f.readline().strip()
    return Anthropic(api_key=api_key)


def request_inference(prompt, anthropic, model="claude-2", max_tokens_to_sample=1024, verbos=False):
    completion = anthropic.completions.create(
        model=model,
        max_tokens_to_sample=max_tokens_to_sample,
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
        top_p=0
    )
    result = completion.completion.strip()
    if verbos:
        print(f"[CHAT] : {result}")
    return result


# UNITTEST
if __name__ == "__main__":
    agent = get_agent()
    prompt=f"{HUMAN_PROMPT} Can you help me effectively ask for a raise at work?{AI_PROMPT}"
    request_inference(prompt, agent, verbos=True)
