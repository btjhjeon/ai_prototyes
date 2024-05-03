import warnings
import tiktoken

from . import (
    openai,
    anthropic,
    glm
)


def get_response(prompt, system_prompt=None, agent="openai", model="gpt-3.5-turbo"):
    if agent == "openai":
        enc = tiktoken.get_encoding("cl100k_base")
        token_len = len(enc.encode(prompt))

        if model in "gpt-3.5-turbo" and token_len > 16000:
            model = "gpt-3.5-turbo-16k"

        result = openai.request_inference(prompt, system_prompt, model)

    elif agent == "anthropic":
        model = "claude-2"
        warnings.warn("\"{model}\" API doesn't support system_prompt.")
        result = anthropic.request_inference(prompt, model, max_tokens_to_sample=8192)

    else:
        raise NotImplementedError(f"\"{agent}\" not supported yet!")

    return result, model

