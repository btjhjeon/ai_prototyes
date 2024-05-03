from ..api import get_response
from . import google


def translate(input, keep_str=None, agent="openai", model='gpt-3.5-turbo'):
    if len(input) <= 1:
        return None, None

    if agent == "google":
        return google.translate(input), agent

    else:
        system_prompt = "I want you to act as an Korean translator, spelling corrector and improver." \
                        "I will speak to you in English and you will translate it and answer in the corrected and improved version of my text, in Korean." \
                        "I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level Korean words and sentences." \
                        "Keep the meaning same, but make them more literary." \
                        "I want you to only reply the correction, the improvements and nothing else, do not write explanations."
        user_prefix_prompt = "Please translate the following into Korean:\n"
        if keep_str:
            inputs = input.split(keep_str)
        else:
            inputs = [input]
        outputs = [get_response(
                        user_prefix_prompt + p,
                        system_prompt=system_prompt,
                        agent=agent,
                        model=model
                    ) if p else ("", "") for p in inputs]
        model = " & ".join(set([output[1] for output in outputs if output[1]]))
        outputs = [output[0] for output in outputs]
        if keep_str:
            outputs = keep_str.join(outputs)
        else:
            outputs = outputs[0]
        return outputs, model

