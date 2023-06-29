from typing import List, Dict, Optional
import requests
import json
import string
import random

import gradio as gr

from ai_prototypes.language import openai_api, glm_api


BASE_MEMORY_TEMPLATE = [
    "안녕? 나에 대해 소개해줄께. 내 이름은 {user_name}이야. {ups_summary}",
    "나는 {character_name}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요.",
    "{conv_summary}",
    "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.",
]


class ConversationBot:
    def __init__(self, agent):
        self.agent = agent
        self.user_name = agent.user_name
        self.character_name = agent.character_name
        self.ups_summary = ""
        self.conv_summary =""

        self.user_history = [None] * agent.base_memory_script_len()
        self.update_s0("", "")

    def update_s0(self, ups_summary: str, conv_summary: str):
        self.ups_summary = ups_summary
        self.conv_summary = conv_summary
        scripts_len = agent.base_memory_script_len()
        self.user_history[:scripts_len] = self.agent.get_memory_script(self.ups_summary, self.conv_summary)

    def clear_history(self):
        del self.user_history[agent.base_memory_script_len():]

    def conversation(self, utterance: str):
        response = self.agent.conversation(utterance, self.user_history)
        self.user_history = response["history"]
        return response["answer"]


if __name__ == "__main__":
    agent = glm_api.AdotAgent("http://172.27.30.115", "철수")
    # agent = openai_api.ChatGPTAgent("복돌이")
    bot = ConversationBot(agent)

    ups_summary = "나는 30대 남성이며 어쿠스틱 음악을 좋아함. 특히 피아노 치는것이 취미이며 쇼팽을 존경함."
    conv_summary = "아내는 새우 알러지가 있음. 나는 자장면을 싫어함."
    # conv_summary = "나는 축구를 하다 정강이가 부러졌었음. 나는 자장면을 싫어함."
    bot.update_s0(ups_summary, conv_summary)

    # messages = [
    #     "아내와 먹을 저녁을 만들어야 하는데, 새우 요리도 괜찮을까?",
    #     "그럼 뭐가 좋을까?",
    # ]

    # for m in messages:
    #     print(f"human: {m}")
    #     out = bot.conversation(m)
    #     print(f"agent: {out}")

    with gr.Blocks() as demo:
        with gr.Row():
            text_ups_summary = gr.Textbox(label="ups_summary", value=ups_summary, interactive=True)
            text_conv_summary = gr.Textbox(label="conv_summary", value=conv_summary, interactive=True)
        text_chatbot = gr.Chatbot()
        text_msg = gr.Textbox()
        button_clear = gr.Button("Clear chat")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def assistant(history):
            bot_message = bot.conversation(history[-1][0])
            print(bot.user_history)
            print("\n")
            history[-1][1] = bot_message
            return history

        def clear():
            bot.clear_history()
            return None
        
        def update_ups_summary(value):
            bot.update_s0(value, bot.conv_summary)
            return value
        
        def update_conv_summary(value):
            bot.update_s0(bot.ups_summary, value)
            return value

        text_msg.submit(user, [text_msg, text_chatbot], [text_msg, text_chatbot], queue=False).then(
            assistant, text_chatbot, text_chatbot
        )
        button_clear.click(clear, None, text_chatbot, queue=False)
        text_ups_summary.change(update_ups_summary, text_ups_summary)
        text_conv_summary.change(update_conv_summary, text_conv_summary)

        demo.launch(server_name="0.0.0.0", server_port=6006)