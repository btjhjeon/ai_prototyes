from ai_prototypes.language import openai_api, glm_api


class ConversationBot:
    def __init__(self, agent):
        self.agent = agent
        self.user_name = agent.user_name
        self.character_name = agent.character_name

        self.user_history = [None] * agent.base_memory_script_len()
        self.update_s0("", "")

    def update_s0(self, ups_summary: str, conv_summary: str):
        scripts_len = self.agent.base_memory_script_len()
        self.user_history[:scripts_len] = self.agent.get_memory_script(ups_summary, conv_summary)

    def clear_history(self):
        del self.user_history[self.agent.base_memory_script_len():]

    def conversation(self, utterance: str):
        response = self.agent.conversation(utterance, self.user_history)
        self.user_history = response["history"]
        return response["answer"]


if __name__ == "__main__":
#     input = "어제 내가 어딜 다녀왔는지 알아?"
#     prompt = \
# f"""아래 정보 중 질문과 관련있는 내용들만 활용하여 AI의 답변을 완성하시오.

# 나는 어제 저녁에 버스를 타고 백화점에 다녀옴.
# 백화점에서 간식과 빵, 고기를 구매함.
# 오래전에 축구를 하다 정강이가 부러져 고생함.
# 아내는 새우 알러지가 있음.

# 나: {input}
# AI:
# """
#     print("[INPUT]")
#     print(prompt)
#     print()
#     print("[USER] : {}".format(input))

#     glm_api.request_inference(prompt)
#     openai_api.request_inference(prompt)

    adot_agent = glm_api.AdotAgent("http://172.27.30.115", "철수")
    adot_bot = ConversationBot(adot_agent)
    chatgpt_agent = openai_api.ChatGPTAgent("철수")
    chatgpt_bot = ConversationBot(chatgpt_agent)

    ups_summary = "나는 30대 남성이며 어쿠스틱 음악을 좋아함. 특히 피아노 치는것이 취미이며 쇼팽을 존경함."
    conv_summary = "나는 축구를 하다 정강이가 부러졌었음. 나는 자장면, 파스타를 싫어함. 아내는 새우 알러지가 있음."
    adot_bot.update_s0(ups_summary, conv_summary)
    chatgpt_bot.update_s0(ups_summary, conv_summary)

    messages = [
        "아내와 저녁식사를 하려는데, 새우 요리가 어떨까?",
        "그럼 뭐가 좋을까?",
    ]

    for m in messages:
        print(f"[HUMAN]   : {m}")
        adot_out = adot_bot.conversation(m)
        print(f"[ADOT]    : {adot_out}")
        chatgpt_out = chatgpt_bot.conversation(m)
        print(f"[CHATGPT] : {chatgpt_out}")