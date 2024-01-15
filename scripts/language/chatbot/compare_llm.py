from ai_prototypes.language.api import openai, glm


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

    def get_base_scripts(self):
        scripts_len = self.agent.base_memory_script_len()
        return self.user_history[:scripts_len]

    def clear_history(self):
        del self.user_history[self.agent.base_memory_script_len():]

    def conversation(self, utterance: str):
        response = self.agent.conversation(utterance, self.user_history)
        self.user_history = response["history"]
        return response["answer"]


if __name__ == "__main__":
    adot_agent = glm.AdotAgent("http://172.27.30.115", "철수")
    adot_bot = ConversationBot(adot_agent)
    chatgpt_agent = openai.ChatGPTAgent("철수")
    chatgpt_bot = ConversationBot(chatgpt_agent)

    ups_summary = "나는 9살 남자 어린이이며 6살 동생이 있음."
    conv_summary = "나는 포켓몬스터를 좋아함. 포켓몬스터 스티커를 모으는데 동생이 자꾸 뺏음."

    adot_bot.update_s0(ups_summary, conv_summary)
    chatgpt_bot.update_s0(ups_summary, conv_summary)
    base_scripts = adot_bot.get_base_scripts()

    for script in base_scripts:
        print(f"[HUMAN]   : {script['user']}")
        print(f"[BOT]     : {script['agent']}")

    while True:
        try:
            m = input("[HUMAN]   : ")
            adot_out = adot_bot.conversation(m)
            print(f"[ADOT]    : {adot_out}")
            chatgpt_out = chatgpt_bot.conversation(m)
            print(f"[CHATGPT] : {chatgpt_out}")
        except KeyboardInterrupt:
            break
