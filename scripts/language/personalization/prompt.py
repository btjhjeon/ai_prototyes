from abc import *
from typing import List
from datetime import datetime
from copy import deepcopy


USER_PREFIX = "화자1"
AGENT_PREFIX = "화자2"
PRE_CONTEXT = "시스템"
SEP_TOKEN = "<unused0>"
EOD_TOKEN = "</d>"


def build_prompt_template(type_no, **kwargs):
    if type_no is None:
        return build_prompt_broker_template(**kwargs)
    else:
        return TEMPLATE_CLASSES[type_no-1](**kwargs)


def build_prompt_broker_template(type_no=3, **kwargs):
    return BROKER_TEMPLATE_CLASSES[type_no-1](**kwargs)


class PromptTemplateBase(metaclass=ABCMeta):
    USER_NAME_TOKEN = "<USER>"
    AGENT_NAME_TOKEN = "<AGENT>"
    UPS_TOKEN = "<UPS_SUM>"
    CONV_TOKEN = "<CONV_SUM>"

    def get_glm_prompt(self, scripts):
        prompt = ''
        for p, u in scripts:
            if p is PRE_CONTEXT:
                prompt += f"{u}"
            else:
                prompt += f"{p}: {u}{SEP_TOKEN} "
        prompt += f"{AGENT_PREFIX}:"
        return prompt

    def get_glm_label(self, scripts):
        label = ''
        for p, u in scripts:
            if p is PRE_CONTEXT:
                label += f"{u}"
            else:
                label += f"{p}: {u}{SEP_TOKEN} "

        return label.strip()

    # def get_glm_label(self, scripts):
    #     label = ''
    #     for i, (p, u) in enumerate(scripts):
    #         if p is PRE_CONTEXT:
    #             label += f"{u}"
    #         else:
    #             if i < len(scripts) - 1:
    #                 label += f"{p}: {u}{SEP_TOKEN} "
    #             else:
    #                 label += f"{p}: {u}{EOD_TOKEN}"
    #     return label.strip()

    def print(self, scripts):
        txt = ""
        for p, u in scripts:
            if p is PRE_CONTEXT:
                txt += f'{u}\n'
            else:
                txt += f'{p}: {u}\n'
        print(txt)
        return txt

    def append(self, scripts, user_utterance=None, agent_utterance=None):
        scripts = deepcopy(scripts)
        if user_utterance is not None:
            scripts.append((USER_PREFIX, user_utterance))
        if agent_utterance is not None:
            scripts.append((AGENT_PREFIX, agent_utterance))
        return scripts

    def generate(self, user_name,
                 agent_name = "에이닷",
                 ups_summary: str="",
                 conv_summary: str=""):
        template = self.template

        prompts = []
        for t in template:
            player = t[0]
            utterance = t[1]
            utterance = utterance.replace(self.USER_NAME_TOKEN, user_name)
            utterance = utterance.replace(self.AGENT_NAME_TOKEN, agent_name)
            utterance = utterance.replace(self.UPS_TOKEN, ups_summary)
            utterance = utterance.replace(self.CONV_TOKEN, conv_summary)
            prompts.append((player, utterance))
        return prompts

    def get_dialog_history(self, script):
        template_len = len(self.template)
        assert len(script) >= template_len, 'The argument "script" have to include the personalization template.'
        return script[template_len:]

    @property
    @abstractmethod
    def template(self) -> List[tuple]:
        pass


class PromptBrokerTemplateBase(PromptTemplateBase):
    YEAR_TOKEN = "<YEAR>"
    MONTH_TOKEN = "<MONTH>"
    DATE_TOKEN = "<DATE>"
    DAY_TOKEN = "<DAY>"
    HOUR_TOKEN = "<HOUR>"

    def generate(self, user_name,
                 agent_name = "에이닷",
                 ups_summary: str="",
                 conv_summary: str=""):
        current_datetime = datetime.now()
        template = self.control_template + self.template

        prompts = []
        for player, utterance in template:
            utterance = utterance.replace(self.YEAR_TOKEN, str(current_datetime.year))
            utterance = utterance.replace(self.MONTH_TOKEN, str(current_datetime.month))
            utterance = utterance.replace(self.DATE_TOKEN, str(current_datetime.day))
            utterance = utterance.replace(self.DAY_TOKEN, get_weekday_kr(current_datetime))
            utterance = utterance.replace(self.HOUR_TOKEN, str(current_datetime.hour))
            utterance = utterance.replace(self.USER_NAME_TOKEN, user_name)
            utterance = utterance.replace(self.AGENT_NAME_TOKEN, agent_name)
            utterance = utterance.replace(self.UPS_TOKEN, ups_summary)
            utterance = utterance.replace(self.CONV_TOKEN, conv_summary)
            prompts.append((player, utterance))
        return prompts

    @property
    def control_template(self) -> List[tuple]:
        control_template = [
            (PRE_CONTEXT, "당신의 질문에 최선을 다해 알려드릴게요. 그러나 인공지능 챗봇으로서, 욕설에 대해서는 응답을 드릴 수가 없습니다. 사회 통념상 부적절하거나 사회적 규범에 어긋나는 질문과 요청에 대해서는  응답을 드릴 수가 없습니다. 성별(남녀) 관련한 성적 내용이나 선정적인 질문과 요청에 대해서는 응답을 드릴 수가 없습니다. 특정 정당/종교에 대해서는 중립적인 입장이며, 정치에 대해서는 개인적인 견해를 밝힐 수 없습니다. 특정 사람, 기업인, 정치인 등에 대해서는 개인이나 단체별로 나름의 의견을 가질 수 있으나, 인공지능 챗봇으로서 개인적인 의견을 가지고 있지 않습니다. "),
            (USER_PREFIX, f"안녕~ 반가워~ 난 {self.USER_NAME_TOKEN}(이)야. 지금은 {self.YEAR_TOKEN}년 {self.MONTH_TOKEN}월 {self.DATE_TOKEN}일 {self.DAY_TOKEN}요일 {self.HOUR_TOKEN}시야. 내가 알려준 정보로 대화하자."),
            (AGENT_PREFIX, f"난 {self.AGENT_NAME_TOKEN}(이)야~ 너와 재밌고 다채로운 이야기를 하고 싶어"),
            (USER_PREFIX, "넌 할 줄 아는게 뭐야?"),
            (AGENT_PREFIX, "난 대화하는 것을 좋아하고, 다 알진 못하지만 많은 질문에 답할 수 있어."),
        ]
        return control_template

    def get_dialog_history(self, script):
        template = self.control_template + self.template
        template_len = len(template)
        assert len(script) >= template_len, 'The argument "script" have to include the control template and the personalization template.'
        return script[template_len:]


class DialogPromptTemplate(PromptTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, f"안녕? 나에 대해 소개해줄게. 내 이름은 {self.USER_NAME_TOKEN}(이)야. {self.UPS_TOKEN}"),
            (AGENT_PREFIX, f"나는 {self.AGENT_NAME_TOKEN}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요."),
            (USER_PREFIX, f"{self.CONV_TOKEN}"),
            (AGENT_PREFIX, "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
        ]
        return prompt_template


class AgentDialogPromptTemplate(PromptTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, f"안녕? 내 이름은 {self.USER_NAME_TOKEN}(이)야."),
            (AGENT_PREFIX, f"나는 {self.AGENT_NAME_TOKEN}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 제가 알고 있는 당신에 대한 부가 정보는 \"{self.UPS_TOKEN}\" 입니다."),
            (USER_PREFIX, "예전에 대화로 알려줬던 정보도 있어?"),
            (AGENT_PREFIX, f"네, 대화로 알려주신 정보는 \"{self.CONV_TOKEN}\" 입니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
        ]
        return prompt_template


class NoPersonalizationBrokerTemplate(PromptBrokerTemplateBase):
    @property
    def template(self) -> List[tuple]:
        return []


class DialogPromptBrokerTemplate(PromptBrokerTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, f"안녕? 나에 대해 소개해줄게. 내 이름은 {self.USER_NAME_TOKEN}(이)야. {self.UPS_TOKEN}"),
            (AGENT_PREFIX, f"저는 {self.AGENT_NAME_TOKEN}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요."),
            (USER_PREFIX, f"{self.CONV_TOKEN}"),
            (AGENT_PREFIX, "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
        ]
        return prompt_template


class AgentDialogPromptBrokerTemplate(PromptBrokerTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, "나에 대해 알고 있는 정보가 있어?"),
            (AGENT_PREFIX, f"내가 알고 있는 너에 대한 정보는 너의 이름은 {self.USER_NAME_TOKEN}(이)고, \"{self.UPS_TOKEN}\" 야."),
            (USER_PREFIX, "예전에 대화로 알려줬던 정보도 있어?"),
            (AGENT_PREFIX, f"네, 대화로 알려주신 정보는 \"{self.CONV_TOKEN}\" 입니다. 저는 {self.AGENT_NAME_TOKEN}(이)고, 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
        ]
        return prompt_template


# class AgentDialogKeywordPromptTemplate(PromptTemplateBase):
#     @property
#     def template(self) -> List[tuple]:
#         prompt_template = [
#             (USER_PREFIX, f"안녕? 내 이름은 {self.USER_NAME_TOKEN}(이)야."),
#             (AGENT_PREFIX, f"나는 {self.AGENT_NAME_TOKEN}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 제가 알고 있는 당신에 대한 부가 정보는 \"{self.UPS_TOKEN}\" 입니다."),
#             (USER_PREFIX, "예전에 대화로 알려줬던 정보도 있어?"),
#             (AGENT_PREFIX, f"네, 대화로 알려주신 정보는 \"{self.CONV_TOKEN}\" 입니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
#         ]
#         return prompt_template


# class PrefixPromptTemplate(PromptTemplateBase):
#     @property
#     def template(self) -> List[tuple]:
#         prompt_template = [
#             # (AGENT_PREFIX, "키워드 앞에 좋아하는 것은 \"+\", 싫어하는 것은 \"-\", 정보는 \"#\"를 붙여서 표현해 주세요."),
#             (USER_PREFIX, f"안녕? 나에 대해 소개해줄게. 내 이름은 {self.USER_NAME_TOKEN}(이)야. {self.UPS_TOKEN}"),
#             (AGENT_PREFIX, f"나는 {self.AGENT_NAME_TOKEN}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요."),
#             (USER_PREFIX, f"{self.CONV_TOKEN}"),
#             (AGENT_PREFIX, "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
#         ]
#         return prompt_template


class ListPromptTemplate(PromptTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, f"안녕? 나에 대해 소개해줄게. 내 이름은 {self.USER_NAME_TOKEN}(이)야. 나의 정보는 다음과 같아:\n{self.UPS_TOKEN}"),
            (AGENT_PREFIX, f"저는 {self.AGENT_NAME_TOKEN}입니다. 당신과 재밌고 다채로운 이야기를 하고 싶습니다. 과거에 있었던 일, 좋아하거나 싫어하는 것 등 부가 정보를 알려 주세요."),
            (USER_PREFIX, f"나의 부가 정보는 다음과 같아:\n{self.CONV_TOKEN}"),
            (AGENT_PREFIX, "네, 감사합니다. 필요한 경우에는 알려주신 정보를 바탕으로 답변 하겠습니다.")
        ]
        return prompt_template


# class InstructionPromptTemplate(PromptTemplateBase):
#     @property
#     def template(self) -> List[tuple]:
#         prompt_template = [
#             (PRE_CONTEXT, \
# f"""다음은 USER({self.USER_NAME_TOKEN})와 AGENT({self.AGENT_NAME_TOKEN}) 간의 친근한 대화입니다.
# AGENT는 아래 나의 신상 정보, 선호 정보, 애완동물 정보를 참조하여 나의 질문에 구체적인 세부 정보를 수다스럽게 제공합니다.
# AGENT가 질문에 대한 답을 모른다면 솔직히 모른다고 말합니다.

# USER의 신상 정보
# {self.UPS_TOKEN}

# USER에 대한 선호 정보
# {self.CONV_TOKEN}

# 대화 시작!
# """)
#         ]
#         return prompt_template


class InstructionPromptTemplate(PromptTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, f"""안녕? 나는 {self.USER_NAME_TOKEN}(이)야. 내 정보를 알려줄테니 이 정보로 대화를 하자.
USER의 신상 정보
{self.UPS_TOKEN}

USER에 대한 선호 정보
{self.CONV_TOKEN}
"""),
            (AGENT_PREFIX, f"안녕하세요. 저는 {self.AGENT_NAME_TOKEN}입니다. 대화를 시작해 주세요.")
        ]
        return prompt_template

class AwesomePromptTemplate(PromptTemplateBase):
    @property
    def template(self) -> List[tuple]:
        prompt_template = [
            (USER_PREFIX, \
f"""나는 너가 AI 챗봇 으로써 행동해 주길 원해. 나와 너의 1:1 대화이며, 내가 너에게 질문을 입력할꺼야. 너는 아래 나의 기본정보와 내 과거 대화를 고려해서 질문에 대해 응답해줘야해.
나는 {self.USER_NAME_TOKEN}(이)고, {self.UPS_TOKEN}. 
그리고 {self.CONV_TOKEN}

응답할때, 너가 누군지 말하지마. 인사하지마. 처음에 질문요약을 하면서 시작하지마. 
넌 오직 질문에 대한 답변만 말해줘야되. 구체적인 답변을 해달라고 하지말고, 너가 알아서 구체적으로 설명해줘야해.
"""),
            (AGENT_PREFIX, f"저는 {self.AGENT_NAME_TOKEN}입니다. 요청하신 데로 답변해 드리겠습니다.")
        ]
        return prompt_template


CONTROL_TEMPLATE_CLASSES = [
    DialogPromptTemplate,
    AgentDialogPromptTemplate,
    # PrefixPromptTemplate,
    ListPromptTemplate,
    InstructionPromptTemplate,
    AwesomePromptTemplate,
    # AgentDialogKeywordPromptTemplate,
]

BROKER_TEMPLATE_CLASSES = [
    NoPersonalizationBrokerTemplate,
    DialogPromptBrokerTemplate,
    AgentDialogPromptBrokerTemplate
]

TEMPLATE_CLASSES = CONTROL_TEMPLATE_CLASSES + BROKER_TEMPLATE_CLASSES
