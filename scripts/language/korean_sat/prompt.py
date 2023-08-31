

def get_prompt(
    data,
    paragraph_num:int,
    problem_num:int,
    need_integrated:bool=False,
    num_shot=-1
):
    paragraph = data[paragraph_num]
    problem = paragraph["problems"][problem_num]

    if num_shot < 0:
        if "type" in problem:
            prompt_func = get_prompt_by_type(int(problem["type"]))
        else:
            prompt_func = get_prompt_by_type(int(paragraph["type"]))
    else:
        prompt_func = fewshot_prompt

    no_paragraph = "no_paragraph" in problem
    if "question_plus" in problem:
        question_plus_text = problem["question_plus"]
    else:
        question_plus_text = ""
    return prompt_func(paragraph=data[paragraph_num]["paragraph"],
                       question=problem["question"],
                       choices=problem["choices"],
                       question_plus=question_plus_text,
                       need_integrated=need_integrated,
                       no_paragraph=no_paragraph,
                       num_shot=num_shot)


def get_prompt_by_type(type_num: int) -> callable:
    # 0 : 비문학, 1 : 문학, 2 : 화법과 작문, 3 : 문법
    if type_num == 0:
        return literature_prompt
    elif type_num == 1:
        return literature_prompt
    elif type_num == 2:
        return literature_prompt
    else:
        return grammar_prompt


def fewshot_prompt(
    paragraph,
    question,
    choices,
    question_plus="",
    need_integrated=False,
    no_paragraph=False,
    num_shot=0,
    **kwargs
):
    if num_shot == 0:
        example = ""
    elif num_shot == 1:
        example = """### 지문:
" 다음 글을 읽고 물음에 답하시오. 어떤 독서 이론도 이 한 장의 사진만큼 독서의 위대함을 분명하게 말해 주지 못할 것이다. 사진은 제2차 세계 대전 당시 처참하게 무너져 내린 런던의 한 건물 모습이다. ㉠(폐허 속에서도 사람들이 책을 찾아 서가 앞에 선 이유는 무엇일까?) 이들은 갑작스레 닥친 상황에서 독서를 통해 무언가를 구하고자 했을 것이다.독서는 자신을 살피고 돌아볼 계기를 제공함으로써 어떻게 살 것인가의 문제를 생각하게 한다. 책은 인류의 지혜와 경험이 담겨 있는 문화유산이며, 독서는 인류와의 만남이자 끝없는 대화이다. 독자의 경험과 책에 담긴 수많은 경험들의 만남은 성찰의 기회를 제공함으로써 독자의 내면을 성장시켜 삶을 바꾼다. 이런 의미에서 독서는 자기 성찰의 행위이며, 성찰의 시간은 깊이 사색하고 스스로에게 질문을 던지는 시간이어야 한다. 이들이 책을 찾은 것도 혼란스러운 현실을 외면하려 한 것이 아니라 자신의 삶에 대한 숙고의 시간이 필요했기 때문이다.또한 ㉡(독서는 자신을 둘러싼 현실을 올바로 인식하고 당면한 문제를 해결할 논리와 힘을 지니게 한다.) 책은 세상에 대한 안목을 키우는 데 필요한 지식을 담고 있으며, 독서는 그 지식을 얻는 과정이다. 독자의 생각과 오랜 세월 축적된 지식의 만남은 독자에게 올바른 식견을 갖추고 당면한 문제를 해결할 방법을 모색하도록 함으로써 세상을 바꾼다. 세상을 변화시킬 동력을 얻는 이 시간은 책에 있는 정보를 이해하는 데 그치는 것이 아니라 그 정보가 자신의 관점에서 문제를 해결할 수 있는 타당한 정보인지를 판단하고 분석하는 시간이어야 한다. 서가 앞에 선 사람들도 시대적 과제를 해결할 실마리를 책에서 찾으려 했던 것이다.독서는 자기 내면으로의 여행이며 외부 세계로의 확장이다. 폐허 속에서도 책을 찾은 사람들은 독서가 지닌 힘을 알고, 자신과 현실에 대한 이해를 구하고자 책과의 대화를 시도하고 있었던 것이다."
위 지문을 읽고 다음 문제를 푸시오.
### 문제: 윗글을 바탕으로 할 때, ㉠의 답으로 적절하지 않은 것은?
### 선택지:
1. 인류의 지혜와 경험을 배우기 위해
2. 현실로부터 도피할 방법을 구하기 위해
3. 시대적 과제를 해결할 실마리를 찾기 위해
4. 자신의 삶에 대해 숙고할 시간을 갖기 위해
5. 세상에 대한 안목을 키우는 지식을 얻기 위해
### 정답: 2. 현실로부터 도피할 방법을 구하기 위해

"""
    if question_plus:
        prompt = f"""{example}### 지문:
"{paragraph}"
### 보기:
"{question_plus}"
위 지문과 보기을 읽고 다음 문제를 푸시오.
### 문제: {question}
### 선택지:
1. {choices[0]}
2. {choices[1]}
3. {choices[2]}
4. {choices[3]}
5. {choices[4]}
### 정답: """
    else:
        prompt = f"""{example}### 지문:
"{paragraph}"
위 지문을 읽고 다음 문제를 푸시오.
### 문제: {question}
### 선택지
1. {choices[0]}
2. {choices[1]}
3. {choices[2]}
4. {choices[3]}
5. {choices[4]}
### 정답: """
    if need_integrated:
        return prompt
    return {
        "user_prompt": prompt
    }


def basic_prompt(
    paragraph,
    question,
    choices,
    question_plus="",
    need_integrated=False,
    no_paragraph=False,
    **kwargs
):
    system_prompt = """
        국어 시험 문제를 푸는 똑똑한 학생으로써 다음 문제의 답을 구하세요.
        지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.
    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}

        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 
        정답 : """

    if need_integrated:
        return system_prompt + "\n\n" + user_prompt
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


def talk_prompt(
    paragraph,
    question,
    choices,
    question_plus="",
    need_integrated=False,
    no_paragraph=False,
    **kwargs
):
    system_prompt = """
        국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

         문제를 풀이할 때, 반드시 지문을 참고하세요.
         문제는 무조건 1개의 정답만 있습니다.
         문제를 풀이할 때 모든 선택지들을 검토하세요.
         모든 선택지마다 근거를 지문에서 찾아 설명하세요.

         다음의 형식을 따라 답변하세요.

        최종 정답: (최종 정답)
         1번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 1번에 대한 답변)
         2번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 2번에 대한 답변)
         3번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 3번에 대한 답변)
         4번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 4번에 대한 답변)
         5번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 5번에 대한 답변)

    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            이 문제는 아래와 같이 <보기>가 주어져 있습니다. 문제의 각 선택지들을 해결하기 위한 배경 지식을 설명해 주고 있는 것이 <보기>로써, 각 선택지들을 지문과 연결시키고, <보기>의 지식을 활용하면 각 선택지의 참과 거짓을 판단할 수 있습니다.
            문제를 해결할 때, 반드시 <보기>의 내용을 이용해서 문제를 해결해야 합니다.
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}

        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 
        정답 : """

    if need_integrated:
        return system_prompt + "\n\n" + user_prompt
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


def literature_prompt(
    paragraph,
    question,
    choices,
    question_plus="",
    need_integrated=False,
    no_paragraph=False,
    **kwargs
):
    system_prompt = """
        국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

         문제를 풀이할 때, 반드시 지문을 참고하세요.
         문제는 무조건 1개의 정답만 있습니다.
         문제를 풀이할 때 모든 선택지들을 검토하세요.
         모든 선택지마다 근거를 지문에서 찾아 설명하세요.

         다음의 형식을 따라 답변하세요.
         최종 정답: (최종 정답)
         1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
         2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
         3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
         4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
         5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"

    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            이 문제는 아래와 같이 <보기>가 주어져 있습니다. 문제의 각 선택지들을 해결하기 위한 배경 지식을 설명해 주고 있는 것이 <보기>로써, 각 선택지들을 지문과 연결시키고, <보기>의 지식을 활용하면 각 선택지의 참과 거짓을 판단할 수 있습니다.
            문제를 해결할 때, 반드시 <보기>의 내용을 이용해서 문제를 해결해야 합니다.
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}

        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 
        정답 : """

    if need_integrated:
        return system_prompt + "\n\n" + user_prompt
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


def grammar_prompt(
    paragraph,
    question,
    choices,
    question_plus="",
    need_integrated=False,
    no_paragraph=False,
    **kwargs
):
    system_prompt = """
        당신은 국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 최종 정답을 고르시오.

        '지문 속 목적어의 성격'과 '선택지 속 목적어의 성격'이 서로 같은 선택지를 1개만 고르세요.
        모두 같은 선택지는 무조건 1개만 존재합니다.

        문제를 풀이할 때 5개의 모든 선택지를 검토하세요.

        자료나 돈처럼 실제 손으로 만질 수 있는 것은 '실제적인 단어'입니다.
        관심, 집중, 인기 이론처럼, 실제 손으로 만질 수 없는 것은 '추상적인 단어'입니다.

        다음의 형식대로만 답변하세요.
        최종 정답: (지문 속 목적어와 선택지 속 목적어의 성격이 서로 같은 선택지는 "(최종 정답)"입니다.
        1번: - 지문 속 동사ⓐ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓐ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        2번: - 지문 속 동사ⓑ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓑ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        3번: - 지문 속 동사ⓒ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓒ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        4번: - 지문 속 동사ⓓ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓓ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        5번: - 지문 속 동사ⓔ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓔ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"

    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            이 문제는 아래와 같이 <보기>가 주어져 있습니다. 문제의 각 선택지들을 해결하기 위한 배경 지식을 설명해 주고 있는 것이 <보기>로써, 각 선택지들을 지문과 연결시키고, <보기>의 지식을 활용하면 각 선택지의 참과 거짓을 판단할 수 있습니다.
            문제를 해결할 때, 반드시 <보기>의 내용을 이용해서 문제를 해결해야 합니다.
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}
        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 
        정답 : """
    if need_integrated:
        return system_prompt + "\n\n" + user_prompt
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


if __name__ == "__main__":
    import os, json, tqdm
    data_path = "2023.json"
    output_dir = 'korean_sat/2023_prompt/'
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, 'r') as f:
        data = json.load(f)

    # 0-shot prompt
    no = 0
    for i, d in tqdm.tqdm(enumerate(data)):
        for j, p in enumerate(d['problems']):
            prompt = get_prompt(data, i, j, True, 0)
            no += 1

            output_path = os.path.join(output_dir, f"{no:02d}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(prompt)


    # CoT prompt
    output_dir = 'korean_sat/2023_prompt_CoT/'
    os.makedirs(output_dir, exist_ok=True)

    no = 0
    for i, d in tqdm.tqdm(enumerate(data)):
        for j, p in enumerate(d['problems']):
            prompt = get_prompt(data, i, j, True)
            no += 1

            output_path = os.path.join(output_dir, f"{no:02d}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(prompt)
