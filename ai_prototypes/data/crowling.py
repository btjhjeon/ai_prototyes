import os
import sys
import urllib.request
import urllib
import json
import requests

client_id = "fpIp7M8mLQGenh5u4A7t"
client_secret = "tgZ1N4DuNs"
query = "" #"남산" #"경복궁"
save_path = "/t2meta/dataset/korean_place_image/seoul_gates" #"./image/namsan" #"./image/gyeongbokgung_palace"
start = 1       #default 1, max 1000
display = 100    #max 100

query_place = {
    # 사대문
    "북대문": "Bukdaemun",
    "숙정문": "Sukjeongmun",    # 북대문
    "동대문": "Dongdaemun",
    "흥인지문": "Heunginjimun", # 동대문
    "남대문": "Namdaemun",
    "숭례문": "Sungnyemun",     # 남대문
    "서대문": "Seodaemun",
    "돈의문": "Donuimun",       # 서대문
    # 사소문
    "북소문": "Buksomun",
    "창의문": "Changuimun",     # 북소문
    "동소문": "Dongsomun",
    "혜화문": "Hyehwamun",      # 동소문
    "남소문": "Namsomun",
    "광희문": "Gwanghuimun",    # 남소문
    "서소문": "Seosomun",
    "소의문": "Souimun",        # 서소문
    # 주요문
    "광화문" : "Gwanghwamun",
    "대한문" : "Daehanmun",
    "독립문" : "Dongnimmun",
}

# query_place = {
#     "" : "original",
#     "광화문" : "Gwanghwamun",
#     "근정전" : "Geunjeongjeon",
#     "경회루" : "Gyeonghoeru",
#     "국립민속박물관" : "TheNationalFolkMuseumOfKorea",
#     # "어린이박물관" : "The National Folk Museum of Korea(1)",
#     "내부" : "inside"
# }

# query_place = {
#     "" : "original",
#     "서울타워" : "SeoulTower",
#     "케이블카" : "CableCar",
#     "둘레길" : "Trail",
#     "도서관" : "Library",
#     "팔각정" : "OctagonalPavillion"
# }

optional_query = {
    "원경" : "distantView",
    "옆모습" : "sideView",
    "앞모습" : "frontView",
    "밤" : "night",
    "야간" : "night(1)",
    # "야간개장" : "night(2)",
    "야경" : "night(2)",
    "낮" : "daytime",
    "과거" : "past",
    "봄" : "spring",
    "여름" : "summer",
    "가을" : "fall",
    "겨울" : "winter",
    "관광객" : "people"
}

os.makedirs(save_path, exist_ok=True)
for place in list(query_place.keys()):
    query_with_place = " ".join([query, place])

    for option in list(optional_query.keys()):
        query_with_option = " ".join([query_with_place, option])

        print("##############", query_with_option)

        encText = urllib.parse.quote(query_with_option)
        url = f"https://openapi.naver.com/v1/search/image?display={display}&start={start}&query=" + encText # json 결과
        # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            # print(response_body.decode('utf-8'))
            response_body = json.loads(response_body.decode('utf-8'))
            i = start
            # print("length of response body: ", len(response_body["items"]))
            for item in response_body["items"]:
                img_link = item["link"]
                img_height = item["sizeheight"]
                print(place, i)
                img_name = "_".join([query_place[place], optional_query[option], str(i)])
                img_path = os.path.join(save_path, img_name) + '.jpg'
                try:
                    img = requests.get(img_link).content
                    # print(len(img))
                    if len(img) < 1000: #bytes 길이가 너무 짧으면 이미지가 아닐 확률 높음
                        raise Exception("Too short bytes length < 1000")

                    f = open(img_path, 'wb')
                    f.write(img)
                    f.close()
                    print(f'"{img_path}" is saved successfully!!!')
                except:
                    print('@@@@Error occured: #{}, url:{}'.format(i, img_link))
                    img_path = os.path.join(save_path, img_name) + '_error.jpg'
                    f = open(img_path, 'wb')
                    f.close()
                i += 1
                
        else:
            print("Error Code:" + rescode)