import requests

def translate_eng_to_kor(string, client_id, client_secret):
    url = "https://openapi.naver.com/v1/papago/n2mt"

    req_header = {"X-Naver-Client-Id":client_id, "X-Naver-Client-Secret":client_secret}
    req_param = {"source": "en", "target": "ko", "text": string}
    res = requests.post(url,headers=req_header, data=req_param)

    if res.ok:
        trans_txt = res.json()['message']['result']['translatedText']
        return trans_txt
    else:
        print('error code', res.status_code)

def generate_prompt(string, mood, client_id, client_secret):
    eng_prompt = [f"{string} in the {mood} mood",
                  f"in the {mood} mood, {string}"]
    kor_prompt = [translate_eng_to_kor(prom , client_id, client_secret) for prom in eng_prompt]
    return kor_prompt

