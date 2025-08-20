import urllib.request
import urllib.error
import json

from dotenv import load_dotenv
import os
import io
import base64
import requests
from PIL import Image  # ← RGB 변환용

# .env
load_dotenv()

# get environment variables
rest_api_key = os.getenv("REST_API_KEY")
rest_ep_url = os.getenv("REST_ENDPOINT_URL")

if not rest_api_key:
    raise Exception("A key should be provided to invoke the endpoint")
if not rest_ep_url:
    raise Exception("REST_ENDPOINT_URL not set")

# 1) 이미지 → RGB → PNG → data URI
def image_url_to_data_uri_rgb(image_url: str) -> str:
    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")  # RGB 강제 변환
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # 알파 제거된 RGB PNG로 저장
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# 샘플 이미지
image_url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/1379.png"
img_src = image_url_to_data_uri_rgb(image_url)

# ChartGemma 권장 스타일: program of thought
prompt = (
    "Tell me all the information you get from this chart."
)

# 2) TGI 규격: 마크다운 이미지 + 질문
inputs_str = f"![]({img_src})\n{prompt}\n\n"

print(f"Inputs string:\n{inputs_str}\n")

payload = {
    "inputs": inputs_str,
    # "parameters": {"max_new_tokens": 256, "do_sample": False}
}

body = json.dumps(payload).encode("utf-8")

url = rest_ep_url
api_key = rest_api_key
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': ('Bearer ' + api_key)
}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    print(result.decode("utf-8"))
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))