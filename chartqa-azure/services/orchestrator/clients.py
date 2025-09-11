# orchestrator/clients.py

import os
import json
import requests
from typing import Optional
import re
import base64

# =========================
# YOLO classifier (local/remote)
# =========================
YOLO_URL: Optional[str] = os.getenv("YOLO_URL")  # e.g., http://127.0.0.1:7001/classify
YOLO_KEY: Optional[str] = os.getenv("YOLO_KEY")  # optional
YOLO_TIMEOUT: int = int(os.getenv("YOLO_TIMEOUT", "60"))

def _b64_to_bytes(image_b64: str) -> bytes:
    s = (image_b64 or "").strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]  # data URI면 헤더 제거
    return base64.b64decode(s, validate=False)

def call_yolo_cls(image_b64: str, topk: int = 2) -> dict | None:
    """YOLO 엔드포인트가 Form/File을 기대하므로 멀티파트로 전송."""
    if not YOLO_URL:
        return None
    headers = {"Accept": "application/json"}
    if YOLO_KEY:
        headers["Authorization"] = f"Bearer {YOLO_KEY}"

    img_bytes = _b64_to_bytes(image_b64)
    files = {"file": ("chart.png", img_bytes, "image/png")}
    data = {"topk": str(topk)}
    r = requests.post(YOLO_URL, headers=headers, files=files, data=data, timeout=YOLO_TIMEOUT)
    r.raise_for_status()
    return r.json()

# =========================
# ChartGemma (AML TGI /generate, key auth) - base64 슬림 + 다중 스키마 폴백
# =========================
import io
import base64
import time
from typing import Any, Dict, List, Tuple
from PIL import Image

CG_URL: Optional[str] = os.getenv("CHARTGEMMA_URL", "").rstrip("/")
CG_KEY: str = os.getenv("CHARTGEMMA_KEY", "")
CG_DEPLOYMENT: str = os.getenv("CHARTGEMMA_DEPLOYMENT", "")
CG_TIMEOUT: int = int(os.getenv("CHARTGEMMA_TIMEOUT", "180"))
CG_MAX_NEW_TOKENS = int(os.getenv("CHARTGEMMA_MAX_NEW_TOKENS", "512"))
CG_TEMPERATURE    = float(os.getenv("CHARTGEMMA_TEMPERATURE", "0.1"))

# ---- 인라인 data URI용 슬림 파라미터 (토큰 초과 방지) ----
# 대략 1토큰≈4문자 가정, 8192-512 여유 고려해 28k 기본
CG_MAX_B64_CHARS = int(os.getenv("CHARTGEMMA_MAX_B64_CHARS", "28000"))
CG_IMG_MAX_W     = int(os.getenv("CHARTGEMMA_IMG_MAX_W", "768"))
CG_IMG_MAX_H     = int(os.getenv("CHARTGEMMA_IMG_MAX_H", "768"))
CG_JPEG_QUALITY  = int(os.getenv("CHARTGEMMA_JPEG_QUALITY", "80"))
CG_JPEG_QUALITY_FLOOR = int(os.getenv("CHARTGEMMA_JPEG_QUALITY_FLOOR", "35"))

CG_SYS = os.getenv(
    "CHARTGEMMA_SYS",
    "You are a chart-reading VLM. The chart image is provided inline. "
    "Read values and trends precisely (include % if shown). "
    "Be concise; if approximating, say 'approx.'; show arithmetic for differences."
)

_DISC_RE = re.compile(
    r"(cannot\s+analy[sz]e\s+images|can[’']?t\s+analy[sz]e\s+images|"
    r"as\s+an\s+ai\s+language\s+model|i\s+cannot\s+view\s+images|"
    r"cannot\s+create\s+(an\s+)?image|can[’']?t\s+create\s+images?)",
    re.I,
)

def _headers_json() -> Dict[str, str]:
    if not CG_KEY:
        raise RuntimeError("CHARTGEMMA_KEY is empty. Set endpoint key for key-based auth.")
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {CG_KEY}",
    }
    if CG_DEPLOYMENT:
        h["azureml-model-deployment"] = CG_DEPLOYMENT
    return h

def _clean_answer(ans: str) -> str:
    if not isinstance(ans, str):
        ans = str(ans)
    lines = [ln for ln in ans.splitlines() if not _DISC_RE.search(ln)]
    ans = "\n".join(lines).strip()
    return ans.replace("Answer:", "").replace("A:", "").strip()

def _parse_text_like(j: Any) -> str | None:
    if isinstance(j, dict):
        for k in ("generated_text","text","output_text","output","answer","result"):
            v = j.get(k)
            if isinstance(v, str) and v.strip():
                return v
        try:
            v = j["choices"][0]["message"]["content"]
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
        if "predictions" in j and j["predictions"]:
            pv = _parse_text_like(j["predictions"][0])
            if pv: return pv
        if "outputs" in j and j["outputs"]:
            ov = j["outputs"][0] if isinstance(j["outputs"], list) else j["outputs"]
            if isinstance(ov, str) and ov.strip():
                return ov
            pv = _parse_text_like(ov)
            if pv: return pv
    if isinstance(j, list) and j and isinstance(j[0], dict):
        for k in ("generated_text","text","output_text","output","answer","result"):
            v = j[0].get(k)
            if isinstance(v, str) and v.strip():
                return v
    if isinstance(j, str) and j.strip():
        return j
    return None

def _b64_to_bytes(image_b64: str) -> bytes:
    s = (image_b64 or "").strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s, validate=False)

def _shrink_b64_for_tgi(image_b64: str) -> tuple[str, dict]:
    """
    data URI 인라인 시 토큰 초과 방지를 위해 JPEG로 리사이즈/재인코딩.
    목표: base64 길이 <= CG_MAX_B64_CHARS
    """
    raw = _b64_to_bytes(image_b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")

    # 1) 1차 리사이즈
    w,h = im.size
    scale = min(1.0, CG_IMG_MAX_W/max(1,w), CG_IMG_MAX_H/max(1,h))
    if scale < 1.0:
        im = im.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.LANCZOS)

    q = CG_JPEG_QUALITY
    attempts = []
    out_b64 = ""
    for step in range(8):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        attempts.append({"step": step, "w": im.size[0], "h": im.size[1], "q": q, "b64_len": len(b64)})
        out_b64 = b64
        if len(b64) <= CG_MAX_B64_CHARS:
            break
        # 더 줄이기: 크기↓, 품질↓
        if max(im.size) > 256:
            im = im.resize((max(1,int(im.size[0]*0.85)), max(1,int(im.size[1]*0.85))), Image.LANCZOS)
        if q > CG_JPEG_QUALITY_FLOOR:
            q = max(CG_JPEG_QUALITY_FLOOR, int(q*0.85))

    return out_b64, {"attempts": attempts}

def call_chartgemma(chart_b64: str, prompt: str) -> tuple[str, dict]:
    """
    (answer:str, info:dict) 반환.
    우선 TGI /generate에 data URI 방식으로 보내되,
    base64 길이를 강제로 줄여 토큰 한도를 넘지 않게 함.
    실패 시 몇 가지 대안 스키마 폴백.
    """
    if not CG_URL:
        return ("[ChartGemma not configured: set CHARTGEMMA_URL]", {"mode_used": "not_configured"})

    # base64 슬림 처리
    slim_b64, slim_info = _shrink_b64_for_tgi(chart_b64)
    data_uri = f"data:image/jpeg;base64,{slim_b64}"

    md = f"![]({data_uri})\n\n{CG_SYS}\n\nQuestion: {prompt}\nAnswer:"

    headers = _headers_json()
    tried: List[dict] = []

    def _try(payload: dict, mode: str):
        t0 = time.time()
        r = requests.post(CG_URL, headers=headers, json=payload, timeout=CG_TIMEOUT)
        info = {
            "mode": mode,
            "status": r.status_code,
            "latency_ms": int((time.time() - t0) * 1000),
            "req_size": len(json.dumps(payload)) if payload else 0,
        }
        if not (200 <= r.status_code < 300):
            info["body"] = r.text[:500]
            tried.append(info)
            return None, info
        try:
            j = r.json()
            ans_raw = _parse_text_like(j) or json.dumps(j)[:4000]
            return _clean_answer(ans_raw), {**info, "raw_json": True}
        except Exception:
            txt = r.text.strip()
            ans_raw = _parse_text_like(txt) or txt[:4000]
            return _clean_answer(ans_raw), {**info, "raw_json": False}

    # 1) 표준 TGI inputs 경로
    tgi_payload = {
        "inputs": md,
        "parameters": {
            "max_new_tokens": CG_MAX_NEW_TOKENS,
            "temperature": CG_TEMPERATURE,
            "do_sample": False,
            "details": False
        }
    }
    ans, meta = _try(tgi_payload, "tgi_inputs_data_uri")
    if ans:
        return ans, {"mode_used": "tgi_inputs_data_uri", "b64_slim": slim_info, **meta}

    # 2) (혹시) image_b64 + prompt custom 스키마
    ans, meta = _try(
        {
            "image_b64": slim_b64,
            "prompt": prompt,
            "system_prompt": CG_SYS,
            "max_new_tokens": CG_MAX_NEW_TOKENS,
            "temperature": CG_TEMPERATURE,
        },
        "json_b64_prompt"
    )
    if ans:
        return ans, {"mode_used": "json_b64_prompt", "b64_slim": slim_info, **meta}

    # 3) instances/parameters
    ans, meta = _try(
        {
            "instances": [{"image_b64": slim_b64, "prompt": prompt, "system_prompt": CG_SYS}],
            "parameters": {"max_new_tokens": CG_MAX_NEW_TOKENS, "temperature": CG_TEMPERATURE}
        },
        "instances_parameters"
    )
    if ans:
        return ans, {"mode_used": "instances_parameters", "b64_slim": slim_info, **meta}

    # 4) input_data (text-only)
    ans, meta = _try(
        {
            "input_data": {"columns": ["inputs"], "data": [[md]]},
            "params": {"max_new_tokens": CG_MAX_NEW_TOKENS, "temperature": CG_TEMPERATURE}
        },
        "input_data_text"
    )
    if ans:
        return ans, {"mode_used": "input_data_text", "b64_slim": slim_info, **meta}

    # 5) prompt only
    ans, meta = _try(
        {"prompt": md, "max_new_tokens": CG_MAX_NEW_TOKENS, "temperature": CG_TEMPERATURE},
        "prompt_only"
    )
    if ans:
        return ans, {"mode_used": "prompt_only", "b64_slim": slim_info, **meta}

    return ("Chart model call failed.", {"mode_used": "error", "url": CG_URL, "b64_slim": slim_info, "tried": tried})


# # =========================
# # ChartGemma (TGI-like)
# # =========================

# CG_URL: Optional[str] = os.getenv("CHARTGEMMA_URL")       # e.g., http://host:port/generate
# CG_KEY: str = os.getenv("CHARTGEMMA_KEY", "")
# CG_TIMEOUT: int = int(os.getenv("CHARTGEMMA_TIMEOUT", "120"))

# def _normalize_tgi_response(j: dict | list) -> str:
#     """
#     다양한 TGI 변형 응답을 관대하게 파싱.
#     """
#     if isinstance(j, list):
#         if j and isinstance(j[0], dict):
#             for k in ("generated_text", "output_text", "text"):
#                 if k in j[0]:
#                     return str(j[0][k])
#             # 일부 서버는 choices 형태
#             ch = j[0].get("choices")
#             if isinstance(ch, list) and ch and "text" in ch[0]:
#                 return str(ch[0]["text"])
#     elif isinstance(j, dict):
#         for k in ("generated_text", "output_text", "text"):
#             if k in j:
#                 return str(j[k])
#         ch = j.get("choices")
#         if isinstance(ch, list) and ch and "text" in ch[0]:
#             return str(ch[0]["text"])
#     # 안전한 fallback: 원본 json 문자열 일부
#     return json.dumps(j, ensure_ascii=False)[:4000]

# def call_chartgemma(chart_b64: str, prompt: str) -> str:
#     if not CG_URL:
#         return "[ChartGemma not configured: set CHARTGEMMA_URL]"
#     data_uri = f"data:image/png;base64,{chart_b64}"
#     # 다수 TGI 서버가 'inputs' 방식 지원. 불가 시 서버 로그 참고.
#     payload = {"inputs": f"![]({data_uri})\n{prompt}\n\n"}
#     headers = {"Content-Type": "application/json", "Accept": "application/json"}
#     if CG_KEY:
#         headers["Authorization"] = f"Bearer {CG_KEY}"
#     r = requests.post(CG_URL, headers=headers, json=payload, timeout=CG_TIMEOUT)
#     r.raise_for_status()
#     try:
#         j = r.json()
#     except Exception:
#         return r.text[:4000]
#     return _normalize_tgi_response(j)

# =========================
# Azure OpenAI (text route)
# =========================
from openai import AzureOpenAI

AOAI_ENDPOINT: Optional[str]   = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY: Optional[str]        = os.getenv("AZURE_OPENAI_KEY")
AOAI_DEPLOYMENT: str           = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AOAI_API_VER: str              = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AOAI_TEMPERATURE: float        = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.2"))

def _require_env(v: Optional[str], name: str) -> str:
    if not v or not isinstance(v, str) or not v.strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()

def call_aoai(context: str, question: str) -> str:
    endpoint = _require_env(AOAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT")
    key = _require_env(AOAI_KEY, "AZURE_OPENAI_KEY")

    client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=AOAI_API_VER)
    sys_prompt = (
        "You answer questions about a PDF. If the question requires reading a chart image, "
        "say you need a chart image. Otherwise, use the provided OCR text only. Answer concisely."
    )
    # context 길이 보호 — AOAI 입력 크기 제한 완화
    ctx = (context or "")[:12000]
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context from OCR (may be partial):\n{ctx}"},
        {"role": "user", "content": question or ""},
    ]
    resp = client.chat.completions.create(
        model=AOAI_DEPLOYMENT,
        messages=messages,
        temperature=AOAI_TEMPERATURE,
    )
    return resp.choices[0].message.content


# import os, json, requests

# # ---------- YOLO cls (로컬/원격 호환) ----------
# YOLO_URL = os.getenv("YOLO_URL")          # e.g., http://127.0.0.1:7001/classify
# YOLO_KEY = os.getenv("YOLO_KEY")          # 미사용 가능
# USE_AAD_FOR_YOLO = os.getenv("YOLO_USE_AAD", "false").lower() == "true"

# def _aad_bearer_for_aml() -> str:
#     from azure.identity import DefaultAzureCredential
#     cred = DefaultAzureCredential()
#     token = cred.get_token("https://ml.azure.com/.default")
#     return f"Bearer {token.token}"

# def call_yolo_cls(image_b64: str) -> dict | None:
#     """YOLO 엔드포인트 미설정 시 None 반환 → 상위에서 폴백."""
#     if not YOLO_URL:
#         return None
#     headers = {"Content-Type": "application/json", "Accept": "application/json"}
#     if USE_AAD_FOR_YOLO:
#         headers["Authorization"] = _aad_bearer_for_aml()
#     elif YOLO_KEY:
#         headers["Authorization"] = f"Bearer {YOLO_KEY}"
#     payload = {"image_b64": image_b64}
#     r = requests.post(YOLO_URL, headers=headers, json=payload, timeout=60)
#     r.raise_for_status()
#     return r.json()

# # ---------- ChartGemma (TGI style) ----------
# CG_URL = os.getenv("CHARTGEMMA_URL")      # e.g., http://host:port/generate
# CG_KEY = os.getenv("CHARTGEMMA_KEY", "")

# def call_chartgemma(chart_b64: str, prompt: str) -> str:
#     if not CG_URL:
#         return "[ChartGemma not configured: set CHARTGEMMA_URL]"
#     data_uri = f"data:image/png;base64,{chart_b64}"
#     payload = {"inputs": f"![]({data_uri})\n{prompt}\n\n"}
#     headers = {"Content-Type": "application/json", "Accept": "application/json"}
#     if CG_KEY:
#         headers["Authorization"] = f"Bearer {CG_KEY}"
#     r = requests.post(CG_URL, headers=headers, json=payload, timeout=120)
#     r.raise_for_status()
#     j = r.json()
#     if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
#         return j[0]["generated_text"]
#     if isinstance(j, dict) and "generated_text" in j:
#         return j["generated_text"]
#     return json.dumps(j)[:4000]

# # ---------- Azure OpenAI (text route) ----------
# from openai import AzureOpenAI

# AOAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
# AOAI_KEY        = os.getenv("AZURE_OPENAI_KEY")
# AOAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
# AOAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# def call_aoai(context: str, question: str) -> str:
#     client = AzureOpenAI(api_key=AOAI_KEY, azure_endpoint=AOAI_ENDPOINT, api_version=AOAI_API_VER)
#     sys_prompt = (
#         "You answer questions about a PDF. If the question requires reading a chart image, "
#         "say you need a chart image. Otherwise, use the provided OCR text only. Answer concisely."
#     )
#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": f"Context from OCR (may be partial):\n{context[:8000]}"},
#         {"role": "user", "content": question}
#     ]
#     resp = client.chat.completions.create(model=AOAI_DEPLOYMENT, messages=messages, temperature=0.2)
#     return resp.choices[0].message.content