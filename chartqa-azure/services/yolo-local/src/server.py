# services/yolo-local/src/server.py

import json
from pathlib import Path
from typing import Optional, Tuple

import base64
import io
import logging

import requests
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from .config import (
    BACKEND, MODEL_PATH, MODEL_CARD, IMGSZ, TOPK, HOST, PORT, LOG_LEVEL, ensure_path
)
from .backends import UltralyticsBackend, TorchscriptBackend, ONNXBackend
from .utils.io_utils import pil_from_base64  # 그대로 사용 가능 (data URI도 처리 가능하면 우선 사용)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"))
log = logging.getLogger("yolo-local")

app = FastAPI(title="YOLO Classifier (Local)")

# ---------------------------
# Model & Backend bootstrap
# ---------------------------
def load_classes_from_card(p: Path) -> list[str]:
    if not p.exists():
        log.warning("model_card.json not found; fallback to ['chart','nonchart']")
        return ["chart", "nonchart"]
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        classes_map = j.get("classes") or {}
        classes = [v for k, v in sorted(classes_map.items(), key=lambda x: int(x[0]))]
        return classes or ["chart", "nonchart"]
    except Exception as e:
        log.warning("failed to read model_card.json: %s; use default classes", e)
        return ["chart", "nonchart"]

MODEL_PATH_P = ensure_path(MODEL_PATH)
CLASSES = load_classes_from_card(ensure_path(MODEL_CARD))

def build_backend():
    if BACKEND == "torchscript":
        log.info(f"Backend: TorchScript -> {MODEL_PATH_P}")
        return TorchscriptBackend(str(MODEL_PATH_P), CLASSES, imgsz=IMGSZ)
    elif BACKEND == "onnx":
        log.info(f"Backend: ONNX -> {MODEL_PATH_P}")
        return ONNXBackend(str(MODEL_PATH_P), CLASSES, imgsz=IMGSZ)
    else:
        log.info(f"Backend: Ultralytics -> {MODEL_PATH_P}")
        return UltralyticsBackend(str(MODEL_PATH_P), CLASSES, imgsz=IMGSZ)

BACKEND_IMPL = build_backend()

# ---------------------------
# Request Models & Helpers
# ---------------------------
class ClassifyJSON(BaseModel):
    # JSON 본문에서 허용할 여러 alias 필드 (아무 하나만 채워도 됨)
    image_b64: Optional[str] = None
    image_data_uri: Optional[str] = None
    image: Optional[str] = None
    image_base64: Optional[str] = None
    image_bytes_b64: Optional[str] = None
    image_url: Optional[str] = None
    return_probs: Optional[bool] = True
    topk: Optional[int] = None  # JSON으로 topk를 보낼 수도 있음

def _strip_data_uri(s: str) -> str:
    """
    data:image/png;base64,..... → 콤마 뒤의 실 base64만 반환
    """
    if not isinstance(s, str):
        return s
    i = s.find(",")
    if s[:5].lower() == "data:" and i != -1:
        return s[i + 1 :]
    return s

def _decode_b64_to_bytes(s: str) -> bytes:
    pure = _strip_data_uri(s)
    return base64.b64decode(pure, validate=False)

def _image_from_bytes(b: bytes) -> Image.Image:
    if not b or len(b) < 200:
        raise HTTPException(status_code=400, detail=f"image too small (bytes={len(b) if b else 0})")
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"not an image: {e}")

def _load_image_from_json(json_body: ClassifyJSON) -> Tuple[Image.Image, bool]:
    """
    JSON 바디에서 이미지를 로드. (URL > base64 aliases)
    returns: (PIL.Image, return_probs)
    """
    # 1) URL 지원
    if json_body.image_url:
        try:
            resp = requests.get(json_body.image_url, timeout=20)
            resp.raise_for_status()
            im = _image_from_bytes(resp.content)
            return im, (json_body.return_probs is not False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"image_url fetch failed: {e}")

    # 2) 다양한 base64 alias 중 하나
    cand = (
        json_body.image_b64
        or json_body.image_data_uri
        or json_body.image
        or json_body.image_base64
        or json_body.image_bytes_b64
    )
    if cand:
        try:
            # util이 data uri를 알아서 처리한다면 그대로 사용
            try:
                im = pil_from_base64(cand).convert("RGB")
            except Exception:
                # util 실패 시 수동 처리
                im = _image_from_bytes(_decode_b64_to_bytes(cand))
            return im, (json_body.return_probs is not False)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64")
    raise HTTPException(
        status_code=400,
        detail="missing image payload: expected one of "
               "['image_b64','image_data_uri','image','image_base64','image_bytes_b64','image_url']"
    )

def _load_image_from_multipart(file: UploadFile | None, image_b64_form: Optional[str]) -> Image.Image:
    """
    multipart/form-data 경로:
      - file= 업로드 파일
      - 또는 image_b64= base64 문자열
    """
    if file is not None:
        try:
            return Image.open(file.file).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid file: {e}")
    if image_b64_form:
        try:
            # util 우선
            try:
                return pil_from_base64(image_b64_form).convert("RGB")
            except Exception:
                return _image_from_bytes(_decode_b64_to_bytes(image_b64_form))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64 in form field")
    raise HTTPException(status_code=400, detail="no file or image_b64 provided")

def _resolve_topk(json_topk: Optional[int], form_topk: Optional[int]) -> int:
    if isinstance(json_topk, int) and json_topk > 0:
        return json_topk
    if isinstance(form_topk, int) and form_topk > 0:
        return form_topk
    return TOPK

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": BACKEND_IMPL.name,
        "imgsz": IMGSZ,
        "classes": CLASSES
    }

@app.post("/classify")
async def classify(
    # JSON 바디(선호)
    payload: Optional[ClassifyJSON] = Body(default=None),
    # multipart/form-data 대체 입력도 허용
    file: UploadFile = File(default=None),
    image_b64_form: Optional[str] = Form(default=None, alias="image_b64"),
    topk_form: Optional[int] = Form(default=None, alias="topk"),
):
    try:
        # 1) 이미지 로드
        if payload is not None:
            img, want_probs = _load_image_from_json(payload)
            topk = _resolve_topk(payload.topk, topk_form)
        else:
            img = _load_image_from_multipart(file, image_b64_form)
            want_probs = True
            topk = _resolve_topk(None, topk_form)

        # 2) 추론
        res = BACKEND_IMPL.predict(img, topk=topk)

        # 3) 응답 정규화
        out = {
            "label": res.get("label"),
            "prob": float(res.get("prob", 0.0)),
            "backend": BACKEND_IMPL.name,
            "imgsz": IMGSZ
        }
        if want_probs:
            # probs/topk가 없을 수도 있으니 관대하게
            if "probs" in res and isinstance(res["probs"], (list, tuple)):
                # 클래스별 맵으로 치환
                try:
                    out["probs"] = {CLASSES[i]: float(p) for i, p in enumerate(res["probs"])}
                except Exception:
                    out["probs"] = res["probs"]
            if "topk" in res:
                out["topk"] = res["topk"]

        return JSONResponse(out)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Inference error")
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.yolo-local.src.server:app", host=HOST, port=PORT, reload=False)

# import json
# from pathlib import Path
# from typing import Optional

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from PIL import Image

# from .config import BACKEND, MODEL_PATH, MODEL_CARD, IMGSZ, TOPK, HOST, PORT, LOG_LEVEL, ensure_path
# from .backends import UltralyticsBackend, TorchscriptBackend, ONNXBackend
# from .utils.io_utils import pil_from_base64

# import logging
# logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"))
# log = logging.getLogger("yolo-local")

# app = FastAPI(title="YOLO Classifier (Local)")

# def load_classes_from_card(p: Path) -> list[str]:
#     if not p.exists():
#         log.warning("model_card.json not found; fallback to ['chart','nonchart']")
#         return ["chart", "nonchart"]
#     j = json.loads(p.read_text(encoding="utf-8"))
#     classes_map = j.get("classes") or {}
#     classes = [v for k, v in sorted(classes_map.items(), key=lambda x: int(x[0]))]
#     return classes or ["chart", "nonchart"]

# MODEL_PATH_P = ensure_path(MODEL_PATH)
# CLASSES = load_classes_from_card(ensure_path(MODEL_CARD))

# def build_backend():
#     if BACKEND == "torchscript":
#         log.info(f"Backend: TorchScript -> {MODEL_PATH_P}")
#         return TorchscriptBackend(str(MODEL_PATH_P), CLASSES, imgsz=IMGSZ)
#     elif BACKEND == "onnx":
#         log.info(f"Backend: ONNX -> {MODEL_PATH_P}")
#         return ONNXBackend(str(MODEL_PATH_P), CLASSES, imgsz=IMGSZ)
#     else:
#         log.info(f"Backend: Ultralytics -> {MODEL_PATH_P}")
#         return UltralyticsBackend(str(MODEL_PATH_P), CLASSES, imgsz=IMGSZ)

# BACKEND_IMPL = build_backend()

# class ClassifyIn(BaseModel):
#     image_b64: str

# @app.get("/health")
# def health():
#     return {"status": "ok", "backend": BACKEND_IMPL.name, "imgsz": IMGSZ, "classes": CLASSES}

# @app.post("/classify")
# async def classify(
#     payload: Optional[ClassifyIn] = None,
#     file: UploadFile = File(default=None),
#     topk: int = Form(default=TOPK)
# ):
#     try:
#         if payload and payload.image_b64:
#             img = pil_from_base64(payload.image_b64)
#         elif file is not None:
#             img = Image.open(file.file).convert("RGB")
#         else:
#             return JSONResponse({"error": "image_b64 or file is required"}, status_code=400)

#         res = BACKEND_IMPL.predict(img, topk=topk)
#         return {
#             "label": res["label"],
#             "prob": res["prob"],
#             "probs": res.get("probs"),
#             "topk": res.get("topk"),
#             "backend": BACKEND_IMPL.name,
#             "imgsz": IMGSZ
#         }
#     except Exception as e:
#         log.exception("Inference error")
#         return JSONResponse({"error": str(e)}, status_code=500)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("services.yolo-local.src.server:app", host=HOST, port=PORT, reload=False)
