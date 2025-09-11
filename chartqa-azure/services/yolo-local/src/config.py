import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return str(v) if v is not None else None

BACKEND    = _get("BACKEND", "ultralytics").lower()  # ultralytics | torchscript | onnx
MODEL_PATH = _get("MODEL_PATH", "./models/yolo_cls/best.pt")
MODEL_CARD = _get("MODEL_CARD", "./models/yolo_cls/model_card.json")
IMGSZ      = int(_get("IMGSZ", "224"))
TOPK       = int(_get("TOPK", "2"))

HOST       = _get("HOST", "127.0.0.1")
PORT       = int(_get("PORT", "7001"))
LOG_LEVEL  = _get("LOG_LEVEL", "INFO").upper()

def ensure_path(p: str) -> Path:
    return Path(p).expanduser().resolve()
