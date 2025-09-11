import argparse
from pathlib import Path
from PIL import Image
import json

from .config import BACKEND, MODEL_PATH, MODEL_CARD, IMGSZ, TOPK, ensure_path
from .backends import UltralyticsBackend, TorchscriptBackend, ONNXBackend

def load_classes(model_card: Path) -> list[str]:
    if not model_card.exists():
        return ["chart", "nonchart"]
    j = json.loads(model_card.read_text(encoding="utf-8"))
    cls_map = j.get("classes") or {}
    return [v for k, v in sorted(cls_map.items(), key=lambda x: int(x[0]))] or ["chart","nonchart"]

def build_backend():
    classes = load_classes(ensure_path(MODEL_CARD))
    mp = str(ensure_path(MODEL_PATH))
    if BACKEND == "torchscript":
        return TorchscriptBackend(mp, classes, imgsz=IMGSZ)
    elif BACKEND == "onnx":
        return ONNXBackend(mp, classes, imgsz=IMGSZ)
    else:
        return UltralyticsBackend(mp, classes, imgsz=IMGSZ)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="이미지 파일 또는 폴더")
    ap.add_argument("--topk", type=int, default=TOPK)
    args = ap.parse_args()

    backend = build_backend()
    p = Path(args.path)
    exts = {".jpg",".jpeg",".png",".bmp",".gif"}
    files = [p] if p.is_file() else sorted([x for x in p.glob("*") if x.suffix.lower() in exts])
    for f in files:
        im = Image.open(f).convert("RGB")
        res = backend.predict(im, topk=args.topk)
        print(f"{f.name}: {res['label']} ({res['prob']:.4f})  topk={res['topk']}")

if __name__ == "__main__":
    main()
