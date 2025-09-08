import base64, io, json, os
from typing import Dict
from PIL import Image
import torch

# Prefer TorchScript; your model asset includes best.torchscript / best.pt / best.onnx
# We load TorchScript and assume class order from your model card:
# index 0 -> "chart", index 1 -> "nonchart"

CLASSES = ["chart", "nonchart"]

# AML mounts registered model under AZUREML_MODEL_DIR
MODEL_DIR = os.getenv("AZUREML_MODEL_DIR", ".")
TS_CANDIDATES = [
    os.path.join(MODEL_DIR, "best.torchscript"),
    os.path.join(MODEL_DIR, "model", "best.torchscript"),
]
MODEL_PATH = next((p for p in TS_CANDIDATES if os.path.exists(p)), None)
model = None

@torch.inference_mode()
def init():
    global model
    if MODEL_PATH is None:
        raise RuntimeError("best.torchscript not found in model asset. Ensure it is included or adjust loader.")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()


def _preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    import numpy as np
    x = (np.asarray(img).astype("float32") / 255.0)
    # ImageNet mean/std
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)  # HWC→CHW
    return torch.from_numpy(x).unsqueeze(0)


def run(raw_data: str) -> str:
    payload = json.loads(raw_data)
    b64 = payload.get("image_b64")
    assert b64, "image_b64 required"
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    x = _preprocess(img)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    # According to model card: 0->chart, 1->nonchart
    p_chart = float(probs[0])
    p_non = float(probs[1]) if probs.numel() > 1 else float(1.0 - probs[0])

    idx = int(torch.argmax(probs).item())
    label = CLASSES[idx] if idx < len(CLASSES) else ("chart" if idx == 0 else "nonchart")

    # Return chart probability under key 'prob' for orchestrator compatibility
    return json.dumps({
        "label": label,
        "prob": p_chart,
        "probs": {"chart": p_chart, "nonchart": p_non}
    })