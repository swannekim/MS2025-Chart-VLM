from typing import Dict, List
from PIL import Image
import numpy as np
import torch
from .base import ClassifierBackend
from ..utils.image_utils import preprocess_for_cls, softmax

class TorchscriptBackend(ClassifierBackend):
    def __init__(self, weights_path: str, classes: List[str], imgsz: int = 224):
        super().__init__(classes, imgsz)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(weights_path, map_location=self.device)
        self.model.eval()

    def predict(self, img: Image.Image, topk: int = 2) -> Dict:
        chw, _ = preprocess_for_cls(img, self.imgsz)
        x = torch.from_numpy(chw).unsqueeze(0).to(self.device)  # [1,3,H,W]
        with torch.no_grad():
            out = self.model(x)  # 로짓 가정: [1,C] 또는 [C]
        logits = out[0] if out.ndim > 1 else out
        logits = logits.detach().cpu().numpy().astype("float32")
        probs = softmax(logits)
        idx = int(probs.argmax())
        label = self.classes[idx] if idx < len(self.classes) else str(idx)
        prob = float(probs[idx])

        order = np.argsort(-probs)[:topk]
        topk_pairs = [(self.classes[i] if i < len(self.classes) else str(i), float(probs[i])) for i in order]
        probs_map = {self.classes[i] if i < len(self.classes) else str(i): float(probs[i]) for i in range(len(probs))}
        return {"label": label, "prob": prob, "probs": probs_map, "topk": topk_pairs}
