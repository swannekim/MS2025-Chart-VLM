from typing import Dict, List
from PIL import Image
import numpy as np
from ultralytics import YOLO
from .base import ClassifierBackend

class UltralyticsBackend(ClassifierBackend):
    def __init__(self, weights_path: str, classes: List[str], imgsz: int = 224):
        super().__init__(classes, imgsz)
        self.model = YOLO(weights_path)  # 분류 가중치(.pt)
        # 모델 내 names가 있으면 우선 사용
        try:
            ul_names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
            if ul_names and len(ul_names) == len(self.classes):
                self.classes = ul_names
        except Exception:
            pass

    def predict(self, img: Image.Image, topk: int = 2) -> Dict:
        res = self.model.predict(img, imgsz=self.imgsz, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()  # shape: [C]
        idx = int(probs.argmax())
        label = self.classes[idx] if idx < len(self.classes) else str(idx)
        prob = float(probs[idx])

        order = np.argsort(-probs)[:topk]
        topk_pairs = [(self.classes[i] if i < len(self.classes) else str(i), float(probs[i])) for i in order]
        probs_map = {self.classes[i] if i < len(self.classes) else str(i): float(probs[i]) for i in range(len(probs))}
        return {"label": label, "prob": prob, "probs": probs_map, "topk": topk_pairs}
