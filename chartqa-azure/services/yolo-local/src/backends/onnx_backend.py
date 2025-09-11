from typing import Dict, List
from PIL import Image
import numpy as np
import onnxruntime as ort
from .base import ClassifierBackend
from ..utils.image_utils import preprocess_for_cls, softmax

class ONNXBackend(ClassifierBackend):
    def __init__(self, weights_path: str, classes: List[str], imgsz: int = 224):
        super().__init__(classes, imgsz)
        self.sess = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img: Image.Image, topk: int = 2) -> Dict:
        chw, _ = preprocess_for_cls(img, self.imgsz)   # CHW, 0~1
        x = np.expand_dims(chw, 0).astype("float32")   # [1,3,H,W]
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        logits = out[0] if out.ndim > 1 else out
        probs = softmax(logits.astype("float32"))
        idx = int(np.argmax(probs))
        label = self.classes[idx] if idx < len(self.classes) else str(idx)
        prob = float(probs[idx])

        order = np.argsort(-probs)[:topk]
        topk_pairs = [(self.classes[i] if i < len(self.classes) else str(i), float(probs[i])) for i in order]
        probs_map = {self.classes[i] if i < len(self.classes) else str(i): float(probs[i]) for i in range(len(probs))}
        return {"label": label, "prob": prob, "probs": probs_map, "topk": topk_pairs}
