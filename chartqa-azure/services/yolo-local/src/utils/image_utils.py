from typing import Tuple
import numpy as np
from PIL import Image

def preprocess_for_cls(im: Image.Image, size: int = 224) -> Tuple[np.ndarray, tuple[int, int]]:
    im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im).astype("float32") / 255.0  # HWC
    chw = np.transpose(arr, (2, 0, 1))             # CHW
    return chw, (size, size)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
