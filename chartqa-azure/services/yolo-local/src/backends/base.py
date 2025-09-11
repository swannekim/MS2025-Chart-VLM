from abc import ABC, abstractmethod
from typing import Dict, List
from PIL import Image

class ClassifierBackend(ABC):
    def __init__(self, classes: List[str], imgsz: int = 224):
        self.classes = classes
        self.imgsz = imgsz

    @abstractmethod
    def predict(self, img: Image.Image, topk: int = 2) -> Dict:
        """
        Returns:
        {
          "label": "chart",
          "prob": 0.93,
          "probs": {"chart": 0.93, "nonchart": 0.07},
          "topk": [("chart", 0.93), ("nonchart", 0.07)]
        }
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
