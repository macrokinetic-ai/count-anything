from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel


class DetectedBox(BaseModel):
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    geometric_score: Optional[float] = None  # hybrid mode only
    semantic_score: Optional[float] = None   # hybrid mode only


class ModelAdapter(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
        max_boxes: int,
    ) -> List[DetectedBox]:
        pass
