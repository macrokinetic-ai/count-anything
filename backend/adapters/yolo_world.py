import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List
import numpy as np
from PIL import Image
from model_adapter import DetectedBox, ModelAdapter

# Core book vocabulary always used regardless of the user's prompt.
# Covers spines, flat covers, children's picture books, and thick board books.
_BOOK_CLASSES = [
    "book spine",
    "book cover",
    "book",
    "children book",
    "thin board book",
    "picture book",
    "paperback book",
]

_MODEL_FILES = {
    "s": ("yolov8s-worldv2.pt", "yolo_world_s"),
    "m": ("yolov8m-worldv2.pt", "yolo_world_m"),
    "l": ("yolov8l-worldv2.pt", "yolo_world_l"),
}


class YOLOWorldAdapter(ModelAdapter):

    def __init__(self) -> None:
        from ultralytics import YOLO
        size = os.getenv("MODEL_SIZE", "s").lower()
        if size not in _MODEL_FILES:
            raise ValueError(f"MODEL_SIZE must be one of: {list(_MODEL_FILES.keys())}")
        model_file, self.model_name = _MODEL_FILES[size]
        # Downloads weights on first run if not cached
        self._model = YOLO(model_file)
        print(f"[yolo_world] loaded {model_file} (MODEL_SIZE={size})")

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
        max_boxes: int,
    ) -> List[DetectedBox]:
        img_w, img_h = image.size
        img_array = np.array(image)

        # Merge user-defined comma-separated classes with the core vocabulary.
        # dict.fromkeys preserves insertion order and deduplicates.
        user_classes = [p.strip() for p in prompt.split(",") if p.strip()]
        classes = list(dict.fromkeys(user_classes + _BOOK_CLASSES))

        print(f"[yolo_world] classes={classes} conf={threshold}")
        self._model.set_classes(classes)

        results = self._model.predict(
            img_array,
            conf=threshold,
            iou=0.5,    # per-class NMS inside YOLO
            verbose=False,
        )

        raw: List[DetectedBox] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_idx = int(box.cls[0])
                print(f"  hit class={classes[cls_idx]!r} score={score:.3f} "
                      f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                raw.append(DetectedBox(
                    id=0,
                    x1=x1 / img_w,
                    y1=y1 / img_h,
                    x2=x2 / img_w,
                    y2=y2 / img_h,
                    score=round(score, 4),
                ))

        # Cross-class NMS: prevents the same book being counted once per matching class
        deduped = _cross_class_nms(raw, iou_threshold=0.4)

        # Sort by confidence descending, cap, re-index
        deduped.sort(key=lambda b: b.score, reverse=True)
        final = deduped[:max_boxes]
        return [b.model_copy(update={"id": i}) for i, b in enumerate(final)]


def _cross_class_nms(boxes: List[DetectedBox], iou_threshold: float) -> List[DetectedBox]:
    if len(boxes) < 2:
        return boxes

    import torch
    import torchvision.ops

    coords = torch.tensor(
        [[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32
    )
    scores = torch.tensor([b.score for b in boxes], dtype=torch.float32)
    keep = torchvision.ops.nms(coords, scores, iou_threshold)
    return [boxes[i] for i in keep.tolist()]
