import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Set, Tuple
import numpy as np
from PIL import Image
from model_adapter import DetectedBox, ModelAdapter

# Per-prompt semantic expansion table.
# "positive": semantic variants that describe the same target — all counted.
# "negative": visually similar distractors — detected but suppressed in output.
#
# NMS runs across ALL classes before filtering, so a high-confidence negative
# detection (e.g. USB drive at 25%) will correctly suppress the lower-confidence
# positive detection (battery at 19%) for the same object region.
_SEMANTIC_EXPANSIONS: dict = {
    "battery": {
        "positive": [
            "battery",
            "AA battery",
            "AAA battery",
            "cylindrical battery",
            "alkaline battery",
            "duracell",
        ],
        "negative": [
            # Only suppress shapes that are visually similar cylinders/sticks
            # but are definitively NOT batteries. Broad classes like "pen" or
            # "marker" are intentionally omitted because the orange battery
            # scores higher on those labels than on "battery" itself.
            "USB flash drive",
            "USB drive",
            "USB stick",
        ],
    },
    "book spine": {
        "positive": [
            "book spine", "book cover", "book", "children book",
            "thin board book", "picture book", "paperback book",
        ],
        "negative": [],
    },
    "book": {
        "positive": [
            "book", "book spine", "book cover", "children book",
            "thin board book", "picture book", "paperback book",
        ],
        "negative": [],
    },
    "book cover": {
        "positive": [
            "book cover", "book spine", "book", "children book",
            "picture book", "paperback book",
        ],
        "negative": [],
    },
}

_MODEL_FILES = {
    "s": ("yolov8s-worldv2.pt", "yolo_world_s"),
    "m": ("yolov8m-worldv2.pt", "yolo_world_m"),
    "l": ("yolov8l-worldv2.pt", "yolo_world_l"),
}


class YOLOWorldAdapter(ModelAdapter):

    def __init__(self) -> None:
        from ultralytics import YOLO
        size = os.getenv("MODEL_SIZE", "m").lower()
        if size not in _MODEL_FILES:
            raise ValueError(f"MODEL_SIZE must be one of: {list(_MODEL_FILES.keys())}")
        model_file, self.model_name = _MODEL_FILES[size]
        self._model = YOLO(model_file)
        print(f"[yolo_world] loaded {model_file}  MODEL_SIZE={size}")

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
        max_boxes: int,
    ) -> List[DetectedBox]:
        img_w, img_h = image.size
        img_array = np.array(image)

        classes, negative_indices = _build_classes(prompt)
        print(f"[yolo_world] positive={[c for i,c in enumerate(classes) if i not in negative_indices]}")
        print(f"[yolo_world] negative={[c for i,c in enumerate(classes) if i in negative_indices]}")
        print(f"[yolo_world] conf={threshold}")

        self._model.set_classes(classes)

        results = self._model.predict(
            img_array,
            conf=threshold,
            iou=0.5,
            verbose=False,
        )

        # Collect every raw detection with a flag for whether it's a decoy class.
        raw_boxes: List[DetectedBox] = []
        raw_is_neg: List[bool] = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_idx = int(box.cls[0])
                is_neg = cls_idx in negative_indices
                label = classes[cls_idx]
                tag = "DECOY" if is_neg else "HIT"
                print(f"  {tag:5s}  {label!r:<28s} {score:.3f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                raw_boxes.append(DetectedBox(
                    id=0,
                    x1=x1 / img_w, y1=y1 / img_h,
                    x2=x2 / img_w, y2=y2 / img_h,
                    score=round(score, 4),
                ))
                raw_is_neg.append(is_neg)

        # Cross-class NMS over ALL boxes (positive + negative).
        # A decoy detected at higher confidence than an overlapping positive will
        # win the NMS and both get dropped — correctly suppressing the false positive.
        kept = _cross_class_nms_indices(raw_boxes, iou_threshold=0.4)

        positive = [
            raw_boxes[i] for i in kept if not raw_is_neg[i]
        ]

        positive.sort(key=lambda b: b.score, reverse=True)
        final = positive[:max_boxes]
        return [b.model_copy(update={"id": i}) for i, b in enumerate(final)]


def _build_classes(prompt: str) -> Tuple[List[str], Set[int]]:
    """
    Expand comma-separated user prompt via the semantic expansion table.
    Returns (all_classes_list, set_of_negative_indices).
    """
    tokens = [t.strip().lower() for t in prompt.split(",") if t.strip()]

    positive: List[str] = []
    negative: List[str] = []

    for token in tokens:
        exp = _SEMANTIC_EXPANSIONS.get(token, {})
        if exp:
            positive.extend(exp.get("positive", []))
            negative.extend(exp.get("negative", []))
        else:
            positive.append(token)

    # Deduplicate preserving order; positive always wins over negative
    positive = list(dict.fromkeys(positive))
    negative = [n for n in dict.fromkeys(negative) if n not in positive]

    all_classes = positive + negative
    negative_indices: Set[int] = set(range(len(positive), len(all_classes)))
    return all_classes, negative_indices


def _cross_class_nms_indices(boxes: List[DetectedBox], iou_threshold: float) -> List[int]:
    if len(boxes) < 2:
        return list(range(len(boxes)))

    import torch
    import torchvision.ops

    coords = torch.tensor(
        [[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32
    )
    scores = torch.tensor([b.score for b in boxes], dtype=torch.float32)
    return torchvision.ops.nms(coords, scores, iou_threshold).tolist()
