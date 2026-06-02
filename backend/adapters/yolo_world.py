import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Set, Tuple
import numpy as np
from PIL import Image
from model_adapter import DetectedBox, ModelAdapter

# ── Target-class recall + display heuristic (v1.2.0) ──────────────────────────
# Immunity gate: target ("positive") classes are detected down to a fixed 2% raw
# floor regardless of the UI threshold, so faint zero-shot batteries on dark
# backgrounds (raw 2-6%) survive instead of being eliminated at the gate.
RECALL_FLOOR = 0.02   # fixed raw-confidence floor for positives (immunity gate)

# Display overhaul: any positive that survives the recall floor AND the decoy
# cross-suppression has its score OVERWRITTEN into a high bracket so it reads as a
# confident hit on the phone. NOTE: this discards the model's real confidence —
# every surviving positive looks ~88-99% regardless of true certainty. It is a
# deliberate display choice, not statistical calibration.
BOOST_BASE = 0.85
BOOST_SLOPE = 1.5
MAX_DISPLAY = 0.99


def _boost(raw: float) -> float:
    """Map a raw positive score into the confident display bracket (0.85-0.99)."""
    return min(MAX_DISPLAY, BOOST_BASE + raw * BOOST_SLOPE)

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
            "Halfords battery",
            "ASDA battery",
            "Halfords AA battery",
            "ASDA AA battery",
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
        size = os.getenv("MODEL_SIZE", "l").lower()
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
        # Immunity gate: run the model at the fixed 2% recall floor (NOT the UI
        # threshold). Faint target detections survive here and are boosted later;
        # the UI threshold no longer gates them out before scoring.
        print(f"[yolo_world] positive={[c for i,c in enumerate(classes) if i not in negative_indices]}")
        print(f"[yolo_world] negative={[c for i,c in enumerate(classes) if i in negative_indices]}")
        print(f"[yolo_world] ui_threshold={threshold}  recall_floor={RECALL_FLOOR}")

        self._model.set_classes(classes)

        results = self._model.predict(
            img_array,
            conf=RECALL_FLOOR,
            iou=0.5,
            verbose=False,
        )

        # Separate positive and negative detections immediately — never mix them
        # in NMS. The previous bug: USB stick at 0.14 would win NMS over battery
        # at 0.13 for the same box, then get dropped as DECOY, silently erasing
        # the battery detection.
        pos_boxes: List[DetectedBox] = []
        neg_boxes: List[DetectedBox] = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_idx = int(box.cls[0])
                is_neg = cls_idx in negative_indices
                label = classes[cls_idx]
                tag = "DECOY" if is_neg else "HIT"
                print(f"  {tag:5s}  {label!r:<28s} {score:.3f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                entry = DetectedBox(
                    id=0,
                    x1=x1 / img_w, y1=y1 / img_h,
                    x2=x2 / img_w, y2=y2 / img_h,
                    score=round(score, 4),
                )
                if is_neg:
                    neg_boxes.append(entry)
                else:
                    pos_boxes.append(entry)

        # Step 1: NMS on positives only — removes duplicate boxes when multiple
        # positive classes (e.g. "battery" + "alkaline battery") fire on the same object.
        deduped_pos = _nms(pos_boxes, iou_threshold=0.4)

        # Step 2: NMS on negatives only — consolidates negative regions.
        deduped_neg = _nms(neg_boxes, iou_threshold=0.4)

        # Step 3: Cross-polarity suppression — a positive is only suppressed if a
        # negative box covers the SAME region (IoU > 0.6) AND beats it by a clear
        # margin (50% higher score). This prevents a marginal DECOY win from erasing
        # a legitimate detection.
        final_pos = _suppress_by_negatives(deduped_pos, deduped_neg,
                                           iou_threshold=0.6, margin=1.5)

        final_pos.sort(key=lambda b: b.score, reverse=True)
        final = final_pos[:max_boxes]

        # Display overhaul: overwrite each surviving positive's score into the
        # confident bracket, then apply the UI threshold to the BOOSTED score so the
        # slider still does something (at 30% all survivors show; at 95% only the
        # strongest raw detections do). Boosted scores are ~0.88+, so targets "pop"
        # at the default 30% slider as intended.
        boosted: List[DetectedBox] = []
        for b in final:
            disp = round(_boost(b.score), 4)
            if disp >= threshold:
                boosted.append(b.model_copy(update={"score": disp}))
        return [b.model_copy(update={"id": i}) for i, b in enumerate(boosted)]

    def verify_crops(
        self,
        crops: List[Image.Image],
        classes: List[str],
        negative_indices: Set[int],
        threshold: float = 0.01,
    ) -> List[float]:
        """
        Batch semantic scoring for hybrid mode.
        Runs YOLO-World on each pre-cropped image at low confidence (0.01) to
        get raw semantic alignment scores. Returns the highest positive-class
        confidence found per crop (0.0 if nothing detected).
        """
        self._model.set_classes(classes)
        scores: List[float] = []
        for crop in crops:
            crop_arr = np.array(crop.convert("RGB"))
            results = self._model.predict(crop_arr, conf=threshold, iou=0.5, verbose=False)
            best = 0.0
            for result in results:
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    if cls_idx not in negative_indices:
                        best = max(best, float(box.conf[0]))
            scores.append(best)
        return scores


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


def _nms(boxes: List[DetectedBox], iou_threshold: float) -> List[DetectedBox]:
    """Standard NMS within a single polarity group (all-positive or all-negative)."""
    if len(boxes) < 2:
        return list(boxes)
    import torch, torchvision.ops
    coords = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32)
    scores = torch.tensor([b.score for b in boxes], dtype=torch.float32)
    keep = torchvision.ops.nms(coords, scores, iou_threshold).tolist()
    return [boxes[i] for i in keep]


def _iou(a: DetectedBox, b: DetectedBox) -> float:
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


def _suppress_by_negatives(
    positives: List[DetectedBox],
    negatives: List[DetectedBox],
    iou_threshold: float,
    margin: float,
) -> List[DetectedBox]:
    """
    Remove a positive only when a negative box overlaps it by >= iou_threshold
    AND the negative's score is >= positive_score * margin.
    The margin requirement prevents a marginally-higher DECOY from erasing a
    legitimate hit just because "USB stick" edges out "battery" by 0.007.
    """
    result = []
    for pos in positives:
        suppressed = any(
            _iou(pos, neg) >= iou_threshold and neg.score >= pos.score * margin
            for neg in negatives
        )
        if not suppressed:
            result.append(pos)
        else:
            print(f"  SUPPRESSED positive {pos.score:.3f} by overlapping decoy")
    return result
