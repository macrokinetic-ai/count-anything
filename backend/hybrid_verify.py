"""
Hybrid edge-server verification pipeline.

When the Android client sends on-device geometry proposals (from YOLOv8n on the
Exynos NPU), this module crops each proposal region and asks YOLO-World-L to
score the semantic match. A decision-fusion formula combines the geometric
confidence (from the phone) with the semantic confidence (from the server) into
a single final score. Proposals that fail the semantic minimum threshold are
rejected before they reach the Android UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from PIL import Image

from model_adapter import DetectedBox

if TYPE_CHECKING:
    from adapters.yolo_world import YOLOWorldAdapter

# Fraction of image dimension added as padding on each side of a proposal crop.
# Prevents tight crops from cutting off context the model needs for recognition.
_CROP_PADDING = 0.05

# Fusion weights — semantic score carries more weight because YOLO-World-L is
# a stronger semantic reasoner than the on-device geometry-only YOLOv8n.
_GEO_WEIGHT = 0.30
_SEM_WEIGHT = 0.70

# Minimum semantic score for a proposal to pass. Below this, even a high
# geometric confidence cannot save the box — it is a semantic non-match.
_SEMANTIC_MIN = 0.10

# Maximum IoU between two surviving proposals before one is considered a
# duplicate and removed in the post-fusion NMS step.
_FUSION_NMS_IOU = 0.40

# Crop inference size. Smaller than full-image 640 to keep latency reasonable
# when running N crops sequentially on the server.
_CROP_INFER_SIZE = 320


def crop_and_verify(
    image: Image.Image,
    proposed_boxes: list[dict],
    adapter: "YOLOWorldAdapter",
    prompt: str,
    threshold: float,
    max_boxes: int,
) -> List[DetectedBox]:
    """
    Main entry point for hybrid mode.

    Args:
        image:          Full image (EXIF-corrected, RGB) received from Android.
        proposed_boxes: List of dicts {"x1","y1","x2","y2","score"} in normalized
                        0-1 coords, produced by the on-device edge model.
        adapter:        Loaded YOLOWorldAdapter instance (yolo_world_l).
        prompt:         User text prompt (e.g. "battery").
        threshold:      Confidence threshold (passed in from Android slider).
        max_boxes:      Cap on returned box count.

    Returns:
        List[DetectedBox] with geometric_score, semantic_score, and fused score.
    """
    from adapters.yolo_world import _build_classes, _nms

    img_w, img_h = image.size

    # ── 1. Validate and clamp proposals ──────────────────────────────────────
    valid: list[dict] = []
    for p in proposed_boxes:
        try:
            x1 = max(0.0, min(1.0, float(p["x1"])))
            y1 = max(0.0, min(1.0, float(p["y1"])))
            x2 = max(0.0, min(1.0, float(p["x2"])))
            y2 = max(0.0, min(1.0, float(p["y2"])))
            score = max(0.0, min(1.0, float(p["score"])))
            if x2 - x1 > 0.005 and y2 - y1 > 0.005:   # skip degenerate boxes
                valid.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score})
        except (KeyError, TypeError, ValueError):
            continue  # silently skip malformed entries

    if not valid:
        return []

    # ── 2. Crop with padding ──────────────────────────────────────────────────
    crops: List[Image.Image] = []
    padded_coords: List[tuple] = []   # (px1, py1, px2, py2) in pixels

    for box in valid:
        px1 = max(0, int((box["x1"] - _CROP_PADDING) * img_w))
        py1 = max(0, int((box["y1"] - _CROP_PADDING) * img_h))
        px2 = min(img_w, int((box["x2"] + _CROP_PADDING) * img_w))
        py2 = min(img_h, int((box["y2"] + _CROP_PADDING) * img_h))
        crop = image.crop((px1, py1, px2, py2)).resize(
            (_CROP_INFER_SIZE, _CROP_INFER_SIZE), Image.BILINEAR
        )
        crops.append(crop)
        padded_coords.append((px1, py1, px2, py2))

    # ── 3. Deduplicate nearly-identical proposals before burning server cycles ─
    # If two proposals overlap at IoU > 0.85 keep only the higher-scored one.
    valid, crops, padded_coords = _dedup_proposals(valid, crops, padded_coords, iou=0.85)

    # ── 4. Semantic scoring via YOLO-World ────────────────────────────────────
    classes, negative_indices = _build_classes(prompt)
    semantic_scores = adapter.verify_crops(crops, classes, negative_indices)

    print(f"[hybrid] {len(valid)} proposals → semantic scores: "
          f"{[round(s, 3) for s in semantic_scores]}")

    # ── 5. Decision fusion + semantic filter ──────────────────────────────────
    fused: List[DetectedBox] = []
    for i, (box, sem_score) in enumerate(zip(valid, semantic_scores)):
        if sem_score < _SEMANTIC_MIN:
            print(f"  REJECTED  proposal {i}  geo={box['score']:.3f}  sem={sem_score:.3f}  (below min)")
            continue

        final_score = round(_GEO_WEIGHT * box["score"] + _SEM_WEIGHT * sem_score, 4)
        print(f"  CONFIRMED proposal {i}  geo={box['score']:.3f}  sem={sem_score:.3f}  "
              f"final={final_score:.3f}")
        fused.append(DetectedBox(
            id=i,
            x1=box["x1"],
            y1=box["y1"],
            x2=box["x2"],
            y2=box["y2"],
            score=final_score,
            geometric_score=round(box["score"], 4),
            semantic_score=round(sem_score, 4),
        ))

    # ── 6. Post-fusion NMS ────────────────────────────────────────────────────
    # Removes duplicates that survived dedup (proposals from adjacent anchors).
    deduped = _nms(fused, iou_threshold=_FUSION_NMS_IOU)

    # ── 7. Sort by final score, cap, re-index ─────────────────────────────────
    deduped.sort(key=lambda b: b.score, reverse=True)
    final = deduped[:max_boxes]
    return [b.model_copy(update={"id": i}) for i, b in enumerate(final)]


def _dedup_proposals(
    valid: list[dict],
    crops: list,
    coords: list,
    iou: float,
) -> tuple[list, list, list]:
    """Remove lower-scored proposals that substantially overlap a higher-scored one."""
    keep = [True] * len(valid)
    for i in range(len(valid)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(valid)):
            if not keep[j]:
                continue
            if _box_iou(valid[i], valid[j]) >= iou:
                # Keep the one with higher geometric score
                if valid[i]["score"] >= valid[j]["score"]:
                    keep[j] = False
                else:
                    keep[i] = False

    v = [valid[i] for i in range(len(valid)) if keep[i]]
    c = [crops[i] for i in range(len(crops)) if keep[i]]
    p = [coords[i] for i in range(len(coords)) if keep[i]]
    return v, c, p


def _box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)
