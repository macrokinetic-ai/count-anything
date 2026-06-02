from typing import List
from model_adapter import DetectedBox


def apply_nms(boxes: List[DetectedBox], iou_threshold: float = 0.4) -> List[DetectedBox]:
    """Suppress overlapping boxes using non-maximum suppression."""
    if len(boxes) < 2:
        return boxes

    import torch
    import torchvision.ops

    coords = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32)
    scores = torch.tensor([b.score for b in boxes], dtype=torch.float32)

    keep = torchvision.ops.nms(coords, scores, iou_threshold)
    kept = [boxes[i] for i in keep.tolist()]

    # Re-index ids to be contiguous
    for i, box in enumerate(kept):
        box = box.model_copy(update={"id": i})
        kept[i] = box

    return kept
