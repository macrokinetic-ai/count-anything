import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List
from PIL import Image
from model_adapter import DetectedBox, ModelAdapter


class LocateAnythingAdapter(ModelAdapter):
    """
    LocateAnything-3B adapter (Phase 2 — requires GPU with ~10 GB VRAM).

    Model: nvidia/LocateAnything-3B on HuggingFace.
    Architecture: Moon-ViT encoder + Qwen2.5 decoder + MLP bridge.
    Inference: Parallel Box Decoding (PBD) — predicts full boxes in one step.

    To enable: set env var ADAPTER=locate_anything before starting the server.
    """

    model_name = "locate_anything_3b"

    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError(
                "LocateAnything-3B requires a CUDA GPU. "
                "Use ADAPTER=yolo_world for CPU-only machines."
            )

        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            "nvidia/LocateAnything-3B", trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            "nvidia/LocateAnything-3B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
        max_boxes: int,
    ) -> List[DetectedBox]:
        import torch

        inputs = self._processor(
            images=image,
            text=f"Locate all {prompt}",
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=512)

        # Parse model output into bounding boxes.
        # LocateAnything returns coordinates as normalized values in model output tokens.
        # Exact parsing depends on the model's output format — refer to NVlabs/Eagle repo.
        raw = self._processor.decode(outputs[0], skip_special_tokens=True)
        boxes = self._parse_output(raw, image.size, threshold, max_boxes)
        return boxes

    def _parse_output(
        self,
        raw_output: str,
        image_size: tuple,
        threshold: float,
        max_boxes: int,
    ) -> List[DetectedBox]:
        # Placeholder: implement box parsing from LocateAnything token output.
        # Format TBD based on NVlabs/Eagle inference code.
        raise NotImplementedError(
            "LocateAnything output parsing not yet implemented. "
            "See NVlabs/Eagle repo for token format details."
        )
