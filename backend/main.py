import io
import json
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ExifTags
from pydantic import BaseModel

from model_adapter import DetectedBox

DEBUG_SAVE = os.getenv("DEBUG_SAVE", "0") == "1"
_DEBUG_PATH = os.path.join(os.path.dirname(__file__), "debug_input.jpg")

VERSION_ID = "v1.2.0_ENTERPRISE_CALIBRATED_FINAL"


class DetectResponse(BaseModel):
    count: int
    image_width: int
    image_height: int
    model: str
    version_id: str
    inference_ms: int
    mode: str  # "hybrid" | "server_only"
    boxes: List[DetectedBox]


@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter_name = os.getenv("ADAPTER", "yolo_world")

    if adapter_name == "yolo_world":
        from adapters.yolo_world import YOLOWorldAdapter
        app.state.adapter = YOLOWorldAdapter()
    elif adapter_name == "locate_anything":
        from adapters.locate_anything import LocateAnythingAdapter
        app.state.adapter = LocateAnythingAdapter()
    else:
        raise ValueError(f"Unknown ADAPTER: {adapter_name!r}. Use 'yolo_world' or 'locate_anything'.")

    print(f"[startup] adapter={app.state.adapter.model_name}  DEBUG_SAVE={DEBUG_SAVE}  version={VERSION_ID}")
    yield


app = FastAPI(title="Count Anything Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "adapter": app.state.adapter.model_name, "version": VERSION_ID}


@app.post("/detect", response_model=DetectResponse)
async def detect(
    image: UploadFile,
    prompt: str = Form(default="book spine"),
    threshold: float = Form(default=0.30),
    max_boxes: int = Form(default=60),
    proposed_boxes: Optional[str] = Form(default=None),
):
    start_ms = time.monotonic() * 1000

    raw = await image.read()
    img = Image.open(io.BytesIO(raw))

    exif_orientation = _read_exif_orientation(img)
    original_size = img.size

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img_w, img_h = img.size

    print(
        f"[detect] received {original_size} exif_orientation={exif_orientation} "
        f"→ after_transpose={img.size}  prompt={prompt!r}  threshold={threshold}  "
        f"mode={'hybrid' if proposed_boxes else 'server_only'}"
    )

    if DEBUG_SAVE:
        img.save(_DEBUG_PATH, "JPEG", quality=95)
        print(f"[debug] image saved → {_DEBUG_PATH}")

    boxes: List[DetectedBox]
    mode: str

    if proposed_boxes:
        try:
            proposals = json.loads(proposed_boxes)
            from hybrid_verify import crop_and_verify
            boxes = crop_and_verify(img, proposals, app.state.adapter, prompt, threshold, max_boxes)
            mode = "hybrid"
            print(f"[detect] hybrid: {len(proposals)} proposals → {len(boxes)} confirmed")
        except Exception as e:
            print(f"[detect] hybrid path failed ({e}), falling back to server_only")
            boxes = app.state.adapter.detect(img, prompt, threshold, max_boxes)
            mode = "server_only"
    else:
        boxes = app.state.adapter.detect(img, prompt, threshold, max_boxes)
        mode = "server_only"

    inference_ms = int(time.monotonic() * 1000 - start_ms)
    print(f"[detect] found {len(boxes)} boxes in {inference_ms} ms  [{mode}]")

    return DetectResponse(
        count=len(boxes),
        image_width=img_w,
        image_height=img_h,
        model=app.state.adapter.model_name,
        version_id=VERSION_ID,
        inference_ms=inference_ms,
        mode=mode,
        boxes=boxes,
    )


def _read_exif_orientation(img: Image.Image) -> int | None:
    try:
        exif = img._getexif()
        if not exif:
            return None
        orient_tag = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        return exif.get(orient_tag)
    except Exception:
        return None
