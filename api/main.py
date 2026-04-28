"""FastAPI inference service for the shoplifting-detection YOLOv26 model.

Loads best.pt once at startup, exposes POST /predict that accepts an image
and returns detections plus an annotated PNG (base64).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "runs" / "shoplifting_yolo26" / "weights" / "best.pt"

app = FastAPI(title="Shoplifting Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found at {MODEL_PATH}. Train first.")

model = YOLO(str(MODEL_PATH))
class_names = model.names
print(f"Loaded model — classes: {class_names}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    img_array = np.array(image)
    results = model.predict(img_array, conf=conf, verbose=False)
    r = results[0]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
            )

    annotated_bgr = r.plot()
    _, buf = cv2.imencode(".png", annotated_bgr)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image_b64": annotated_b64,
        "image_width": image.width,
        "image_height": image.height,
    }
