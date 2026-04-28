"""FastAPI inference + labelling service for the shoplifting-detection project.

Two surfaces:
  - /predict  : runs the trained best.pt model (Shoplifting / normal classes)
  - /label/*  : assists labelling new images using a COCO-pretrained yolo26n
                (finds persons, lets the dashboard assign classes, writes YOLO labels)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
TRAINED_MODEL_PATH = ROOT / "runs" / "shoplifting_yolo26" / "weights" / "best.pt"
MODELS_DIR = ROOT / "models"
DEFAULT_PERSON_MODEL = "yolo26n"
PERSON_MODEL_VARIANTS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
RAW_FRAMES_ROOT = ROOT / "raw_frames"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_CLASSES = ["Shoplifting", "normal"]

app = FastAPI(title="Shoplifting Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

if not TRAINED_MODEL_PATH.exists():
    raise SystemExit(f"Model not found at {TRAINED_MODEL_PATH}. Train first.")
if not (MODELS_DIR / f"{DEFAULT_PERSON_MODEL}.pt").exists():
    raise SystemExit(
        f"Default person detector not found at {MODELS_DIR}/{DEFAULT_PERSON_MODEL}.pt. "
        f"Run download_models.py first."
    )

trained_model = YOLO(str(TRAINED_MODEL_PATH))
trained_class_names = trained_model.names
print(f"Loaded trained model — classes: {trained_class_names}")

# Lazy cache: only load each person-detection variant when it's first requested.
_person_models: dict[str, YOLO] = {}


def get_person_model(variant: str) -> YOLO:
    if variant not in PERSON_MODEL_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unknown model variant '{variant}'")
    if variant not in _person_models:
        path = MODELS_DIR / f"{variant}.pt"
        if not path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Model {variant}.pt not downloaded. Run download_models.py.",
            )
        print(f"Loading {variant} on first use...")
        _person_models[variant] = YOLO(str(path))
    return _person_models[variant]


# Eagerly load only the default — others come on demand.
get_person_model(DEFAULT_PERSON_MODEL)

RAW_FRAMES_ROOT.mkdir(exist_ok=True)
app.mount("/raw_frames", StaticFiles(directory=str(RAW_FRAMES_ROOT)), name="raw_frames")


def safe_within_raw(*parts: str) -> Path:
    """Resolve a path inside RAW_FRAMES_ROOT. Reject anything that escapes."""
    target = (RAW_FRAMES_ROOT.joinpath(*parts)).resolve()
    root_resolved = RAW_FRAMES_ROOT.resolve()
    if not (target == root_resolved or target.is_relative_to(root_resolved)):
        raise HTTPException(status_code=400, detail="Path escapes raw_frames root")
    return target


def find_image_path(folder: str, name: str) -> Path:
    """Locate an image inside a folder, whether it lives at the flat root
    or inside the prepared images/ subfolder."""
    folder_path = safe_within_raw(folder)
    candidates = [folder_path / name, folder_path / "images" / name]
    for c in candidates:
        if c.exists() and c.suffix.lower() in IMAGE_EXTS:
            return c
    raise HTTPException(status_code=404, detail="Image not found")


def label_path_for(folder: str, name: str) -> Path:
    """Path to the YOLO label file in the labels/ subfolder."""
    folder_path = safe_within_raw(folder)
    return folder_path / "labels" / (Path(name).stem + ".txt")


def relative_image_url(folder: str, image_path: Path) -> str:
    folder_path = safe_within_raw(folder)
    rel = image_path.relative_to(folder_path).as_posix()
    return f"/raw_frames/{folder}/{rel}"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": trained_class_names}


# ---------- /predict (existing) ----------


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
    results = trained_model.predict(img_array, conf=conf, verbose=False)
    r = results[0]

    detections = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": trained_class_names[cls_id],
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


# ---------- /label (new) ----------


@app.get("/label/classes")
def label_classes() -> dict:
    return {"classes": TARGET_CLASSES}


@app.get("/label/models")
def list_models() -> dict:
    out = []
    for variant in PERSON_MODEL_VARIANTS:
        path = MODELS_DIR / f"{variant}.pt"
        out.append(
            {
                "name": variant,
                "available": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1) if path.exists() else None,
                "loaded": variant in _person_models,
            }
        )
    return {"models": out, "default": DEFAULT_PERSON_MODEL}


def _count_images(folder_path: Path) -> int:
    flat = sum(1 for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    images_dir = folder_path / "images"
    sub = 0
    if images_dir.is_dir():
        sub = sum(1 for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    return flat + sub


def _count_labels(folder_path: Path) -> int:
    labels_dir = folder_path / "labels"
    if not labels_dir.is_dir():
        return 0
    return sum(1 for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt")


@app.get("/label/folders")
def list_folders() -> dict:
    folders = []
    for p in sorted(RAW_FRAMES_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        folders.append(
            {"name": p.name, "n_images": _count_images(p), "n_labels": _count_labels(p)}
        )
    return {"folders": folders}


@app.get("/label/images")
def list_images(folder: str) -> dict:
    folder_path = safe_within_raw(folder)
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")

    labels_dir = folder_path / "labels"
    images_dir = folder_path / "images"

    sources: list[Path] = []
    sources.extend(p for p in sorted(folder_path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if images_dir.is_dir():
        sources.extend(p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    images = []
    for p in sources:
        label_file = labels_dir / (p.stem + ".txt")
        n_boxes = 0
        if label_file.exists():
            n_boxes = sum(1 for line in label_file.read_text().splitlines() if line.strip())
        images.append(
            {
                "name": p.name,
                "url": relative_image_url(folder, p),
                "labelled": label_file.exists(),
                "n_boxes": n_boxes,
                "in_images_dir": p.parent.name == "images",
            }
        )
    return {"folder": folder, "images": images}


@app.post("/label/detect")
def label_detect(
    folder: str = Form(...),
    name: str = Form(...),
    conf: float = Form(0.4),
    model: str = Form(DEFAULT_PERSON_MODEL),
) -> dict:
    """Run fresh person detection on the image, ignoring any existing label file."""
    img_path = find_image_path(folder, name)
    selected_model = get_person_model(model)

    image = Image.open(img_path).convert("RGB")
    img_array = np.array(image)
    results = selected_model.predict(img_array, conf=conf, classes=[0], verbose=False)
    r = results[0]

    boxes = []
    if r.boxes is not None and len(r.boxes) > 0:
        for i, box in enumerate(r.boxes, start=1):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            boxes.append(
                {
                    "id": i,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_id": None,
                }
            )

    return {
        "image_width": image.width,
        "image_height": image.height,
        "boxes": boxes,
        "source": "detect",
        "model": model,
    }


@app.get("/label/load")
def label_load(folder: str, name: str) -> dict:
    """Load saved YOLO labels back as pixel-coord boxes for editing."""
    img_path = find_image_path(folder, name)

    image = Image.open(img_path)
    width, height = image.width, image.height
    label_path = label_path_for(folder, name)

    boxes = []
    if label_path.exists():
        for i, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            boxes.append(
                {
                    "id": i,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": None,
                    "class_id": cls_id,
                }
            )

    return {"image_width": width, "image_height": height, "boxes": boxes, "source": "load"}


class SaveBox(BaseModel):
    class_id: int
    bbox_xyxy: list[float]


class SaveRequest(BaseModel):
    folder: str
    name: str
    boxes: list[SaveBox]
    image_width: int
    image_height: int


@app.post("/label/save")
def label_save(req: SaveRequest) -> dict:
    find_image_path(req.folder, req.name)

    if req.image_width <= 0 or req.image_height <= 0:
        raise HTTPException(status_code=400, detail="Invalid image dimensions")

    lines = []
    for b in req.boxes:
        if b.class_id < 0 or b.class_id >= len(TARGET_CLASSES):
            raise HTTPException(status_code=400, detail=f"Invalid class_id {b.class_id}")
        x1, y1, x2, y2 = b.bbox_xyxy
        cx = ((x1 + x2) / 2) / req.image_width
        cy = ((y1 + y2) / 2) / req.image_height
        w = (x2 - x1) / req.image_width
        h = (y2 - y1) / req.image_height
        lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path = label_path_for(req.folder, req.name)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return {"saved": str(label_path.relative_to(ROOT)), "n_boxes": len(lines)}


@app.post("/label/prepare")
def label_prepare(folder: str = Form(...)) -> dict:
    """Restructure a labelled folder into Roboflow's images/ + labels/ layout.

    For every label file under <folder>/labels/, find the matching image at
    the flat root and move it into <folder>/images/. Images that don't have a
    label stay in the flat root (they were skipped during labelling).
    """
    folder_path = safe_within_raw(folder)
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")

    labels_dir = folder_path / "labels"
    images_dir = folder_path / "images"

    if not labels_dir.is_dir():
        return {"moved": 0, "already_organised": 0, "missing_image": 0}

    images_dir.mkdir(exist_ok=True)
    moved = 0
    already = 0
    missing = 0

    for label_file in labels_dir.iterdir():
        if not (label_file.is_file() and label_file.suffix.lower() == ".txt"):
            continue
        stem = label_file.stem

        already_in_sub = [
            p for p in images_dir.glob(f"{stem}.*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if already_in_sub:
            already += 1
            continue

        flat = [
            p for p in folder_path.glob(f"{stem}.*")
            if p.is_file() and p.parent == folder_path and p.suffix.lower() in IMAGE_EXTS
        ]
        if not flat:
            missing += 1
            continue

        shutil.move(str(flat[0]), str(images_dir / flat[0].name))
        moved += 1

    return {"moved": moved, "already_organised": already, "missing_image": missing}
