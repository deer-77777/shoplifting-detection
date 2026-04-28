"""Train YOLOv26 on the Shoplifting-Detection dataset.

Run after `split_dataset.py` has populated train/valid/test folders.
Install deps first:
    pip install ultralytics

If a CUDA GPU is unavailable the script falls back to CPU automatically,
but training 7.5k images on CPU will be very slow — use a GPU machine
for real runs.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "Shoplifting-Detection" / "data.yaml"

# Pick a variant. Larger = more accurate but more VRAM/time.
#   yolo26n / yolo26s / yolo26m / yolo26l / yolo26x
MODEL_VARIANT = "yolo26n"
MODEL_WEIGHTS = ROOT / "models" / f"{MODEL_VARIANT}.pt"


def main() -> None:
    if not MODEL_WEIGHTS.exists():
        raise SystemExit(f"{MODEL_WEIGHTS} not found. Run download_models.py first.")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}, base weights: {MODEL_WEIGHTS.name}")

    model = YOLO(str(MODEL_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project=str(ROOT / "runs"),
        name="shoplifting_yolo26",
        patience=20,
        save=True,
        plots=True,
    )

    metrics = model.val()
    print("Validation metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
