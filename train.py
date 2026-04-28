"""Train YOLOv26 on the Shoplifting-Detection dataset.

Run after `split_dataset.py` has populated train/valid/test folders.
Install deps first:
    pip install ultralytics

If a CUDA GPU is unavailable the script falls back to CPU automatically,
but training 7.5k images on CPU will be very slow — use a GPU machine
for real runs.

Offline note:
    Ultralytics' AMP self-check tries to load `yolo26n.pt` by basename and
    will download it from the internet if it isn't found in the current
    working directory. _ensure_offline_assets() below symlinks it from
    models/ so no network call is needed.

    Plot rendering also wants Arial.ttf (~/.config/Ultralytics/Arial.ttf).
    Pre-stage it via assets/Arial.ttf — see docs/TRAINING.md for the
    offline checklist.
"""

import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "Shoplifting-Detection" / "data.yaml"

# Pick a variant. Larger = more accurate but more VRAM/time.
#   yolo26n / yolo26s / yolo26m / yolo26l / yolo26x
MODEL_VARIANT = "yolo26n"
MODEL_WEIGHTS = ROOT / "models" / f"{MODEL_VARIANT}.pt"

ASSETS_DIR = ROOT / "assets"
ULTRALYTICS_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Ultralytics"


def _ensure_offline_assets() -> None:
    """Make sure Ultralytics never has to reach the internet:
    - `yolo26n.pt` at CWD (for the AMP self-check)
    - `Arial.ttf` in the Ultralytics user config dir (for plot labels)
    """
    amp_link = ROOT / "yolo26n.pt"
    amp_source = ROOT / "models" / "yolo26n.pt"
    if not amp_link.exists() and amp_source.exists():
        try:
            amp_link.symlink_to(amp_source)
        except OSError:
            shutil.copy2(amp_source, amp_link)
        print(f"  staged AMP weight: {amp_link.name} -> {amp_source.relative_to(ROOT)}")

    bundled_font = ASSETS_DIR / "Arial.ttf"
    cached_font = ULTRALYTICS_CONFIG_DIR / "Arial.ttf"
    if bundled_font.exists() and not cached_font.exists():
        ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled_font, cached_font)
        print(f"  staged font: {cached_font}")


def main() -> None:
    if not MODEL_WEIGHTS.exists():
        raise SystemExit(f"{MODEL_WEIGHTS} not found. Run download_models.py first.")

    _ensure_offline_assets()

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
