"""Quick CPU training run — produces a working best.pt for pipeline testing.

This is NOT a model you would deploy — accuracy will be poor. The goal is to
finish in ~30 minutes on CPU so you can verify detect_live.py end-to-end
against your IP camera.

For a real model, run train.py on a GPU machine.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "Shoplifting-Detection" / "data.yaml"


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Quick training on device: {device}")

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=5,
        imgsz=320,
        batch=8,
        device=device,
        project=str(ROOT / "runs"),
        name="shoplifting_yolo26",
        fraction=0.1,
        workers=2,
        plots=False,
        exist_ok=True,
    )

    print("Done. Weights at: runs/shoplifting_yolo26/weights/best.pt")


if __name__ == "__main__":
    main()
