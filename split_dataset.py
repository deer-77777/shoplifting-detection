"""Split Roboflow YOLO26 export into train/valid/test folders.

The Roboflow export only contains a `train/` folder, but data.yaml references
`valid/` and `test/`. This script moves a stratified random subset of the
images+labels into `valid/` and `test/` so the YAML resolves correctly.
"""

from pathlib import Path
import random
import shutil

ROOT = Path(__file__).parent / "Shoplifting-Detection"
SRC_IMG = ROOT / "train" / "images"
SRC_LBL = ROOT / "train" / "labels"

VAL_RATIO = 0.15
TEST_RATIO = 0.05
SEED = 42


def main() -> None:
    images = sorted(p for p in SRC_IMG.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No images found in {SRC_IMG}")

    random.Random(SEED).shuffle(images)
    n = len(images)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "test": images[:n_test],
        "valid": images[n_test : n_test + n_val],
    }

    for split, files in splits.items():
        (ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (ROOT / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in files:
            lbl = SRC_LBL / (img.stem + ".txt")
            shutil.move(str(img), ROOT / split / "images" / img.name)
            if lbl.exists():
                shutil.move(str(lbl), ROOT / split / "labels" / lbl.name)
        print(f"{split}: {len(files)} samples")

    remaining = sum(1 for _ in SRC_IMG.iterdir())
    print(f"train: {remaining} samples")


if __name__ == "__main__":
    main()
