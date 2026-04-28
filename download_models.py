"""Download all YOLOv26 size variants into models/.

Each variant is the same architecture with different depth/width multipliers:
    n (nano)    ~5  MB   fastest, lowest accuracy
    s (small)   ~22 MB
    m (medium)  ~50 MB
    l (large)   ~85 MB
    x (extra)   ~140 MB  slowest, highest accuracy

Run once before starting the API. Already-downloaded weights are skipped.
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
VARIANTS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    for variant in VARIANTS:
        target = MODELS_DIR / f"{variant}.pt"
        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  [skip] {variant}.pt exists ({size_mb:.1f} MB)")
            continue

        print(f"  Downloading {variant} ...")
        # YOLO() will download to the current working directory if the file
        # isn't found. We then move it into models/.
        YOLO(f"{variant}.pt")

        cwd_copy = ROOT / f"{variant}.pt"
        if cwd_copy.exists():
            shutil.move(str(cwd_copy), str(target))
        else:
            # Some Ultralytics versions store under ~/.cache; try that too.
            cache_copy = Path.home() / ".cache" / "ultralytics" / f"{variant}.pt"
            if cache_copy.exists():
                shutil.copy(str(cache_copy), str(target))
            else:
                print(f"  ! could not locate downloaded {variant}.pt")
                continue

        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"    -> {target.relative_to(ROOT)} ({size_mb:.1f} MB)")

    print("\nDone. Variants in models/:")
    for p in sorted(MODELS_DIR.glob("*.pt")):
        print(f"  {p.name}: {p.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
