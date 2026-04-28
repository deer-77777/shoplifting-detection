"""Download all YOLOv26 size variants directly into models/.

Each variant is the same architecture with different depth/width multipliers:
    n (nano)    ~5   MB    fastest, lowest accuracy
    s (small)   ~20  MB
    m (medium)  ~42  MB
    l (large)   ~51  MB
    x (extra)   ~113 MB    slowest, highest accuracy

Safe to re-run:
    - Existing valid files are skipped (size sanity check).
    - Partial/corrupted files are detected and re-downloaded.
    - One failed download does NOT abort the others.

Run once before starting the API:
    python3 download_models.py

The API also calls this same set of weights at request time
(see api/main.py — DEFAULT_PERSON_MODEL and PERSON_MODEL_VARIANTS).
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
RELEASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
FONT_URL = "https://ultralytics.com/assets/Arial.ttf"
VARIANTS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]

# A weight file smaller than this is almost certainly a partial / corrupted
# download (e.g. interrupted, or GitHub returned an HTML error page).
MIN_VALID_SIZE_MB = 1.0


def progress_hook(variant: str):
    def hook(blocks_done: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(total_size, blocks_done * block_size)
        pct = downloaded * 100 // total_size
        bar = "█" * (pct // 5) + "─" * (20 - pct // 5)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    [{bar}] {pct:3d}%  {mb:6.1f} / {total_mb:6.1f} MB")
        sys.stdout.flush()

    return hook


def download(variant: str) -> bool:
    target = MODELS_DIR / f"{variant}.pt"

    if target.exists():
        size_mb = target.stat().st_size / (1024 * 1024)
        if size_mb >= MIN_VALID_SIZE_MB:
            print(f"  [skip] {variant}.pt   {size_mb:6.1f} MB  (already present)")
            return True
        print(f"  [redo] {variant}.pt is too small ({size_mb:.2f} MB) — re-downloading")
        target.unlink()

    url = f"{RELEASE_URL}/{variant}.pt"
    print(f"  [get ] {variant}.pt   <- {url}")
    try:
        urllib.request.urlretrieve(url, target, reporthook=progress_hook(variant))
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n  ! download failed: {exc}")
        if target.exists():
            target.unlink()
        return False

    size_mb = target.stat().st_size / (1024 * 1024)
    if size_mb < MIN_VALID_SIZE_MB:
        print(f"  ! downloaded {variant}.pt is suspiciously small ({size_mb:.2f} MB), removing")
        target.unlink()
        return False

    return True


def download_font() -> bool:
    """Fetch Arial.ttf so plot rendering works offline.

    Ultralytics caches it in ~/.config/Ultralytics/. We stash a copy in
    assets/ alongside the project so it can be moved to an offline machine.
    train.py copies it to the cache dir on first run.
    """
    target = ASSETS_DIR / "Arial.ttf"
    if target.exists() and target.stat().st_size > 100_000:
        size_kb = target.stat().st_size / 1024
        print(f"  [skip] Arial.ttf       {size_kb:6.1f} KB  (already present)")
        return True
    print(f"  [get ] Arial.ttf      <- {FONT_URL}")
    try:
        urllib.request.urlretrieve(FONT_URL, target)
    except Exception as exc:
        print(f"  ! download failed: {exc}")
        return False
    size_kb = target.stat().st_size / 1024
    print(f"    -> {target.name} ({size_kb:.1f} KB)")
    return True


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    print(f"Models  -> {MODELS_DIR}")
    print(f"Assets  -> {ASSETS_DIR}\n")

    successes = 0
    for variant in VARIANTS:
        if download(variant):
            successes += 1
    font_ok = download_font()

    print(f"\nDone. {successes}/{len(VARIANTS)} variants OK. Font: {'OK' if font_ok else 'MISSING'}")
    print("\nFiles in models/:")
    for p in sorted(MODELS_DIR.glob("*.pt")):
        print(f"  {p.name:14s}  {p.stat().st_size / (1024 * 1024):6.1f} MB")
    print("\nFiles in assets/:")
    for p in sorted(ASSETS_DIR.iterdir()) if ASSETS_DIR.exists() else []:
        if p.is_file():
            print(f"  {p.name:14s}  {p.stat().st_size / 1024:6.1f} KB")

    return 0 if (successes == len(VARIANTS) and font_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
