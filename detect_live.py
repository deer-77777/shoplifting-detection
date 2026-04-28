"""Real-time shoplifting detection on an RTSP IP camera stream.

Loads the trained YOLOv26 weights, opens an RTSP stream, runs detection
with persistent track IDs, and saves a snapshot the first time each
person is flagged as 'Shoplifting'.

Usage:
    python3 detect_live.py
    RTSP_URL=rtsp://user:pass@host:554/stream python3 detect_live.py

Notes:
- Use stream=True so frames are processed as a generator (no memory blowup).
- track(persist=True) keeps the same ID across frames -> one alert per person.
- vid_stride skips frames if the model can't keep up with the camera FPS.
"""

import os
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "runs" / "shoplifting_yolo26" / "weights" / "best.pt"

RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://user:password@192.168.1.100:554/stream1",
)

SHOPLIFTING_CLASS_ID = 0
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
VID_STRIDE = 1
ALERT_DIR = ROOT / "alerts"

# RTSP over TCP is more reliable than UDP for most IP cameras.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Trained weights not found at {MODEL_PATH}. Run train.py first.")

    ALERT_DIR.mkdir(exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_PATH} on device {device}")
    model = YOLO(str(MODEL_PATH))

    print(f"Connecting to {RTSP_URL}")
    results = model.track(
        source=RTSP_URL,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        vid_stride=VID_STRIDE,
        device=device,
        show=True,
        verbose=False,
    )

    alerted_ids: set[int] = set()

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        cls = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for c, track_id, conf in zip(cls, ids, confs):
            if c != SHOPLIFTING_CLASS_ID or track_id in alerted_ids:
                continue
            alerted_ids.add(int(track_id))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = ALERT_DIR / f"alert_{ts}_id{track_id}_conf{conf:.2f}.jpg"
            cv2.imwrite(str(snapshot), r.plot())
            print(f"[ALERT] Shoplifting suspected | track_id={track_id} conf={conf:.2f} -> {snapshot}")


if __name__ == "__main__":
    main()
