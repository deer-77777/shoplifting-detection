# Shoplifting Detection

A computer-vision pipeline that trains a YOLOv26 object-detection model on the
Roboflow `Shoplifting-Detection` dataset and exposes the resulting model
through four deployment surfaces:

1. **Predict dashboard** — Next.js + MUI page for uploading still images and viewing detections.
2. **Labelling dashboard** — Same dashboard, second tab. Helps a human label new store-camera frames using a pre-trained person detector, then writes Roboflow-compatible YOLO label files.
3. **REST API** — FastAPI service that backs both dashboard tabs (`/predict` and `/label/*`).
4. **Live RTSP loop** — Standalone script that connects to an IP camera and triggers alerts.

The model classifies each detected person as either **`Shoplifting`** (suspicious behaviour observed) or **`normal`**.

---

## Quick start

### Prerequisites

| Tool | Version tested | Notes |
|---|---|---|
| Python | 3.12 | Use a venv — system Python is externally-managed on Ubuntu. |
| Node.js | 23.x | Bundled `npm` works. |
| GPU (optional) | CUDA-capable | CPU works for the API, but full training is impractical without one. |

### One-time setup

```bash
# 1. Python venv + ML deps
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics
pip install -r api/requirements.txt

# 2. Download the YOLOv26 size variants (n/s/m/l/x) into models/
python3 download_models.py

# 3. Split dataset (train/valid/test out of the single Roboflow train/ folder)
python3 split_dataset.py

# 4. Frontend deps
cd dashboard && npm install && cd ..
```

### Train a model

```bash
# Quick smoke run (~5 min on CPU, NOT accurate — for pipeline verification only)
python3 train_cpu_quick.py

# Full run (recommend running on a GPU machine)
python3 train.py
```

Both produce `runs/shoplifting_yolo26/weights/best.pt`.

### Run the dashboard

```bash
# Terminal A — backend (port 8000)
source .venv/bin/activate
uvicorn api.main:app --port 8000 --reload

# Terminal B — frontend (port 3000)
cd dashboard && npm run dev
```

Two pages:

- **<http://localhost:3000>** — *Predict.* Upload an image, get boxes + alert chip.
- **<http://localhost:3000/label>** — *Labelling.* Pick a folder under `raw_frames/`, the page runs person detection and lets you assign a class to each detection. Saves YOLO label files compatible with the training set.

### Add new images for labelling

```bash
mkdir raw_frames/store_2026_04_28
cp /path/to/your/camera/frames/*.jpg raw_frames/store_2026_04_28/
# Reload the dashboard — the new folder appears in the Labelling page dropdown.
```

After labelling, click **Prepare for training** in the dashboard to reorganise into Roboflow's `images/` + `labels/` layout, then merge into the training set:

```bash
cp raw_frames/store_2026_04_28/images/* Shoplifting-Detection/train/images/
cp raw_frames/store_2026_04_28/labels/* Shoplifting-Detection/train/labels/
python3 train.py    # retrain with the expanded data
```

### Run the live IP-camera loop

```bash
source .venv/bin/activate
RTSP_URL="rtsp://user:pass@192.168.1.100:554/stream1" python3 detect_live.py
```

Snapshots of suspected shoplifting events are saved to `alerts/`.

---

## Project layout

```
shoplifting-detection/
├── README.md                       <- this file
├── docs/                           <- design + technical documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── TRAINING.md
│   ├── LABELLING.md
│   └── DEPLOYMENT.md
│
├── Shoplifting-Detection/          <- Roboflow YOLO26 export
│   ├── data.yaml
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/      (created by split_dataset.py)
│   └── test/{images,labels}/       (created by split_dataset.py)
│
├── raw_frames/                     <- staging area for new unlabelled frames
│   └── <batch_name>/
│       ├── *.jpg                   (unlabelled, in flat root)
│       ├── images/                 (created by "Prepare for training")
│       │   └── *.jpg               (labelled images moved here)
│       └── labels/                 (created on first save)
│           └── *.txt               (YOLO labels)
│
├── models/                         <- pre-trained yolo26 weights
│   ├── yolo26n.pt   (5 MB)        ← fastest
│   ├── yolo26s.pt   (20 MB)
│   ├── yolo26m.pt   (42 MB)
│   ├── yolo26l.pt   (51 MB)
│   └── yolo26x.pt   (113 MB)      ← most accurate
│
├── split_dataset.py                <- one-time data split (80/15/5)
├── download_models.py              <- one-time fetch of yolo26 n/s/m/l/x weights
├── train.py                        <- full training (100 epochs, 640px) — GPU
├── train_cpu_quick.py              <- 5-epoch smoke run on 10% of data
├── detect_live.py                  <- RTSP IP-camera inference + alerting
│
├── api/                            <- FastAPI inference + labelling service
│   ├── main.py
│   └── requirements.txt
│
├── dashboard/                      <- Next.js 16 + MUI v9 dashboard
│   └── src/app/
│       ├── layout.tsx
│       ├── theme.ts
│       ├── Nav.tsx                 (Predict / Labelling tab bar)
│       ├── page.tsx                (Predict page)
│       └── label/page.tsx          (Labelling page)
│
├── runs/                           <- training outputs (gitignored)
│   └── shoplifting_yolo26/weights/{best,last}.pt
├── alerts/                         <- jpg snapshots from detect_live.py
└── .venv/                          <- Python virtualenv (gitignored)
```

---

## Documentation

| Document | Topic |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, component responsibilities |
| [docs/DATASET.md](docs/DATASET.md) | Dataset source, format, splits, label structure |
| [docs/TRAINING.md](docs/TRAINING.md) | Model variants, hyperparameters, evaluation, GPU/CPU expectations |
| [docs/LABELLING.md](docs/LABELLING.md) | Labelling workflow: model-assisted, prepare-for-training, raw_frames structure |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | REST API spec, dashboard, RTSP, production hardening |

---

## License & attribution

- Dataset: `rakas-workspace-piisr` on Roboflow (Private license — do not redistribute).
- Model architecture: [Ultralytics YOLOv26](https://docs.ultralytics.com/) (AGPL-3.0).
- Application code in this repo: see project licence.
