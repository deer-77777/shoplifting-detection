# Architecture

## Goal

Detect shoplifting behaviour in retail-store CCTV footage by:

1. Training an object-detection model on labelled still frames.
2. Serving that model through interfaces a non-ML user can operate (web upload form, automated camera loop).
3. Providing a labelling workflow so the dataset can grow with footage from real deployments.
4. Producing alerts a security operator can act on.

This is a **detection** task (bounding boxes around suspicious individuals), not a per-frame classification or temporal action-recognition task. Suspicious behaviour is captured in the *label* of the person box, not in motion across frames.

---

## High-level system

```
                                 ┌───────────────────────────┐
                                 │  Roboflow YOLO26 export   │
                                 │  (7,576 labelled frames)  │
                                 └─────────────┬─────────────┘
                                               │  split_dataset.py
                                               ▼
   ┌────────────────────┐         ┌───────────────────────────┐
   │  raw_frames/       │         │  Shoplifting-Detection/   │
   │  user-supplied     │         │  train / valid / test     │
   │  unlabelled frames │         │  6,062  /  1,136  /  378  │
   └─────────┬──────────┘         └─────────────┬─────────────┘
             │                                  │  train.py
             │                                  ▼
             │                    ┌───────────────────────────┐
             │                    │  best.pt                  │
             │                    │  (trained, 2 classes)     │
             │                    └────┬────────┬─────────────┘
             │                         │        │
             ▼                         ▼        ▼
  ┌──────────────────────┐    ┌────────────────────────┐    ┌────────────────────┐
  │  models/             │    │ api/main.py  (FastAPI) │    │ detect_live.py     │
  │  yolo26 n/s/m/l/x.pt │    │  POST /predict         │    │ RTSP loop +        │
  │  (COCO-pretrained,   │◄───│  POST /label/*         │    │ tracker + alerts   │
  │   for person detect) │    │  /raw_frames/* static  │    └─────────┬──────────┘
  └──────────────────────┘    └─────────────┬──────────┘              ▼
                                            │                     alerts/*.jpg
                                            │  HTTP/JSON
                                            ▼
                              ┌───────────────────────────┐
                              │  dashboard/  Next.js+MUI  │
                              │  /        (Predict)       │
                              │  /label   (Labelling)     │
                              └───────────────────────────┘
```

Two model pools, two purposes:

- **`models/`** — pre-trained YOLOv26 weights (COCO classes, 80 categories including `person`). Used by the labelling workflow to find people in new frames *before* the human assigns a behaviour class.
- **`runs/.../best.pt`** — your trained 2-class model (`Shoplifting`, `normal`). Used by `/predict` and `detect_live.py`.

---

## Components

### 1. Training pipeline

| File | Role |
|---|---|
| `split_dataset.py` | Splits the Roboflow `train/` folder into `train/`, `valid/`, `test/` (80/15/5, seed=42, in-place move). |
| `download_models.py` | One-time fetch of `yolo26n/s/m/l/x.pt` into `models/`. |
| `Shoplifting-Detection/data.yaml` | Tells Ultralytics where the splits live and what the classes are. |
| `train.py` | Production training run: starts from `models/yolo26n.pt`, 100 epochs, 640 px, batch 16, all data. |
| `train_cpu_quick.py` | 5-epoch smoke run on 10 % of the data — proves the pipeline end-to-end without needing a GPU. |

Output: `runs/shoplifting_yolo26/weights/best.pt` (and `last.pt`).

### 2. Inference + labelling service (`api/`)

A single FastAPI app that hosts both surfaces:

**Inference (`/predict`)**
- Loads `best.pt` once on startup.
- Returns a JSON list of detections plus a base64-encoded PNG with boxes pre-drawn.
- CORS-allows `http://localhost:3000`.

**Labelling (`/label/*`)**
- Lazy-loads any of the five `models/yolo26*.pt` variants on first request.
- Lists folders, lists images, runs person detection, loads/saves YOLO label files, and reorganises folders into Roboflow's `images/`+`labels/` layout.
- Mounts `raw_frames/` as static files so the browser can render images by URL.

### 3. Dashboard (`dashboard/`)

A Next.js 16 App-Router app with:
- **Two pages.** A shared `Nav` component (`Nav.tsx`) with tabs that switch between `/` (Predict) and `/label` (Labelling).
- **MUI v9 throughout**, dark theme, `AppRouterCacheProvider` for SSR style hydration.
- **No external state library** — pure component state.
- Both pages are Client Components (`"use client"`); the server only delivers the bundle.

### 4. Live IP-camera loop (`detect_live.py`)

Independent of the API. Connects directly to an RTSP stream, runs `model.track()` (detection + BoT-SORT identity tracking), and writes annotated snapshots to `alerts/` the first time each track ID is classified as `Shoplifting`. Designed for unattended operation.

---

## Data flow — single-image prediction (`/predict`)

```
Browser            Next.js /              FastAPI               Ultralytics
───────            ─────────              ───────               ───────────
upload .jpg ────►  FormData ──POST /predict──► Pillow.open ──► YOLO(best.pt).predict
                   fetch()                       │                  │
                                            numpy array             ▼
                                                                boxes + plot()
                                                                    ▼
                       render ◄─── JSON response ◄── b64 PNG + detections
                       - <img>
                       - <Table>
                       - alert chip
```

Total round-trip on CPU at 320 px: ~100–200 ms.

---

## Data flow — labelling (`/label/*`)

```
Browser            Next.js /label          FastAPI                       Ultralytics
───────            ──────────────          ───────                       ───────────

select folder ───► GET /label/folders ───► sorted(raw_frames.iterdir())
                   GET /label/images   ──► flat root + images/ + labels/
                   GET /label/models   ──► ls models/*.pt

click image  ────► GET /label/load        OR    POST /label/detect
                   (if labelled, parses          (if not, runs YOLO(models/yolo26*).predict
                    saved YOLO .txt back          with classes=[0])
                    to pixel coords)              │
                                                  ▼
                       render ◄── numbered boxes overlaid on <img>

assign + save ───► POST /label/save  ───► writes <folder>/labels/<stem>.txt

prepare      ────► POST /label/prepare ──► moves labelled jpgs into <folder>/images/
```

---

## Data flow — live RTSP loop (`detect_live.py`)

```
IP camera ── RTSP/TCP ──► OpenCV decoder ──► YOLO(best.pt).track(stream=True)
                                                       │
                                          per frame:   ▼
                                          ┌────────────────────────────────────┐
                                          │ for box in r.boxes:                │
                                          │   if cls=="Shoplifting" and        │
                                          │      track_id not seen before:     │
                                          │     save jpg to alerts/            │
                                          └────────────────────────────────────┘
```

Tracker (BoT-SORT) keeps the same numeric ID across frames, so one suspect produces one alert per visit, not one per frame.

---

## Why these choices

| Decision | Rationale |
|---|---|
| **YOLOv26 nano variant as default** | Smallest weights (5.3 MB), fastest CPU inference, lowest VRAM. The other four sit beside it for one-click swapping. |
| **All five variants on disk, lazy-loaded** | Switching the labelling backbone shouldn't require code or a redeploy. RAM is only consumed when a variant is actually used. |
| **Ultralytics framework** | Handles dataset loading, augmentation, training, validation metrics, and tracker integration out of the box. Minimal code surface. |
| **Single FastAPI process for `/predict` + `/label/*`** | Both surfaces share the same `raw_frames/`-aware filesystem helpers and CORS configuration; running two services would duplicate work without solving anything. |
| **Browser-rendered overlay boxes (not server-drawn) on labelling page** | Hover-to-highlight per-box correspondence is impossible if the boxes are baked into a PNG. Server-drawn images are still used on `/predict` because that page is read-only. |
| **`raw_frames/` is the staging area, not part of the dataset** | A clear human boundary between "data we own and trust" (`Shoplifting-Detection/`) and "data the user is mid-labelling" (`raw_frames/`). The user copies into the trusted folder explicitly when ready. |
| **Move-not-copy in `/label/prepare`** | Avoids leaving stale duplicates around. The user's intuition: "this batch is done, organise it." |
| **Two-process backend (Python) + frontend (JS)** | Model is PyTorch, so inference runs in Python. Browser-side ONNX would mean an extra export step and a bigger first-load. |
| **MUI v9 + Next.js App Router** | Component library means no custom CSS for a basic admin UI. App Router for SSR-ready hydration via `AppRouterCacheProvider`. |
| **Separate `detect_live.py` instead of an API streaming endpoint** | Real-time inference is long-running and side-effecting (writing alerts). Keeping it out of the request/response loop simplifies both. |
| **Stratified split with seed** | Reproducible eval — `mAP` numbers between runs are comparable. |

---

## Non-goals (current iteration)

- **No drawing/dragging of bounding boxes in the labeller.** Either the model finds the person or you skip the image. Acceptable while iterating because adjusting confidence + model size handles most cases.
- **No multi-camera multiplexing in `detect_live.py`** (one process per camera).
- **No persistent alert store** (alerts are JPGs on disk; no database, no UI to review them).
- **No user authentication** on the API or dashboard — assumes deployment behind a trusted network.
- **No model versioning or A/B serving** — `best.pt` is a single file, swap it and restart.
- **No temporal model** (e.g. SlowFast, LSTM over tracks). Each frame is classified in isolation.

These are reasonable extensions, but they are out of scope for the v1 pipeline.
