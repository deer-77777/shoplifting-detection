# Architecture

## Goal

Detect shoplifting behaviour in retail-store CCTV footage by:

1. Training an object-detection model on labelled still frames.
2. Serving that model through interfaces a non-ML user can operate (web upload form, automated camera loop).
3. Producing alerts a security operator can act on.

This is a **detection** task (bounding boxes around suspicious individuals), not a per-frame classification or temporal action-recognition task. Suspicious behaviour is captured in the *label* of the person box, not in motion across frames.

---

## High-level system

```
                  ┌─────────────────────────────┐
                  │  Roboflow YOLO26 export     │
                  │  (7,576 labelled frames)    │
                  └──────────────┬──────────────┘
                                 │  split_dataset.py (one-time)
                                 ▼
                  ┌─────────────────────────────┐
                  │  train / valid / test       │
                  │  6,062  /  1,136  /  378    │
                  └──────────────┬──────────────┘
                                 │  train.py  (Ultralytics YOLO)
                                 ▼
                  ┌─────────────────────────────┐
                  │  best.pt  (trained weights) │
                  └────┬───────────┬────────────┘
                       │           │
       ┌───────────────┘           └────────────────┐
       ▼                                            ▼
┌────────────────┐                          ┌──────────────────┐
│ api/main.py    │   <─── HTTP/JSON ───>    │ dashboard/       │
│ FastAPI        │                          │ Next.js + MUI    │
│ POST /predict  │                          │ (browser)        │
└────────────────┘                          └──────────────────┘

       └─────────── separate process ───────────┐
                                                ▼
                                       ┌──────────────────┐
                                       │ detect_live.py   │
                                       │ RTSP loop +      │
                                       │ tracker + alerts │
                                       └──────────┬───────┘
                                                  ▼
                                            alerts/ (jpg files)
```

---

## Components

### 1. Training pipeline

| File | Role |
|---|---|
| `split_dataset.py` | Splits the Roboflow `train/` folder into `train/`, `valid/`, `test/` (80/15/5, seed=42, in-place move). |
| `Shoplifting-Detection/data.yaml` | Tells Ultralytics where the splits live and what the classes are. |
| `train.py` | Production training run: `yolo26n.pt` → 100 epochs, 640 px, batch 16, all data. |
| `train_cpu_quick.py` | 5-epoch smoke run on 10 % of the data — proves the pipeline end-to-end without needing a GPU. |

Output: `runs/shoplifting_yolo26/weights/best.pt` (and `last.pt`).

### 2. Inference service (`api/`)

A FastAPI app that:
- Loads `best.pt` once on startup (slow path, ~1-2 s).
- Exposes `POST /predict` for image uploads.
- Returns both a JSON list of detections and a base64-encoded PNG with boxes drawn on top (saves the frontend from having to render the boxes itself).
- CORS-allows `http://localhost:3000` so the dashboard can call it directly during development.

### 3. Dashboard (`dashboard/`)

A Next.js 16 App-Router app with:
- A single page (`page.tsx`) — upload control, confidence-threshold slider, side-by-side original/annotated images, a detection table, and a red ALERT chip when any `Shoplifting` class fires.
- MUI v9 components, dark theme, no external state library — purely component state.
- All work happens client-side; the page itself is a Client Component (`"use client"`), the server only delivers the bundle.

### 4. Live IP-camera loop (`detect_live.py`)

Independent of the API. Connects directly to an RTSP stream, runs `model.track()` (detection + BoT-SORT identity tracking), and writes annotated snapshots to `alerts/` the first time each track ID is classified as `Shoplifting`. Designed for unattended operation.

---

## Data flow — single image prediction

```
Browser                Next.js page                FastAPI                  Ultralytics
───────                ────────────                ───────                  ───────────
upload .jpg ─────────► FormData ─── POST /predict ────► Pillow.open ─────►  YOLO.predict
                       fetch()                           ▼                    │
                                                   numpy array                ▼
                                                                       boxes + plot()
                                                                              ▼
                          render <───── JSON response ◄──── b64 PNG + detections JSON
                          - <img>
                          - <Table>
                          - alert chip
```

Total round-trip on CPU at 320 px: **~100-200 ms per image** for a model of this size.

---

## Data flow — live RTSP loop

```
IP camera ── RTSP/TCP ──► OpenCV decoder ──► YOLO.track(stream=True) ──► Result generator
                                                                              │
                                                  per frame:                  ▼
                                                  ┌─────────────────────────────────┐
                                                  │ for box in r.boxes:             │
                                                  │   if cls=="Shoplifting" and     │
                                                  │      track_id not seen before:  │
                                                  │     save jpg to alerts/         │
                                                  └─────────────────────────────────┘
```

The tracker keeps the same numeric ID across frames so a single suspect produces one alert per visit, not one per frame.

---

## Why these choices

| Decision | Rationale |
|---|---|
| **YOLOv26 nano variant first** | Smallest weights (5.3 MB), fastest CPU inference, lowest VRAM. Easy to upgrade to `s/m/l/x` later — same training config. |
| **Ultralytics framework** | Handles dataset loading, augmentation, training loop, validation metrics, and tracker integration out of the box. Minimal code surface. |
| **Two-process backend (Python) + frontend (JS)** | Model is PyTorch, so inference must run in Python. Browser-side ONNX inference would mean an extra export step and a bigger first-load. |
| **FastAPI over Flask/Django** | Async, type-validated, automatic OpenAPI docs at `/docs`. Tiny boilerplate for a JSON+file-upload endpoint. |
| **MUI v9 + Next.js App Router** | Component library means no custom CSS for a basic admin UI. App Router for SSR-ready hydration via `AppRouterCacheProvider`. |
| **Separate `detect_live.py` instead of an API streaming endpoint** | Real-time inference is long-running and side-effecting (writing alerts). Keeping it out of the request/response loop simplifies both. |
| **Stratified split with seed** | Reproducible eval — `mAP` numbers between runs are comparable. |

---

## Non-goals (current iteration)

- No multi-camera multiplexing in `detect_live.py` (one process per camera).
- No persistent alert store (alerts are JPGs on disk; no database, no UI to review them).
- No user authentication on the API or dashboard — assumes deployment behind a trusted network.
- No model versioning or A/B serving — `best.pt` is a single file, swap it and restart.
- No temporal model (e.g. SlowFast, LSTM over tracks). Each frame is classified in isolation.

These are reasonable extensions, but they are out of scope for the v1 pipeline.
