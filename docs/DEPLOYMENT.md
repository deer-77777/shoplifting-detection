# Deployment

There are three deployment surfaces, all reading the same `runs/shoplifting_yolo26/weights/best.pt`:

1. **REST API** — `api/main.py` (FastAPI / Uvicorn)
2. **Dashboard** — `dashboard/` (Next.js + MUI)
3. **Live RTSP loop** — `detect_live.py`

This document covers each, plus production hardening notes.

---

## REST API

### Run

```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` is for development. Drop it in production and put the service behind a process manager (`systemd`, `supervisor`, or run inside Docker).

### Endpoints

The API serves two surfaces:
- `/predict` — single-image inference using the trained `best.pt`
- `/label/*` — labelling assistance for new store-camera frames (uses pre-trained `yolo26n.pt` person detector)

#### `GET /health`

Cheap liveness probe. Returns the loaded class map.

```json
{ "status": "ok", "classes": { "0": "Shoplifting", "1": "normal" } }
```

#### `POST /predict`

Multipart upload, optional `?conf=` query parameter (default `0.25`).

| Field | Type | Notes |
|---|---|---|
| `file` | `image/*` | JPEG/PNG/WebP supported by Pillow |
| `conf` | `float` (query) | Confidence threshold, `[0.05, 0.95]` is sensible |

Response:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "Shoplifting",
      "confidence": 0.83,
      "bbox_xyxy": [120.5, 80.0, 340.2, 510.8]
    }
  ],
  "count": 1,
  "annotated_image_b64": "iVBORw0KGgoAAAANS...",
  "image_width": 640,
  "image_height": 480
}
```

- `bbox_xyxy` is in **pixel coordinates** of the original image, not normalised.
- `annotated_image_b64` is a base64 PNG with boxes pre-drawn — saves the frontend from having to render them.
- `count` is the length of `detections`. Zero means nothing crossed the threshold.

Curl example:

```bash
curl -F "file=@frame.jpg" "http://localhost:8000/predict?conf=0.4" | jq .count
```

### CORS

Currently allows only `http://localhost:3000`. Update [api/main.py](../api/main.py) `allow_origins=[...]` when deploying the dashboard to a real domain.

### Model loading

The model is loaded **once at module import**. First request is fast; the cost is paid on process start (~1–2 s on CPU). The API will refuse to start if `best.pt` is missing — train first.

---

## Dashboard

### Run

```bash
cd dashboard
npm run dev          # development, hot reload, port 3000
npm run build && npm run start   # production
```

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the dashboard sends `POST /predict`. Set this when API and dashboard are on different hosts. |

### UI

```
┌─ AppBar ─────────────────────────────────────────┐
│  Shoplifting Detection Dashboard                 │
├──────────────────────────────────────────────────┤
│ ┌─ Card 1: Upload ──────────────────────────┐    │
│ │ [ Choose image ]   filename.jpg (38 KB)  │    │
│ │                                          │    │
│ │ Confidence threshold: 0.25               │    │
│ │ ▭━━━━━━━━━━━━━━━━━━━━━━━▭                │    │
│ │                                          │    │
│ │ [ Run detection ]                        │    │
│ └──────────────────────────────────────────┘    │
│                                                  │
│ ┌─ Card 2: Result ──────────────────────────┐    │
│ │ Original              Annotated           │    │
│ │ ┌──────┐              ┌──────┐            │    │
│ │ │      │              │ □ □  │            │    │
│ │ └──────┘              └──────┘            │    │
│ └──────────────────────────────────────────┘    │
│                                                  │
│ ┌─ Card 3: Detections ──────────────────────┐    │
│ │ 2 found  [ALERT: Shoplifting suspected]  │    │
│ │ ┌──┬──────────────┬───────┬──────────┐    │    │
│ │ │# │ class        │ conf  │ bbox     │    │    │
│ │ ├──┼──────────────┼───────┼──────────┤    │    │
│ │ │1 │ Shoplifting  │ 83 %  │ x,y,x,y  │    │    │
│ │ │2 │ normal       │ 64 %  │ x,y,x,y  │    │    │
│ │ └──┴──────────────┴───────┴──────────┘    │    │
│ └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

State is local React state — refreshing the page clears it. There is no persistence layer.

---

## Live IP-camera loop

### Run

```bash
source .venv/bin/activate
RTSP_URL="rtsp://user:password@192.168.1.100:554/stream1" python3 detect_live.py
```

The script opens a window showing the live annotated stream and writes a snapshot to `alerts/` the first time each tracked person is classified as `Shoplifting`.

### Behaviour

| Setting (constant in `detect_live.py`) | Default | Effect |
|---|---|---|
| `CONF_THRESHOLD` | `0.5` | Drops boxes below this score. Raise to reduce false alarms. |
| `IMG_SIZE` | `640` | Inference resolution. 416 if CPU-bound. |
| `VID_STRIDE` | `1` | Process every Nth frame; raise to 2 or 3 if FPS lags. |
| `tracker` | `botsort.yaml` | Identity tracker. Alternatives: `bytetrack.yaml`. |
| `persist=True` | (always on) | Maintains track IDs across frames — one alert per person, not per frame. |

### Camera URLs

Most IP cameras speak RTSP. Check your vendor's docs; common patterns:

| Vendor | URL pattern |
|---|---|
| Hikvision | `rtsp://user:pass@ip:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0` |
| Generic ONVIF | `rtsp://user:pass@ip:554/stream1` |

The script forces `rtsp_transport=tcp` (more reliable than UDP); if your camera only supports UDP, drop the `OPENCV_FFMPEG_CAPTURE_OPTIONS` line.

### Hardware

Real-time inference at 25-30 FPS requires a CUDA GPU for the `n` and `s` model sizes. CPU-only typically delivers 2-5 FPS at 640 px — fine for offline analysis, not for live alerting unless you accept heavy frame skipping.

---

## Production hardening (not in v1)

These are *not* implemented. They are the obvious next steps if this moves beyond a prototype.

| Concern | Recommendation |
|---|---|
| **Reconnection** | Wrap the `model.track()` generator in a try/except + exponential backoff. Cameras drop. |
| **Multi-camera** | One process per camera, started under `systemd` or Docker Compose. Single process serialising N streams will create latency. |
| **Persistent alerts** | Replace the `alerts/*.jpg` write with a database row + S3/object store upload. |
| **Notification fan-out** | Push alerts to Slack / LINE / SMS / email instead of stdout. Webhook from `detect_live.py` is sufficient. |
| **Evidence clips** | Save a 10-second video around each alert, not a single frame. Use a circular buffer of recent frames. |
| **Temporal smoothing** | Require the `Shoplifting` class to fire in ≥ N of the last M frames before alerting. Cuts false positives dramatically at modest recall cost. |
| **Authentication** | The API and dashboard have none. Put both behind a reverse proxy with HTTP basic auth or your SSO of choice. |
| **TLS** | Required if any non-localhost client talks to the API. Terminate TLS at nginx/Caddy. |
| **Privacy / legal** | Snapshot images contain identifiable people. Store only as long as needed, log access, comply with local CCTV regulations (Japan: Act on the Protection of Personal Information; EU: GDPR; etc.). |
| **Model versioning** | Tag and archive each `best.pt` with the dataset version + git SHA. A rollback plan matters more than you'd think. |
| **Monitoring** | Track `/predict` latency and the `Shoplifting` alert rate. A sudden alert spike is more likely a model bug than a crime wave. |
