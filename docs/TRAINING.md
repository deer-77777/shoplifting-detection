# Training

## Model

[YOLOv26](https://docs.ultralytics.com/) — Ultralytics' detection family. The architecture is the same for all size variants; only the depth/width multipliers change.

| Variant | Speed | Accuracy | VRAM (batch=16, 640 px, train) | Recommended use |
|---|---|---|---|---|
| `yolo26n` | fastest | lowest | ~2 GB | Edge devices, smoke tests |
| `yolo26s` | fast | low-mid | ~4 GB | Real-time on a modest GPU |
| `yolo26m` | mid | mid | ~8 GB | Balanced default |
| `yolo26l` | slow | high | ~14 GB | Server-side, accuracy-first |
| `yolo26x` | slowest | highest | ~22 GB | Offline batch / max mAP |

All five variants ship pre-downloaded in `models/` after running `download_models.py`. The training scripts default to **`models/yolo26n.pt`**. Bumping to `s` or `m` is a one-line change (`MODEL_VARIANT = "yolo26m"` in `train.py`) and uses identical training arguments.

### YOLO pipelines (task heads)

Same backbone, different output head:

| Suffix | Task | Output | Used here? |
|---|---|---|---|
| `yolo26n.pt` | Detect | bounding boxes + class | **yes** |
| `yolo26n-seg.pt` | Segment | pixel masks | no |
| `yolo26n-cls.pt` | Classify | whole-image label | no |
| `yolo26n-pose.pt` | Pose | keypoints (skeleton) | no — but a sensible v2 (e.g. detect concealment gestures) |
| `yolo26n-obb.pt` | Oriented bbox | rotated rectangles | no |

`model.track(...)` adds tracking on top of any detection-style model — see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Scripts

| Script | Purpose | Time on CPU | Time on RTX 4060 (estimate) |
|---|---|---|---|
| `download_models.py` | Downloads all five YOLOv26 variants into `models/` | ~5 min (network bound) | same |
| `train_cpu_quick.py` | Smoke test — 5 epochs, 320 px, 10 % data | ~5 min | ~30 s |
| `train.py` | Real training — 100 epochs, 640 px, all data | weeks (impractical) | 2-4 h |

### `download_models.py` — pre-flight

Run this before either training script:

```bash
python3 download_models.py
```

It writes weights to `models/` using a direct `urllib` download (~230 MB total) and:
- **Skips files that already exist** with a valid size, so re-running is cheap.
- **Detects partial / corrupted downloads** (anything < 1 MB is treated as bad) and re-fetches them.
- **Continues past individual failures** — one variant failing (e.g. transient 503) doesn't abort the others.
- **Exits non-zero** if any variant is still missing at the end, so you can wrap it in CI.

Both training scripts also depend on `models/` being populated — they exit with a clear error if `models/yolo26n.pt` (or whatever `MODEL_VARIANT` you set) is missing.

### Training scripts

The training scripts:
- Auto-detect GPU vs CPU (`device=0` if CUDA available, else `cpu`).
- Load base weights from `models/<variant>.pt` (no re-download).
- Save outputs to `runs/shoplifting_yolo26/weights/{best,last}.pt`.
- Use `exist_ok=True` so re-running overwrites the previous run.

`best.pt` is the checkpoint with the highest validation mAP50-95 across all epochs; `last.pt` is the final epoch's weights.

---

## Hyperparameters

The values in `train.py` (and rationale):

| Argument | Value | Why |
|---|---|---|
| `epochs` | 100 | Enough for convergence on a small 2-class dataset; early-stopping cuts it short if validation plateaus. |
| `imgsz` | 640 | Standard YOLO default. Lower (e.g. 416) speeds training but loses small-object recall. |
| `batch` | 16 | Reasonable for an 8-12 GB consumer GPU. Reduce to 8 or 4 if you OOM. |
| `patience` | 20 | Stop training if validation mAP doesn't improve for 20 consecutive epochs. |
| `optimizer` | `auto` (Ultralytics default) | Picks SGD or AdamW based on dataset size. |
| `plots` | `True` | Writes `results.png`, `confusion_matrix.png`, etc. into the run folder. |

Anything not listed uses Ultralytics' default — these are well-tuned and rarely worth changing for a starter project.

---

## What you get when training finishes

```
runs/shoplifting_yolo26/
├── args.yaml                     <- exact hyperparameters used
├── results.csv                   <- per-epoch loss + metrics
├── results.png                   <- chart of the above
├── confusion_matrix.png          <- per-class TP/FP/FN
├── F1_curve.png  P_curve.png  R_curve.png  PR_curve.png
├── train_batch*.jpg              <- random training batches with labels overlaid (sanity check)
├── val_batch*_labels.jpg         <- validation set with ground truth
├── val_batch*_pred.jpg           <- validation set with predictions
└── weights/
    ├── best.pt                   <- ← used by api/, dashboard, detect_live.py
    └── last.pt
```

Open `results.png` first — if the loss curves are still trending down at the final epoch, train longer. If validation mAP has plateaued, you're done.

---

## Reading the metrics

Ultralytics prints a summary like this every epoch:

```
                Class     Images  Instances        P          R      mAP50  mAP50-95
                  all       1136       1387      0.20       0.21      0.12      0.05
```

| Metric | What it means | Good value (after full training) |
|---|---|---|
| `P` (precision) | Of the boxes the model predicted, what fraction were correct? | > 0.7 |
| `R` (recall) | Of the real boxes in the ground truth, what fraction did the model find? | > 0.6 |
| `mAP50` | Mean Average Precision at IoU ≥ 0.5 — the "did it roughly find the object" score. | > 0.5 — primary headline metric |
| `mAP50-95` | mAP averaged over IoU thresholds 0.5..0.95 step 0.05 — strict box-quality metric. | > 0.3 |

The numbers from `train_cpu_quick.py` (mAP50 ≈ 0.12) are **not** representative — they reflect 5 epochs on a tiny subset. A proper run on a GPU should comfortably exceed mAP50 = 0.5 on this dataset, often higher.

---

## Confidence threshold

The model assigns each box a confidence score in `[0, 1]`. The **confidence threshold** is the cutoff below which detections are discarded.

| Threshold | Effect | Use case |
|---|---|---|
| Low (0.10) | Many detections, high recall, more false positives | Manual review by a guard |
| Default (0.25) | Ultralytics' starting point — usually balanced | Initial experimentation |
| High (0.70) | Few but confident detections | Automated alerts / locking actions |

The threshold is a **runtime** parameter, not a training one. The dashboard exposes a slider for it; the API takes it as a `?conf=` query parameter; `detect_live.py` reads it from the `CONF_THRESHOLD` constant.

---

## When the trained model is not good enough

In rough order of impact:

1. **Train longer / on more data.** Re-run `train.py` (full data, GPU). Most accuracy issues come from undertrained models.
2. **Move up a size.** `yolo26s` → `yolo26m`. Same script, change one string.
3. **Increase image size** to 960 or 1280 if your camera resolution is high. Catches smaller behavioural cues but is slower.
4. **Add augmentation.** Ultralytics applies HSV jitter, mosaic, and flip by default. For a CCTV-only deployment, consider disabling horizontal flip (`fliplr=0.0`) — store layouts have a left/right structure that flipping destroys.
5. **Domain-fine-tune.** Train on the public dataset, then continue training on a small set of frames captured from your own cameras. Often yields a bigger jump than any of the above.
