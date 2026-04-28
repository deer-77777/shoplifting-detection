# Labelling

The labelling dashboard at <http://localhost:3000/label> lets a human label new
store-camera frames quickly by leaning on a pre-trained person detector to do
80 % of the work.

This document explains the workflow, the on-disk layout, and the model-selection
trade-off.

---

## Why this exists

The Roboflow dataset comes pre-labelled. But once you deploy the model in your
own store, the only way to keep improving accuracy is to label new frames from
*your* cameras. Labelling from scratch — drawing every box by hand in a tool
like CVAT or Roboflow — is slow.

This dashboard short-circuits the slow part:

```
        new image  ──►  pre-trained YOLOv26 finds persons  ──►  human picks class
                          (handles 90 % of box drawing)              (the only judgement
                                                                      software can't make)
```

You only spend time on what software can't do: deciding whether each person is
**Shoplifting** or **normal**.

---

## The page

```
┌──────────────────────────────────────────────────────────────────────────┐
│  AppBar:  [ Predict ] [ Labelling ]                                      │
├──────────────┬──────────────────────────────────┬────────────────────────┤
│  LEFT        │   CENTRE                         │  RIGHT                 │
│  Folder ▾    │   filename.jpg  Model ▾  Conf=○  │  Detections            │
│  [Reload]    │                                  │                        │
│  Progress    │   ┌───────────────────────────┐  │  #1 [class ▾] [✕]      │
│  [Prepare]   │   │ ┌──[1]──┐                 │  │  #2 [class ▾] [✕]      │
│              │   │ │       │                 │  │  #3 [class ▾] [✕]      │
│  Image list  │   │ └───────┘     ┌──[2]──┐   │  │                        │
│  ☐ frame1    │   │               │       │   │  │  [ Save labels ]       │
│  ☑ frame2    │   │   ┌──[3]──┐   └───────┘   │  │                        │
│  ☐ frame3    │   │   │       │               │  │                        │
│   ...        │   │   └───────┘               │  │                        │
│              │   └───────────────────────────┘  │                        │
└──────────────┴──────────────────────────────────┴────────────────────────┘
```

### Left panel — folder + image list

| Element | Behaviour |
|---|---|
| **Folder dropdown** | Lists every direct subfolder of `raw_frames/`. Counter shows `(labels / images)`. |
| **Reload** | Re-fetches both the folder list and (if a folder is selected) the image list. |
| **Progress bar** | Fraction of images in the current folder that have a saved label. |
| **Prepare for training** | Reorganises labelled content into Roboflow's `images/`+`labels/` structure. Disabled until at least one image is labelled. |
| **Image rows** | Click to open. Green = labelled with N saved boxes. Default = unlabelled. |

### Centre panel — image preview + boxes

| Element | Behaviour |
|---|---|
| **Filename** | Selected image. |
| **Model dropdown** | Choose which pre-trained yolo26 variant runs the person detection. See [Model selection](#model-selection). |
| **Confidence slider** | Threshold for what counts as a "person". Lower = more boxes (incl. false positives), higher = fewer but more confident. Doesn't auto-rerun — click **Re-detect** to apply. |
| **Re-detect** | Throws away current boxes and re-runs detection with the current model + threshold. |
| **Image** | Live preview with numbered overlay boxes, colour-coded so you can match each box to its row in the right panel. Hover a box → highlight the corresponding row, and vice versa. |

### Right panel — detections

| Element | Behaviour |
|---|---|
| **#N chip** | Box number, colour-matched to the on-image overlay. |
| **Confidence %** | Detector's score for that box (only on freshly-detected images, not on reloaded saved boxes). |
| **Trash icon** | Removes a false-positive box from the list. The deleted box won't be written when you save. |
| **Class dropdown** | Pick `Shoplifting` or `normal`. The Save button stays disabled until every box is assigned. |
| **Save labels** | Writes a YOLO `.txt` to `raw_frames/<folder>/labels/<name>.txt`. After save, the row turns green. |

---

## Workflow

### 1. Drop new frames into `raw_frames/`

```bash
mkdir raw_frames/store_2026_05_01
cp /path/to/camera/captures/*.jpg raw_frames/store_2026_05_01/
```

Click **Reload** in the dashboard — the folder appears in the dropdown.

### 2. Label image-by-image

For each image:

1. Click the row → person detection runs (~80 ms - 1 s on CPU depending on model size).
2. If the boxes look right, assign each one a class and **Save**.
3. If a box is wrong (mannequin, doorway, reflection), click the trash icon to delete it.
4. If a person was missed, lower the confidence slider and **Re-detect**, or try a larger model and Re-detect.
5. If no person is detected and there really is none, just move on — no `.txt` file is created. The image stays in the unlabelled state.

### 3. Re-edit a labelled image

Click any green row → instead of running detection, the dashboard loads the saved label file and shows the boxes with classes pre-filled. Edit, **Save** to overwrite.

To start fresh on a labelled image, click **Re-detect** — boxes from the model replace the saved ones (the file isn't touched until you click **Save**).

### 4. Prepare for training

When you've finished labelling a batch, click **Prepare for training**:

- Creates `raw_frames/<folder>/images/` if it doesn't exist.
- Moves every image that has a matching label file into `images/`.
- Images without a label (skipped because no person was detected, or you weren't sure) stay in the flat root — visually separated.

The folder now matches Roboflow's structure:

```
raw_frames/store_2026_05_01/
├── unlabelled_frame_5.jpg     ← stayed in flat root
├── unlabelled_frame_8.jpg
├── images/
│   ├── frame_1.jpg            ← labelled, moved here
│   ├── frame_2.jpg
│   └── ...
└── labels/
    ├── frame_1.txt
    ├── frame_2.txt
    └── ...
```

### 5. Merge into the training dataset

```bash
cp raw_frames/store_2026_05_01/images/* Shoplifting-Detection/train/images/
cp raw_frames/store_2026_05_01/labels/* Shoplifting-Detection/train/labels/
python3 train.py
```

---

## Model selection

The labelling page uses a **separate model** from the one used for `/predict` (which runs your trained `best.pt`). For labelling we want a generic person detector — Ultralytics' COCO-pretrained YOLOv26.

Five size variants ship in `models/`:

| Variant | File size | CPU per image (~640 px) | Use it when… |
|---|---|---|---|
| `yolo26n` | 5 MB | ~80 ms | CPU-only, interactive labelling, default |
| `yolo26s` | 20 MB | ~150 ms | CPU but you want better recall |
| `yolo26m` | 42 MB | ~350 ms | GPU, balanced |
| `yolo26l` | 51 MB | ~600 ms | GPU, accuracy-first |
| `yolo26x` | 113 MB | ~1.2 s | GPU only, maximum recall |

### Lazy loading

Only `yolo26n` is loaded into memory when the API starts. The other variants
load on first use and stay cached afterwards. So switching from `n` to `m` adds
a one-time ~1-3 s pause, then is instant.

### Practical tip

Start every batch with `yolo26n` to scan through quickly. If you notice the
detector consistently missing people in a particular image (small figures,
crowded shelves, partial occlusion), switch to a heavier variant and **Re-detect**
just that image.

---

## On-disk structure

### Before labelling

```
raw_frames/
└── batch_2026_05_01/
    ├── frame_001.jpg
    ├── frame_002.jpg
    └── frame_003.jpg
```

### Mid-labelling

```
raw_frames/
└── batch_2026_05_01/
    ├── frame_001.jpg
    ├── frame_002.jpg          ← saved a label
    ├── frame_003.jpg          ← skipped (no person)
    └── labels/                ← created on first save
        └── frame_002.txt
```

### After "Prepare for training"

```
raw_frames/
└── batch_2026_05_01/
    ├── frame_001.jpg          ← still unlabelled, in flat root
    ├── frame_003.jpg          ← skipped, in flat root
    ├── images/
    │   └── frame_002.jpg      ← moved here because it has a label
    └── labels/
        └── frame_002.txt
```

The `images/`+`labels/` pair is drop-in compatible with `Shoplifting-Detection/train/`.

---

## API endpoints

The labelling dashboard talks to these (all under `/label`):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/label/classes` | Returns the trained classes (`Shoplifting`, `normal`) |
| `GET` | `/label/folders` | Lists folders in `raw_frames/` with counts |
| `GET` | `/label/images?folder=…` | Lists images in a folder (flat + `images/`) and their label state |
| `GET` | `/label/models` | Lists yolo26 variants in `models/` and which are loaded |
| `GET` | `/label/load?folder=…&name=…` | Reads saved YOLO label, returns pixel-coord boxes with class IDs |
| `POST` | `/label/detect` | Runs person detection on the image (form fields: `folder`, `name`, `conf`, `model`) |
| `POST` | `/label/save` | Writes YOLO label to `<folder>/labels/<name>.txt` |
| `POST` | `/label/prepare` | Reorganises folder into Roboflow's `images/`+`labels/` layout |

Static files at `/raw_frames/<folder>/<path>` serve the images themselves.

---

## Limits (current iteration)

| Limitation | Workaround | Severity |
|---|---|---|
| No drawing of new boxes | If the detector misses a person, lower confidence + Re-detect, or pick a larger model | Medium |
| No box repositioning | Detector boxes are usually accurate enough; reject and try a different model if not | Low |
| One folder at a time | Open another folder when ready | Low |
| Conf threshold is per-detection only | Persists in the UI for the session, lost on browser refresh | Low |
| Only two classes hard-coded (`Shoplifting`, `normal`) | Edit `TARGET_CLASSES` in `api/main.py` and re-export the dataset if you need more | Low for this project |
| No multi-user support | Run a single dashboard instance per labeller | Acceptable for a small team |
