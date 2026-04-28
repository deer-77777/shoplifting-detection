# Dataset

## Source

Exported from Roboflow on **2026-04-27**:

- Workspace: `rakas-workspace-piisr`
- Project: `rakas-workspace-piisr`
- Version: `dataset`
- Licence: **Private** (do not redistribute outside this project)
- Format: **YOLO26** (ordinary YOLO bbox text format — same as YOLOv5/v8/v11/v26, the version number refers to the model the data is intended for, not a new file format)

## Volume

| Counter | Count |
|---|---|
| Total images | **7,576** |
| Total label files | 7,576 (1:1 with images) |
| Classes | 2 |
| Pre-processing applied | None |
| Augmentation applied | None |

## Classes

The model is trained as a **two-class detector**:

| `class_id` | name | Meaning |
|---|---|---|
| `0` | `Shoplifting` | A person whose visible behaviour matches a shoplifting pattern (concealment, scanning surroundings, etc.) |
| `1` | `normal` | A person in the frame who is *not* exhibiting suspicious behaviour |

This means **every person box is one of the two classes** — `normal` is not "background", it is an explicit positive class. As a consequence:

- Frames with no people will produce zero detections (correct).
- Even when a `Shoplifting` person is present, you also expect `normal` boxes for the other shoppers/staff in the frame.
- The dashboard's red ALERT chip only fires when at least one `Shoplifting` box is found, regardless of how many `normal` boxes accompany it.

## Growing the dataset

Once the system is deployed in a real store, the dataset can grow with frames captured from your own cameras. The labelling workflow ([LABELLING.md](LABELLING.md)) writes its output in the same YOLO format as the Roboflow export, organised under `raw_frames/<batch_name>/{images,labels}/`. Merging into the training set is then a plain copy:

```bash
cp raw_frames/<batch_name>/images/* Shoplifting-Detection/train/images/
cp raw_frames/<batch_name>/labels/* Shoplifting-Detection/train/labels/
```

After the merge, re-run `train.py` (or first re-run `split_dataset.py` if you want the new frames spread across train/valid/test).

## Folder layout

The Roboflow export ships with only `train/`. `split_dataset.py` materialises the `valid/` and `test/` folders the YAML expects.

```
Shoplifting-Detection/
├── data.yaml
├── README.roboflow.txt
├── train/
│   ├── images/   *.jpg
│   └── labels/   *.txt
├── valid/             (created by split_dataset.py)
│   ├── images/
│   └── labels/
└── test/              (created by split_dataset.py)
    ├── images/
    └── labels/
```

## Splits

Created by `split_dataset.py` using a fixed seed for reproducibility:

| Split | Count | Ratio | Purpose |
|---|---:|---:|---|
| `train` | 6,062 | 80 % | Model parameter updates |
| `valid` | 1,136 | 15 % | Per-epoch metrics, early-stopping, model selection |
| `test` | 378 | 5 % | Held-out final evaluation, never touched during tuning |

Split is a uniform random shuffle (no class stratification — the dataset has no per-class metadata at the file level, only inside the label files). With ~7.5k samples this is fine; both splits will have well-mixed class distributions.

**Re-running `split_dataset.py` is destructive**: it *moves* files out of `train/`. To re-roll the split, re-export from Roboflow first.

## Label format

YOLO normalised-bbox plain text. One line per object, fields separated by spaces:

```
<class_id> <cx> <cy> <w> <h>
```

All four box fields are normalised to `[0, 1]` relative to image width/height. Example from `train/labels/1_mp4-0_jpg.rf.bp0yv3XcvJPY0MtUZr5a.txt`:

```
1 0.4453125 0.3958984375 0.178125 0.598046875
```

→ class `normal`, centre at (44.5 %, 39.6 %), 17.8 % wide × 59.8 % tall.

## `data.yaml`

```yaml
path: ../Shoplifting-Detection
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Shoplifting', 'normal']
```

`path:` is the dataset root; the train/val/test paths are resolved relative to it. Ultralytics derives the matching `labels/` directory by replacing `images/` with `labels/` in each path — both folders must therefore exist.

## Known caveats

1. **Mixed bbox + segment annotations.** A handful of label files (4 of ~7,500) contain polygon segmentation data alongside bbox data. Ultralytics emits a warning and discards the polygons — fine for our detect-only training, but worth knowing if you ever try to re-purpose this dataset for instance segmentation.

2. **Frame-level labels, not action labels.** A box labelled `Shoplifting` is a *single still frame's* judgement of behaviour. Actual shoplifting is a temporal sequence (approach → conceal → leave). This dataset is therefore well-suited to per-frame detection but cannot directly train a temporal model — it would need additional sequence labelling.

3. **Class imbalance.** Not measured here, but typical surveillance datasets are heavily skewed toward `normal`. If you observe `Shoplifting` recall trailing precision, consider class-weighted loss or oversampling the minority class.

4. **Domain shift.** All footage comes from one workspace's collection. Performance on cameras with very different angles, lighting, or store layouts will likely require fine-tuning on local footage before deployment.
