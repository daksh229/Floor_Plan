# YOLO Detector Training Plan

A side project running in parallel with the interview-evidence POC. Goal:
**replace the template-matching CV stage with a trained YOLOv8-nano detector**
so the system stops needing per-class threshold tuning and reaches the
90-95 % F1 range we measured as the production target.

This document is the **single reference** for the data-generation phase, the
master-annotation schema, and the training pipeline that runs after the
dataset is complete.

---

## 1. Why YOLO over template matching

| | Template matching (current) | YOLO (proposed) |
| --- | --- | --- |
| F1 on variant_b (measured) | ~61 % | 90-95 % target |
| Threshold tuning | per-class, fragile | none |
| Scale invariance | manual scale list | learned |
| Rotation tolerance | 0/90/180/270 only | learned |
| Line-weight / anti-alias robustness | poor | very good |
| Runtime per page | 60-110 s | <1 s |
| Adding a new symbol | edit `library.json` + ingest | annotate + retrain |
| Training data needed | 0 | ~30-50 annotated pages |

The proposal already names this work as **Phase 2 of the engagement** — this
file makes it executable.

---

## 2. Dataset specification

**Target volume:** 50 synthetic plan images.

**Variation strategy:**
- Each image has a **different room layout** (different room counts, sizes,
  positions, connections).
- Each image uses the **same 15-symbol library** that the rest of the POC
  uses, so the trained model speaks the same `library_key` vocabulary as
  `pipeline/schemas.py:Detection.symbol_class`.
- Per-image augmentation built in: scale jitter, rotation, light noise +
  blur — same as `synth/generate_sample.py` already produces.

**Why synthetic, not real plans:**
- Self-annotated — we *know* the ground truth because we place every symbol.
- Reproducible — same seed gives same image.
- No client-confidential data leaving the laptop.
- A model trained on enough diverse synthetic data transfers well to real
  plans because it learns symbol *shape*, not paper texture.

**Image format:**
- PNG, ~2000×1400 px (matches synth fixture's canvas).
- White background, dark symbols + walls + room labels.

**Annotation format:** single master JSON file (`training_data/master.json`)
indexing every image's annotations. Spec in §4.

---

## 3. Generation workflow (user-managed)

> **Source of truth — the user-set protocol from the spec discussion:**
>
> *"I will say go then you will create the floor plan image with different
> structure but same input symbol and then add the exact coordinates in JSON
> X and again I say go then you make other image and save coordinate in
> same X JSON."*

Operational sequence (repeats 50 times):

1. **User says `go`** (in chat).
2. **AI generates one image:**
   - Picks a fresh room-layout seed.
   - Draws walls + room labels.
   - Places symbols per per-room fixture budget (with scale + rotation
     jitter, no overlaps).
   - Saves the rendered PNG to `training_data/images/img_NNN.png` (zero-padded).
3. **AI appends the image's annotations** to `training_data/master.json` —
   one entry per placed symbol with class + bbox + scale + rotation.
4. **AI confirms** to chat: *"Image N saved. Master JSON now has K
   annotations across N images."*
5. **User reviews** (optional spot-check by opening the PNG), then says
   `go` again for the next image.

The user controls cadence — they can stop after 10, 30, or 50 and resume
later. The master JSON is append-only; existing entries are never rewritten.

**Why iterative instead of batched:**
- User can sanity-check layouts as they're generated.
- Easy to bail out early if the layout generator drifts (regenerate batch
  with tweaks).
- Each "go" gives the user a checkpoint.

---

## 4. Master JSON schema (`training_data/master.json`)

Single file, COCO-flavoured, designed to be one-line convertible to either
YOLO format or Ultralytics format.

```json
{
  "schema_version": "1.0",
  "created_at": "2026-05-29T...",
  "image_dir": "training_data/images",
  "categories": [
    {"id": 0, "name": "ceiling_fan"},
    {"id": 1, "name": "ceiling_light"},
    {"id": 2, "name": "data_point"},
    {"id": 3, "name": "dimmer"},
    {"id": 4, "name": "distribution_board"},
    {"id": 5, "name": "double_gpo"},
    {"id": 6, "name": "downlight"},
    {"id": 7, "name": "exhaust_fan"},
    {"id": 8, "name": "single_gpo"},
    {"id": 9, "name": "single_pole_switch"},
    {"id": 10, "name": "smoke_detector"},
    {"id": 11, "name": "tv_point"},
    {"id": 12, "name": "two_way_switch"},
    {"id": 13, "name": "wall_light"},
    {"id": 14, "name": "wp_gpo"}
  ],
  "images": [
    {
      "id": 1,
      "file_name": "img_001.png",
      "width": 2000,
      "height": 1400,
      "layout_seed": 1001,
      "room_count": 7,
      "annotations": [
        {
          "category_id": 5,
          "category_name": "double_gpo",
          "bbox": [340, 612, 80, 80],
          "bbox_format": "xywh",
          "scale": 1.05,
          "rotation_deg": 0,
          "room": "Living"
        }
      ]
    }
  ]
}
```

**Field notes:**
- `bbox_format: "xywh"` — `[top-left-x, top-left-y, width, height]` in
  source-image pixels. Easy to convert to YOLO's normalised
  `[cx/W, cy/H, w/W, h/H]` later.
- `layout_seed` — feed back to the generator to reproduce exactly.
- `room` — useful for downstream analysis (which classes appear in which
  room types).

---

## 5. Training pipeline (after dataset is complete)

Run **after** the master JSON has all 50 images.

### 5.1 Convert master JSON → YOLO format

One-time conversion script: `training/convert_master_to_yolo.py`
- For each image, writes `<filename>.txt` next to the PNG with normalised
  bboxes.
- Splits 40 train / 10 val (sequential by id; or random with fixed seed).
- Writes `training/data.yaml`:
  ```yaml
  path: training_data
  train: images/train
  val: images/val
  names: [ceiling_fan, ceiling_light, ...]  # 15 classes
  ```

### 5.2 Training command

```powershell
pip install ultralytics
yolo detect train data=training/data.yaml model=yolov8n.pt epochs=80 imgsz=1024 batch=8 patience=15
```

**Hyperparameters chosen for this scale:**
- `model=yolov8n.pt` — 3.2 M params, runs CPU-only at ~200 ms/page.
- `epochs=80` — generous; early-stops via `patience=15` if val-mAP plateaus.
- `imgsz=1024` — matches the rendered page resolution we'd use at inference.
- `batch=8` — fits in 8 GB RAM; bump to 16 if a GPU is available.

### 5.3 Expected metrics

With 40 train / 10 val images of clean synthetic data, expect:
- **mAP@50 ≥ 0.95** on val (in-domain).
- On real-world variant_b / variant_c (out-of-domain), expect **F1 = 75-85 %**
  — much better than template-matching's 61 % but not as good as in-domain.

Improving real-world transfer requires either (a) adding 5-10 real plan
pages to the training set, or (b) heavier augmentation (paper-texture
overlays, line-weight variation).

---

## 6. Integration into the POC

The trained model becomes a new module:
`pipeline/yolo_detect.py` with the same `detect_symbols(...) -> list[Detection]`
contract as `pipeline/cv_detect.py`.

`pipeline/runner.py` chooses between detectors via a new flag:
```python
detector: Literal["template", "yolo"] = "yolo"  # default once trained
```

Everything downstream (NMS, RAG matching, BOM assembly, Streamlit UI) is
unchanged — Detection is Detection.

The Streamlit sidebar gets a *"CV backend: template / yolo"* radio (last
toggle in §"Demo controls").

---

## 7. File / directory layout

```
training_data/
├── images/
│   ├── img_001.png       ← created by AI on each "go"
│   ├── img_002.png
│   └── ...
└── master.json           ← appended-to on each "go"

training/
├── generate_image.py     ← the per-"go" image+annotation generator
├── convert_master_to_yolo.py   ← run once when dataset complete
├── data.yaml             ← output of conversion
└── runs/
    └── detect/<run-id>/  ← Ultralytics writes weights + logs here
```

`training_data/` is the **dataset** (gitignored — may grow to ~100 MB).
`training/` is **code** (committed).

---

## 8. Roles & dependencies

| Step | Owner | Tools |
| --- | --- | --- |
| 1. Generate 50 images (iteratively) | User triggers, AI executes | `training/generate_image.py` (built on demand) |
| 2. Sanity-review images | User | image viewer |
| 3. Run conversion script | User or whoever trains | `training/convert_master_to_yolo.py` |
| 4. Train YOLOv8 | Any ML engineer with `ultralytics` installed | `yolo detect train ...` |
| 5. Integrate `pipeline/yolo_detect.py` | Developer | code change + sidebar toggle |

---

## 9. Backlog / future iterations

- **Augmentation library**: add paper-texture overlays, perspective skew,
  line-weight perturbation to bridge the synthetic→real gap.
- **Active learning loop**: every time the user uploads a real plan and the
  YOLO detector misses something, capture the FN region + class as new
  training data and retrain monthly.
- **Symbol vocabulary growth**: when ProCalc adds a 16th symbol class, run
  `generate_image.py --extend` to produce 10 new training images that
  include the new class, then retrain.
- **Confidence calibration**: post-hoc calibration of YOLO confidences via
  Platt scaling so confidence ≈ true positive probability.

---

## 10. Operating instructions

For the user managing data generation:

1. Open the chat with the AI.
2. Confirm the AI has read this file.
3. Say **`go`** to generate image 1.
4. AI replies with: image filename, annotation count, running total.
5. Open the image to sanity-check (optional).
6. Say **`go`** again for image 2. Repeat until you have 50.
7. When done, run the conversion script + start training.

To **resume** after a break: just say **`go`** — the AI reads
`training_data/master.json` to know where you left off and generates the
next sequential image.

To **adjust** the layout style mid-batch (e.g. *"add more multi-storey-style
layouts"*): say so before the next `go`; the AI updates the layout
generator and confirms before producing the next image.
