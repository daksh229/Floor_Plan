# Synthetic Floor-Plan Fixture Generator — How To Regenerate Dummy PDFs

> Everything you need to spin up a fresh test fixture for either the **electrical-BOM POC**, the **measurement-extraction POC**, or — soon — the **combined 6-page fixture** that exercises both at once. Over and over again, with deterministic seeding for variants.

---

## The three fixtures, at a glance

The repo has (or will have) three generators, each producing a different shape of PDF:

| Fixture | Pages | Generator | What it exercises |
|---|---|---|---|
| **A. Electrical-BOM (existing)** | 4 | `poc/synth/generate_sample.py` and `poc/synth/generate_procalc_style.py` | Page classifier, ROI extractor, YOLO symbol detection, OCR, room extraction, RAG matching, BOM assembly |
| **B. Measurement (current)** | 1 | `poc/synth/generate_measurement_sample.py` | Dimension OCR, wall detection, polygon construction, total floor area, scale calibration, bathroom count, window/door counts |
| **C. Combined 6-page (NEW — proposed)** | 6 | `poc/synth/generate_combined_sample.py` *(not yet built — see §"The 6-page combined fixture")* | Both A and B together — single PDF the wizard can route end-to-end, demonstrating the full Phase 1 ProCalc feature set on one upload |

---

## The 6-page combined fixture (NEW — proposed structure)

This is the demo asset for the ProCalc interview. **One PDF, six pages, every Phase 1 feature exercised.**

The page classifier (foundation piece #1 of the proposal — already built) reads the upload, identifies each page's role, and routes each downstream feature to the right page automatically. The wizard ends up with both an electrical BOM AND a measurement-extraction output from a single user click.

### Page-by-page layout

| # | Sheet | Style | Content | Exercises |
|---|---|---|---|---|
| **1** | **Cover** | Title page | Project name, client, address, drawing list (E-101, A-101, A-102, etc.), revision history, north arrow | `page_classifier` → `cover` |
| **2** | **Demolition Plan** | Existing-condition plan | Floor outline (light grey), existing rooms labelled, items to be demolished shown with dashed outline / X overlay | `page_classifier` → `demolition_plan`; future Phase-2 demolition diff |
| **3** | **Proposed Electrical Plan** | Architectural plan + electrical symbols | The architectural envelope from page 5, with electrical symbols (downlights, GPOs, switches, fans, smoke alarms, distribution board) placed in each room | YOLO symbol detection, OCR room extraction, room-wise BOM, RAG matching |
| **4** | **Electrical Legend & General Notes** | Reference page | Symbol legend table (15 symbols), general notes paragraph, signing-off | `page_classifier` → `legend`; legend extractor; user-pack ingestion |
| **5** | **Proposed Ground Floor Plan (Architectural)** | Clean architectural plan | Same envelope as page 3, but WITHOUT electrical symbols. Full perimeter dimensions on all four walls, room labels with internal dimensions, windows (W1–W7), doors (D1, SD1), scale bar, north arrow | **Measurement extraction:** dimension OCR, scale calibration, polygon construction, `total_floor_area`, `bathroom_count`, `glass_doors_windows`, `scale_detected` |
| **6** | **Window/Door Schedule + Area Schedule** | Data tables | Window schedule (W1 through W7: width × height, glazing, opening type), Door schedule (D1, SD1: width × height, type), Area schedule (per-room m², total floor area m², wet area total) | Cross-validation oracle for the measurements extracted from page 5; tests Claude's table-reading |

### Why this structure works

- **Pages 1, 2, 4** stay essentially identical to the existing electrical synth fixture — drop-in reuse of the `generate_procalc_style.py` generators
- **Page 3** is the existing electrical plan, but synchronised with page 5's architectural envelope (same room geometry, same room labels) so the same floor plan supports both extraction flavours
- **Page 5** is the existing `generate_measurement_sample.py` output, dropped in as a sheet
- **Page 6** is new — but it's just static tables that mirror the ground-truth JSON, so the generator can emit it programmatically without new geometric reasoning

### Output (per generated run)

| File | What it is |
|---|---|
| `samples/combined_<seed>.pdf` | The 6-page PDF (the demo asset) |
| `samples/combined_<seed>_page1.png` … `_page6.png` | Each page as a PNG (for inspection) |
| `samples/combined_<seed>_ground_truth.json` | Combined GT covering both electrical + measurement layers |

### Run (once built)

```bash
cd poc
python -m synth.generate_combined_sample --seed 1
```

---

## What you get per run (single-page measurement fixture)

| File | What it is |
|---|---|
| `samples/measurement_sample_<layout>_seed<NNNN>.pdf` | Single-page A3-landscape PDF, ready to drop into the wizard or hand to a client demo |
| `samples/measurement_sample_<layout>_seed<NNNN>.png` | Same image as PNG — quick visual sanity-check without opening the PDF |
| `samples/measurement_sample_<layout>_seed<NNNN>_ground_truth.json` | The machine-readable answer key: every room, wall, dimension, door, window, scale bar — coordinates in BOTH real-world millimetres and rendered pixels |

The ground truth is what makes this a proper test fixture rather than just a pretty picture — you can run the extraction pipeline over the PDF and diff its output against this JSON to score accuracy.

---

## Run it (60-second version)

From the `poc` directory:

```bash
python -m synth.generate_measurement_sample
```

That produces seed `1`, layout `6room`. Output paths print to stdout.

To produce a different variant:

```bash
python -m synth.generate_measurement_sample --seed 17
python -m synth.generate_measurement_sample --seed 42 --layout 6room
```

---

## What's in the generated PDF (every element labelled)

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  SCALE 1:50                                       (N north arrow)   │
 │  ┃▓┃ ┃▓┃ ┃▓┃                                          ⊙             │
 │   0  1m  2m  3m                                                     │
 │                                                                     │
 │   ←-- 2500 --→←-- 2500 --→←-- 2500 --→←-- 2500 --→                  │
 │   ┌───────────┬───────────┬───────────┬───────────┐  ↑              │
 │   │           │░░░░░░░░░░░│░░░░░░░░░░░│           │  |              │
 │   │  BED 2    │░ENSUITE░░░│░BATH░░░░░░│  BED 1    │  4000           │
 │   │ 2500×4000 │░░░░░░░░░░░│░░░░░░░░░░░│ 2500×4000 │  |              │
 │   │           │░░░░░░░░░░░│░░░░░░░░░░░│           │  v              │
 │   ├──┐  ──────┴───────────┴───────────┴────────────┤                │
 │   │ ⤺                                              │  ↑              │
 │   │  D1                                            │  |              │
 │   │      LIVING            │       KITCHEN         │  4000           │
 │   │     5000×4000          │      5000×4000        │  |              │
 │   │                        │                       │  v              │
 │   └────────────────────────┴───────────────────────┘                │
 │            ←------ 5000 ------→←------ 5000 ------→                 │
 │                                                                     │
 │ ┌─────────────┬─────────────────┬──────────────────────┐            │
 │ │ PROJECT     │ DRAWING         │ AREAS                │            │
 │ │ Synth #0001 │ PROPOSED GFP    │ Floor: 80.0 m²       │            │
 │ │ Brunswick E │ A-101  REV.A    │ Rooms: 6  Wet: 2     │            │
 │ └─────────────┴─────────────────┴──────────────────────┘            │
 └─────────────────────────────────────────────────────────────────────┘
```

What each element is there to exercise in the extraction pipeline:

| On the plan | What the extractor should derive from it |
|---|---|
| Wall outlines (thick black lines, ~14 px wall thickness at 1:50) | Polygon construction via Hough-line detection |
| **Perimeter dimensions** (`2500`, `2500`, `2500`, `2500` along the top; `5000`, `5000` along the bottom; `4000`, `4000` on each side) | **The dimension-driven scale calibrator** — read by OCR, paired with the wall they label, → mm-per-pixel ratio |
| Room labels + dimensions ("`BED 2`" + "`2500 x 4000`") | Room extraction + per-room area |
| Wet-area tinting (light blue fill on `BATH` and `ENSUITE`) | `bathroom_count` extraction with bboxes |
| Window markers `W1`–`W7` (two parallel blue lines across wall gap) | `glass_doors_windows` count + position |
| Entry door `D1` (gap in wall + quarter-arc swing) | Door detection — internal vs external |
| Sliding glass door `SD1` (double-line marker on bottom wall) | Glass-door detection, counts toward `glass_doors_windows` |
| Scale bar `SCALE 1:50` with `0–3m` tick labels (top-left) | `scale_detected` field; cross-validates the dimension-based calibration |
| North arrow (top-right) | Sheet metadata |
| Title block (bottom of page, 3 columns: project / drawing / areas) | Project metadata (extracted by Claude) |
| Drawing number `A-101`, revision `A`, scale `1:50` | More project metadata |

---

## The ground-truth JSON — what's inside

For every PDF the generator also writes a JSON file with the answer key. Its top-level structure:

```json
{
  "schema_version": "measurement-v1",
  "layout_name": "6room",
  "seed": 1,
  "page_px": [2480, 1754],
  "plan_origin_px": [350, 220],
  "render_px_per_mm": 0.16,
  "total_w_mm": 10000,
  "total_h_mm": 8000,
  "total_floor_area_m2": 80.0,
  "wall_thickness_mm": 90,
  "rooms":      [ ... ],
  "walls":      [ ... ],
  "dimensions": [ ... ],
  "windows":    [ ... ],
  "doors":      [ ... ],
  "scale_bar":  { ... }
}
```

| Field | Shape | Used for |
|---|---|---|
| `total_floor_area_m2` | float | Compare against the extractor's `total_floor_area` field |
| `rooms[]` | `{name, polygon_mm, centre_mm, centre_px, w_mm, h_mm, area_m2, is_wet}` | Room-extraction accuracy; bathroom-count regression |
| `walls[]` | `{id, kind: outer/internal, start_mm, end_mm, length_mm}` | Hough-line wall-detection regression |
| `dimensions[]` | `{edge, value_mm, segment_start_mm, segment_end_mm}` | Dimension-driven scale calibration — every entry here is a dimension your OCR should read |
| `windows[]` | `{label, wall, width_mm, centre_mm, centre_px}` | `glass_doors_windows` regression |
| `doors[]` | `{label, wall, position_mm, width_mm, is_external, is_glass}` | Door classification |
| `scale_bar` | `{bbox_px, label, represents_mm, length_px}` | `scale_detected` field |

Coordinate convention:

- `_mm` suffix: real-world millimetres, **plan-local** (origin = top-left of the outer wall, not the page)
- `_px` suffix: rendered pixels on the page, **page-local** (origin = top-left of the page)
- Conversion: `page_px = plan_origin_px + (mm * render_px_per_mm)`

---

## Customising the layout

All the layout knobs are at the top of `poc/synth/generate_measurement_sample.py`.

### Tier 1 — small tweaks (just edit numbers)

| Knob | Default | What it controls |
|---|---|---|
| `RENDER_PX_PER_MM` | `0.16` | How big the rendered plan is on the page. 0.16 = 1600 px for 10 m. Larger = bigger plan, smaller dimension labels relatively. |
| `WALL_THK_MM` | `90` | Wall thickness in real mm. 90 = standard timber-frame internal wall. Used for both rendering AND ground truth. |
| `DIM_OFFSET_PX` | `70` | How far the dimension chain sits outside the wall. Bigger = more breathing room, but eats page margin. |
| `DIM_TEXT_SIZE` | `22` | Font size of the dimension values |
| `ROOM_LABEL_SIZE` | `26` | Font size of the room name |
| `WALL_STROKE_PX` | `4` | (Note: not currently used — wall thickness is from `WALL_THK_MM`) |

### Tier 2 — change the room layout

Edit the `LAYOUT_6ROOM` dict in the generator. It's a plain Python dict with three top-level keys:

```python
LAYOUT_6ROOM = {
    "name": "6room_rectangular",
    "total_w_mm": 10000,     # outer wall envelope, width
    "total_h_mm": 8000,      # outer wall envelope, height
    "rooms": [
        Room("BED 2",   0,    0,    2500, 4000),
        Room("ENSUITE", 2500, 0,    2500, 4000, is_wet=True),
        # ...
    ],
    "windows": [
        Window("top",    1250, 1500, "W1"),
        # ...
    ],
    "doors": [
        Door("left",   5500, 900, is_external=True, label="D1"),
        Door("bottom", 5000, 2400, is_external=True, is_glass=True, label="SD1"),
    ],
}
```

**Room** has six fields:

| Field | Meaning |
|---|---|
| `name` | Display name on the plan (also what the extractor will OCR) |
| `x, y` | Top-left of the room in mm, relative to the plan's outer wall corner |
| `w, h` | Width and height in mm |
| `is_wet` | True → tinted light-blue, counts as a "wet area" in ground truth |

**Window** has four fields:

| Field | Meaning |
|---|---|
| `wall` | One of `"top"`, `"bottom"`, `"left"`, `"right"` (which outer wall it sits on) |
| `position_mm` | Centre of the window measured along the wall, from the wall's start corner (top-left convention) |
| `width_mm` | Window opening width (typical: 600 – 1800 mm) |
| `label` | The text label shown on the plan (e.g. `"W1"`) |

**Door** has five fields:

| Field | Meaning |
|---|---|
| `wall`, `position_mm`, `width_mm` | Same convention as windows |
| `is_external` | External doors get the entry-door style (arc swing into the building) |
| `is_glass` | Glass sliding doors — counts toward the `glass_doors_windows` total in the client schema |

### Tier 3 — adding a brand-new layout template

Add another entry to the `LAYOUTS` dict at the bottom of the generator:

```python
LAYOUTS = {
    "6room": LAYOUT_6ROOM,
    "lshape": LAYOUT_LSHAPE,   # new!
}
```

Define `LAYOUT_LSHAPE` the same way. Then run with `--layout lshape`.

For an L-shape plan, the simplest approach is to give it a bounding rectangle and add `Room` entries that don't cover the notch — the area calculation in the ground truth is room-polygon-area-summed, so the result is automatically the L-shape area, not the bounding rectangle.

---

## Variants via random seed

Currently the seed only affects metadata (project name suffix and revision). To make the seed actually vary the geometry, look at the `random.seed(seed)` call inside `render()` — you can add jitter such as:

```python
# Example: jitter window positions by ±200 mm
for win in layout["windows"]:
    win.position_mm += random.randint(-200, 200)
```

This is intentionally left as a hook rather than wired up, because most demos need a stable, predictable plan; you should opt in to randomness only when you want a regression corpus.

---

## Re-running fast (the iteration loop)

Typical workflow when tuning the generator:

```bash
# 1. Edit the layout / styling constants in
#    poc/synth/generate_measurement_sample.py

# 2. Regenerate (takes <1 second)
cd poc && python -m synth.generate_measurement_sample --seed 1

# 3. Inspect the PNG visually
#    Open samples/measurement_sample_6room_seed0001.png

# 4. Or drop the PDF straight into the wizard:
#    streamlit run app.py
#    then upload samples/measurement_sample_6room_seed0001.pdf
```

Everything runs locally with PIL — no Claude API call, no network. You can iterate hundreds of times per hour.

---

## A handful of generation recipes

Common variations you might want to produce:

| Goal | Command / edit |
|---|---|
| Stress-test scale calibration with 10 plans | `for s in {1..10}; do python -m synth.generate_measurement_sample --seed $s; done` |
| Larger building (15 m × 12 m) | Change `total_w_mm` to 15000, `total_h_mm` to 12000; update room geometry to match |
| Tighter scale (1:100 instead of 1:50) | Drop `RENDER_PX_PER_MM` to `0.08`, change `SCALE 1:50` label in `_draw_scale_bar` to `1:100` |
| No glass doors (count = 0 for that field) | Remove the `Door(..., is_glass=True, ...)` entry from `LAYOUT_6ROOM["doors"]` |
| Three wet areas | Add a Laundry room with `is_wet=True` |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Scale bar appears inside a room | Page is too cramped at the chosen `RENDER_PX_PER_MM`. Lower the value (e.g. `0.14`) so the plan shrinks and the scale bar's top-left position has clearance. |
| Dimension text overlaps the wall | Increase `DIM_OFFSET_PX` (e.g. 70 → 90). |
| Vertical dimension text looks blurry | Rotated text quality depends on PIL's `Image.rotate(expand=True, resample=BICUBIC)`; the script already uses it but at very small sizes (`DIM_TEXT_SIZE < 16`) it gets fuzzy. Keep size ≥ 18. |
| `OSError: cannot open resource` on first run | The script tries `arial.ttf`, `DejaVuSans.ttf`, `Helvetica.ttc`. None found on Windows → falls back to PIL's default (smaller, less crisp). Install `arial.ttf` (already present on most Windows systems) or run on macOS/Linux. |
| Want non-rectangular footprint (L-shape, T-shape) | Add a `LAYOUT_LSHAPE` entry — see Tier 3 above. The wall-deduplication logic in `_draw_walls` already handles non-rectangular envelopes; only the outer-perimeter rectangle drawing needs to be replaced with a polygon outline. |
| Ground truth says `area_m2: 80.0` but extractor returns 76 | Expected — extractor measures *internal* area (subtract wall thickness). Update its formula to use `(w - 2 * WALL_THK_MM) * (h - 2 * WALL_THK_MM)` if you want them to agree exactly. |

---

## What this fixture supports (and what it doesn't, yet)

**Supports today:**

- Rectangular footprint with arbitrary internal partition grid
- Perimeter dimension chains (4 edges)
- Wet-area tinting
- Windows + doors on outer walls
- External entry door (arc swing) + sliding glass door (double-line)
- Scale bar, north arrow, title block with project metadata
- A3 landscape page at 150 DPI

**Not yet (deliberate scope cap for v1):**

- Non-rectangular footprints (L-shape, T-shape) — the layout dict already supports it; only the outer-wall drawing code needs a small extension
- Internal door annotations (the only door drawn today is the entry door)
- Furniture / fixture symbols inside rooms (toilet, basin, kitchen island — would help test "this is definitely a bathroom" reasoning)
- Dimension chains that DON'T match wall segments (e.g. a "5000" overall dim above the four 2500/2500/2500/2500 segment dims — common in real plans, useful as a calibration redundancy test)
- Hatching for demolition walls — would be needed for Phase 2 demolition-diff testing

These are good follow-on additions when the extractor catches up to what's already generated.
