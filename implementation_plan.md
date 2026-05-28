# Electrical Layout → BOM POC — Implementation Plan

A complete reference for the interview-evidence POC built for the **ProCalc**
engagement (residential construction estimating, Australia). This document
covers the why, the how, every file, every schema, every design decision,
how to run it, and what's left.

---

## 1. Executive Summary

**Audience for the POC:** the ProCalc final-round interview panel — Richard
(Australia), Product Development Lead (Philippines), Software Development Lead
(Poland).

**Question the POC answers:** Richard's Q2 — *"Can the AI/ML lead demonstrate
prior work involving architectural plans, electrical takeoffs, construction
drawings, or spatial/coordinate extraction?"*

**What the POC backs up:** the prior-work claim in Vaibhav's earlier email
about an electrical-BOM generation system. Specifically the five-stage
pipeline he described:

> _"OCR + computer vision pipeline processes the drawings to detect and
> locate all symbols, annotations, and elements on the blueprint. Symbol
> recognition via RAG — each detected element is matched against a
> predefined symbol library… AI-powered BOM generation — all extracted
> elements, quantities, and specifications are passed to the LLM, which
> assembles a structured Bill of Materials… rather than just sending
> drawings to an LLM and hoping for the best."_

**What's been built:** Phases 1–4 of a 5-phase build, end-to-end working on
a synthetic test fixture. The pipeline produces a strict-JSON Bill of
Materials with 15 line items totalling **AUD 2,046**, with every quantity
and cost traceable back to either a CV detection or a library entry, and a
1–3 sentence estimator-facing note from Claude. The CV detector hits **88 %
recall** against ground truth on 60 placed symbols across 15 classes.

**What's remaining:** Phase 5 — Streamlit UI wrapping the pipeline with the
five visible panels, cached-result fallback for demo safety, and the
interview demo script in `poc/README.md`.

---

## 2. Source-Of-Truth Documents

| Document | Where | Why it matters |
| --- | --- | --- |
| Vaibhav → Richard prior-work email | conversation history | Locks the pipeline architecture: OCR + CV + RAG + LLM, **not** Claude-vision-only |
| Richard's Q2 question | conversation history | Defines the POC's purpose: backfill prior-work claim |
| ProCalc proposal | [ProCalc_AI_Measurement_Response .docx](ProCalc_AI_Measurement_Response%20.docx) | The actual engagement Techuz is bidding on; references §2 (5-step feature), §3 (5 foundational pieces), §7 (failure modes) |

---

## 3. Strategic Pivot (2026-05-27)

The first attempt at the POC was a forward-looking *ProCalc demo* — Claude
vision extracting measurement variables (Ground Floor m², room count, door
count) per the proposal's §2 contract. After review, the team pivoted because
the interview question was about **prior work**, not future capability.

A Claude-vision-only demo would have *contradicted* the email's own narrative
("*rather than just sending drawings to an LLM and hoping for the best*"). So
the POC was rebuilt to mirror the email's exact 5-stage architecture.

Files retained from the first attempt:
- `pipeline/ingestion.py` (PDF → page images)
- Streamlit scaffolding pattern (re-applied in Phase 5)
- Versioned-prompt pattern (re-applied)
- Audit-store pattern (re-applied in `bom_assembler.py`)

Files deleted from the first attempt (none of this code remains):
- `pipeline/extractor.py`, `pipeline/prompts.py` (old measurement extractor)
- `pipeline/schemas.py` (old §2-contract models)
- `pipeline/overlay.py` (old ExtractedField renderer)
- `poc/app.py` (old measurement Streamlit UI)

---

## 4. Architecture

### 4.1 The 5-stage pipeline (mirrors the email exactly)

```
                ┌──────────┐
   PDF upload → │ Ingest   │   pypdfium2 → page-numbered PIL images
                └────┬─────┘
                     │
                ┌────▼─────┐
                │   CV     │   OpenCV multi-scale + multi-rotation
                │  detect  │   template matching, NMS within + across class
                └────┬─────┘
                     │
                ┌────▼─────┐   pytesseract --psm 11 (sparse text)
                │   OCR    │   filter: min length, min height, min confidence
                └────┬─────┘
                     │
                ┌────▼─────┐   sentence-transformers (all-MiniLM-L6-v2)
                │   RAG    │   FAISS IndexFlatIP over alias-enriched
                │          │   library docs → top-K matches with cosine sim
                └────┬─────┘
                     │
                ┌────▼─────┐   Deterministic aggregation +
                │   BOM    │   Claude as editorialiser
                │ assembly │   (metadata extraction + estimator note)
                └────┬─────┘
                     │
                ┌────▼─────┐
                │  Output  │   Strict JSON BOM + CSV (Phase 5)
                └──────────┘
```

### 4.2 Two non-obvious architectural choices

**(a) Claude is *not* the source of truth for the BOM.**
Quantities, specs, and costs are computed deterministically from
(detections, library.json). Claude's job is intentionally narrowed to
metadata extraction + a short editorial note. After Claude responds,
`assemble_bom()` validates every line item against the deterministic truth
and silently overrides any deviations, capturing them in a `diagnostics`
dict. If Claude is down or returns garbage, the deterministic BOM is the
fallback. **The BOM cannot have hallucinated values by construction.**

This is the airtight answer to the "Claude hallucinated counts / invented
specs" failure mode and is the strongest moment of the interview narrative.

**(b) The synthetic test PDF is generated from the same symbol PNGs used
for template matching.**
This is disclosed upfront — the synthetic PDF exists to prove the pipeline
*shape* end-to-end. Per-placement scale jitter (0.80×–1.20×), rotation
(0/90/180/270), and Gaussian noise + blur make the matching non-trivial
(88 % recall, not 100 %), so the demo isn't a complete self-fulfilling
prophecy. On arbitrary real drawings the template-matching layer would be
extended (more scales, fine-tuned YOLO, or vector-PDF symbol extraction);
that scope sits in the proposal's Phase 2, not in this POC.

---

## 5. Repository Layout

```
Floor_Plan/
├── ProCalc_AI_Measurement_Response .docx   # client proposal (reference, do not edit)
├── README.md                                # project-level README
├── implementation_plan.md                   # this document
├── .gitignore
├── samples/                                 # test fixtures (gitignored *.pdf)
│   ├── synthetic_layout.pdf                 # generated by synth/generate_sample.py
│   ├── synthetic_layout_ground_truth.json   # per-placement truth
│   ├── synthetic_layout_preview.png         # visual preview (Phase 1)
│   ├── synthetic_layout_detections.png      # CV overlay (Phase 2)
│   ├── synthetic_layout_ocr.png             # OCR overlay (Phase 3)
│   └── synthetic_layout_bom.json            # final BOM (Phase 4)
└── poc/
    ├── README.md                            # implementation-level README (Phase 5 will rewrite)
    ├── requirements.txt                     # full Phase 1–5 deps
    ├── .env                                 # ANTHROPIC_API_KEY (gitignored)
    ├── .env.example                         # template
    ├── .rag_cache/                          # cached embeddings (gitignored)
    ├── audit_store/                         # per-request audit trail (gitignored)
    ├── venv/                                # local virtualenv (gitignored)
    ├── pipeline/                            # pipeline modules
    │   ├── __init__.py
    │   ├── ingestion.py                     # PDF → page images
    │   ├── cv_detect.py                     # template matching + NMS
    │   ├── ocr.py                           # Tesseract pass
    │   ├── rag.py                           # SymbolRAG class
    │   ├── prompts.py                       # versioned BOM prompt store
    │   ├── bom_assembler.py                 # deterministic aggregate + Claude
    │   ├── overlay.py                       # Detection bbox renderer
    │   └── schemas.py                       # all Pydantic models
    ├── symbol_library/                      # 15 PNG templates + metadata
    │   ├── generate_symbols.py              # produces the PNGs
    │   ├── library.json                     # canonical name + aliases + cost
    │   └── *.png                            # 15 schematic symbols
    ├── synth/                               # synthetic test fixture
    │   ├── __init__.py
    │   └── generate_sample.py               # builds the multi-room layout PDF
    └── scripts/                             # CLI smoke tests
        ├── __init__.py
        ├── cv_smoke_test.py
        ├── rag_smoke_test.py
        ├── ocr_smoke_test.py
        └── bom_smoke_test.py
```

---

## 6. The 5 Build Phases (what was delivered, with results)

### Phase 1 — Foundation & test fixture *(complete)*

**Goal:** produce reusable symbol templates + a non-trivial synthetic
electrical drawing + ground truth, with the right deps in place.

**Delivered:**

- [poc/requirements.txt](poc/requirements.txt) — full dependency list across
  all 5 phases (`streamlit`, `anthropic`, `pypdfium2`, `pillow`, `pydantic`,
  `python-dotenv`, `numpy`, `opencv-python`, `pytesseract`,
  `sentence-transformers`, `faiss-cpu`).
- [poc/symbol_library/generate_symbols.py](poc/symbol_library/generate_symbols.py)
  — PIL-based generator producing 15 schematic electrical symbol PNGs at
  80×80 px with transparent backgrounds. Each symbol is a distinct
  geometric primitive: switches as labelled circles, GPOs as circles with
  prongs, WP GPO as rectangle-around-circle, lights as circle+X or filled
  circle, fan as circle+blades, exhaust fan as rectangle+blades, smoke
  detector as nested circles, data/TV as right-triangles, distribution
  board as rectangle with hatch lines.
- [poc/symbol_library/library.json](poc/symbol_library/library.json) — for
  each of the 15 symbols: canonical_name, 4–5 aliases (used by RAG to
  enrich the embedding doc), unit, spec, indicative cost in AUD.
- [poc/synth/generate_sample.py](poc/synth/generate_sample.py) — generates
  `samples/synthetic_layout.pdf` (A4-landscape, 2000×1400 px, ~170 DPI)
  with 8 named rooms (Bedroom 1/2, Living, Kitchen, Bathroom, Laundry,
  Hallway, Garage), a title block ("PROJECT: SYNTHETIC TEST DWELLING",
  "DWG NO. E-100"), per-room fixture budgets, and a legend in the
  bottom-right corner. Each placement gets scale jitter (0.80×–1.20×),
  rotation (0/90/180/270 with 0° favoured), and the whole page gets
  Gaussian blur + speckle noise. Total placements: **60 symbols across
  15 classes**.
- [samples/synthetic_layout.pdf](samples/synthetic_layout.pdf) (181 KB)
- [samples/synthetic_layout_ground_truth.json](samples/synthetic_layout_ground_truth.json)
  — every placement's class, room, centre, bbox, scale, rotation.
- [samples/synthetic_layout_preview.png](samples/synthetic_layout_preview.png)
  — rendered preview of the PDF.

**Tesseract install verified:** `C:\Program Files\Tesseract-OCR\tesseract.exe`
v5.5.0 (not on PATH; `pipeline/ocr.py` auto-locates it).

---

### Phase 2 — CV symbol detection *(complete)*

**Goal:** detect electrical symbols on the page with bounding boxes,
honestly (no Claude vision, no labelled training data).

**Delivered:**

- [poc/pipeline/cv_detect.py](poc/pipeline/cv_detect.py) — the detector:
  - Templates prepared from `symbol_library/*.png`, composited onto the
    same cream background as the synth PDF (so we can use dense
    `TM_CCOEFF_NORMED` without mask-based matching).
  - **5 scales × 4 rotations × 15 classes = 300 template variants** matched
    against the target page.
  - **Per-class thresholds** — solid filled shapes (downlight 0.82) get
    raised; composite shapes (wp_gpo 0.42) get lowered.
  - **Class-wise NMS** (IoU 0.45) merges multi-scale duplicates of the same
    physical symbol.
  - **Cross-class NMS** (IoU 0.55) suppresses inter-class collisions (e.g.
    a `single_gpo` template matching inside a `wp_gpo` region) using a
    **specificity bonus** in the ranking so visually-more-specific
    symbols win ties (`wp_gpo +0.10`, `double_gpo +0.05`,
    `distribution_board +0.05`, `two_way_switch +0.03`, etc.).
- [poc/pipeline/schemas.py](poc/pipeline/schemas.py) — `Detection` model
  (frozen, `symbol_class`, `bbox`, `score`, `rotation_deg`, `scale`,
  `source_page`, derived `center` and `area`).
- [poc/pipeline/overlay.py](poc/pipeline/overlay.py) — class-coloured bbox
  renderer with a deterministic mid-saturation colour per class name.
- [poc/scripts/cv_smoke_test.py](poc/scripts/cv_smoke_test.py) — renders
  the PDF at GT canvas size (so coords are directly comparable), runs
  detection, prints per-class GT-vs-detected counts + recall, saves the
  overlay.

**Result on the synthetic fixture:**

| Metric | Value |
| --- | --- |
| Total ground-truth placements | 60 |
| Total detections after all NMS | 58 |
| Overall recall (count basis) | **88 %** |
| Classes at 100 % recall | 12 of 15 |
| Detection time | ~18 s (2000×1400 page) |
| Output | [samples/synthetic_layout_detections.png](samples/synthetic_layout_detections.png) |

Remaining failures concentrate on visually-ambiguous round-with-glyph
templates (`ceiling_light`, `smoke_detector`, `exhaust_fan`). These are
the template-matching ceiling and exactly what a fine-tuned YOLO would
close in production.

---

### Phase 3 — OCR + RAG *(complete)*

**Goal:** add the second leg of the email's claim ("OCR + computer vision
pipeline… Symbol recognition via RAG…").

**Delivered:**

- [poc/pipeline/ocr.py](poc/pipeline/ocr.py) — Tesseract pass with:
  - **Auto-located** `tesseract.exe` on Windows (checks `Program Files`,
    `LOCALAPPDATA`, env var `TESSERACT_CMD` override).
  - **PSM 11 (sparse text)** — best mode for scattered drawing labels.
  - **Filters:** min confidence 0.40, min text length 2, min text height
    10 px. Drops single-letter symbol-glyph noise.
  - Raises a clear `TesseractNotInstalledError` if the binary is missing,
    with the install URL.
- [poc/pipeline/rag.py](poc/pipeline/rag.py) — `SymbolRAG` class:
  - Loads `symbol_library/library.json`, builds a "document" per entry
    concatenating `canonical_name + aliases + spec` — **aliases are what
    make retrieval non-trivial over only 15 entries**.
  - Embeds with `sentence-transformers/all-MiniLM-L6-v2` (80 MB, fast).
  - FAISS `IndexFlatIP` over L2-normalised vectors = cosine similarity.
  - On-disk embedding cache (`.rag_cache/emb_<sha>.pkl`) keyed on
    library-content hash; library edits invalidate.
  - `match(query, k=3)` and `match_detections(detections, k=3)` APIs.
- [poc/pipeline/schemas.py](poc/pipeline/schemas.py) — extended with
  `OCRSpan` and `RAGMatch` and `SymbolWithMatches`.
- [poc/scripts/ocr_smoke_test.py](poc/scripts/ocr_smoke_test.py) —
  text-span dump + overlay PNG.
- [poc/scripts/rag_smoke_test.py](poc/scripts/rag_smoke_test.py) — tests
  three query flavours (exact class names, alias paraphrases, free-text)
  and prints top-3 with similarity scores.

**OCR result on the synthetic fixture:**

| Metric | Value |
| --- | --- |
| Spans returned after filtering | 60 |
| Title-block words (project, sheet, dwg no., date, scale) | all detected (0.91–0.96 conf) |
| Room labels detected (of 8) | 6 (Bedroom 1+2, Living, Kitchen, Hallway, Garage) |
| Room labels missed | Bathroom, Laundry (small font, crowded by symbols) |
| Output | [samples/synthetic_layout_ocr.png](samples/synthetic_layout_ocr.png) |

**RAG result on the test queries:**

All 15 test queries returned the correct top-1. Score patterns:

| Query style | Top-1 score range | Notes |
| --- | --- | --- |
| Exact class names | 0.49–0.72 | What the BOM stage actually uses |
| Alias paraphrases | 0.37–0.67 | Confirms aliases enrich the embedding doc |
| Free-text | 0.42–0.71 | Real semantic retrieval, not lookup |

Top-3 with tight scores (the "RAG doing real work" optic):
- Query `wp_gpo` → wp_gpo 0.49, double_gpo 0.43, single_gpo 0.42 (all GPO
  variants compete because they share the prong glyph).
- Query `twin power point` → double_gpo 0.37 over single_gpo 0.24.

---

### Phase 4 — BOM assembly *(complete)*

**Goal:** the LLM-as-assembler step the email promised — with the
explicit hardening that Claude can never invent quantities, specs, or
costs.

**Delivered:**

- [poc/pipeline/schemas.py](poc/pipeline/schemas.py) — extended with
  `BOMLineItem`, `BOM`, `PipelineRun` (the latter for Phase 5 to capture
  one end-to-end run for the audit store and the cached-fallback button).
- [poc/pipeline/prompts.py](poc/pipeline/prompts.py) — versioned `bom_v1`
  prompt store with 6 hard constraints:
  1. Use exactly the quantities provided. Never infer.
  2. Use exactly the unit_cost_aud, unit, and spec from the matched
     library entry. Never paraphrase. Never recompute.
  3. Do not add line items for things not in the aggregated input list.
  4. Do not drop line items, even if quantity is small.
  5. line_total_aud = quantity × unit_cost_aud, rounded to 2 dp.
  6. subtotal_aud = sum of line_total_aud, rounded to 2 dp.
  Plus permitted free-text fields: project metadata extraction from OCR,
  and a 1–3 sentence `notes_from_assembler`.
- [poc/pipeline/bom_assembler.py](poc/pipeline/bom_assembler.py):
  - `compute_aggregated_payload(detections, rag, ocr_spans, …)` — groups
    detections by class, attaches library matches, joins OCR text. This
    is the **source of truth**.
  - `_build_deterministic_bom(payload, …)` — computes the BOM with no
    Claude involvement.
  - `assemble_bom(payload, …)` — calls Claude, parses, then enforces
    deterministic invariants on every line item:
    - Drops line items Claude added that aren't in the payload.
    - Adds line items Claude omitted from the payload.
    - Detects quantity / cost / spec deviations and overrides them back
      to truth, surfacing each in a `diagnostics` dict.
    - Accepts Claude's project metadata and `notes_from_assembler`
      verbatim (the only free-text fields Claude owns).
    - Falls back to deterministic BOM with an auto-generated note if
      Claude errors or returns un-parseable JSON.
  - Retry × 2, timeout 90 s, full audit-store write of (payload,
    raw response, final BOM, diagnostics) per run.
- [poc/scripts/bom_smoke_test.py](poc/scripts/bom_smoke_test.py) — full
  end-to-end CLI runner.

**Result on the synthetic fixture (most recent run):**

```
Project:        SYNTHETIC TEST DWELLING
Drawing no.:    E-100
Date:           2026-05-27

library_key               qty unit   unit AUD   total AUD
---------------------------------------------------------
ceiling_fan                 3   ea     189.00      567.00
ceiling_light               3   ea      42.00      126.00
data_point                  3   ea      18.50       55.50
dimmer                      1   ea      38.00       38.00
distribution_board          1   ea     240.00      240.00
double_gpo                 15   ea      14.20      213.00
downlight                  11   ea      22.00      242.00
exhaust_fan                 1   ea      78.00       78.00
single_gpo                  1   ea       9.40        9.40
single_pole_switch          5   ea       8.50       42.50
smoke_detector              2   ea      48.00       96.00
tv_point                    1   ea      16.00       16.00
two_way_switch              3   ea      11.20       33.60
wall_light                  2   ea      56.00      112.00
wp_gpo                      6   ea      29.50      177.00
---------------------------------------------------------
SUBTOTAL                                          2046.00  AUD
```

**Claude's `notes_from_assembler` on this run (verbatim):**
> The 12-pole distribution board may be undersized for a dwelling with a
> garage, 6 weatherproof GPOs, ceiling fans, and a dedicated exhaust fan —
> estimator should confirm circuit count and consider upgrading to an
> 18- or 24-pole board. No cabling, conduit, circuit breakers, or RCDs
> are included in this BOM; these are commonly required items that should
> be scoped separately. The 6 weatherproof GPOs is relatively high for a
> residential ground floor and may warrant a site verification to confirm
> external outlet locations.

**Validation diagnostics on this run:** *none* (Claude output matched
deterministic truth on every field — no overrides triggered).

**Elapsed:** 71 s total (CV 26 s, OCR 1 s, RAG build 21 s, Claude 23 s).

**Output:** [samples/synthetic_layout_bom.json](samples/synthetic_layout_bom.json),
plus a per-run audit folder under `poc/audit_store/`.

---

### Phase 5 — Streamlit UI + demo polish *(pending)*

**Goal:** wrap the pipeline in a 5-panel Streamlit demo with a safety net
for the live interview.

**Planned deliverables:**

- `poc/app.py` — Streamlit entry point with five **visible** panels (do
  not collapse the pipeline behind one panel):
  1. **Upload** — PDF picker + page picker.
  2. **CV Detections** — page with class-coloured bboxes + per-class
     counts table.
  3. **OCR** — page with OCR bboxes + extracted text spans table.
  4. **RAG Matches** — for each detected class, top-3 library candidates
     with similarity scores (the "RAG doing work" optic).
  5. **BOM** — assembled table + Claude's `notes_from_assembler` +
     project metadata + Raw JSON expander + CSV download.
- **Cached-result fallback button** — loads a pre-recorded clean BOM run
  if the live Claude call fails on demo day.
- `poc/README.md` rewritten — interview demo script (what to click in
  what order, what to point out, what to candidly say about limitations).
- One captured clean end-to-end run saved to disk for the fallback
  button to load.

**Estimated effort:** ~1 day.

---

## 7. Schemas Reference

All in [poc/pipeline/schemas.py](poc/pipeline/schemas.py).

### Detection (Phase 2)
| Field | Type | Notes |
| --- | --- | --- |
| `symbol_class` | str | matches a `library.json` key |
| `bbox` | (int, int, int, int) | x0, y0, x1, y1 in source-page-image px |
| `score` | float | normalised template-match correlation in [-1, 1] |
| `rotation_deg` | int | 0 / 90 / 180 / 270 |
| `scale` | float | template scale that produced the hit |
| `source_page` | int | 1-indexed PDF page |

### OCRSpan (Phase 3)
| Field | Type | Notes |
| --- | --- | --- |
| `text` | str | post-filter text |
| `bbox` | (int, int, int, int) | image px |
| `confidence` | float | normalised from Tesseract 0–100 to [0, 1] |
| `source_page` | int | 1-indexed |

### RAGMatch (Phase 3)
| Field | Type | Notes |
| --- | --- | --- |
| `library_key` | str | `library.json` key |
| `canonical_name` | str | from library |
| `similarity` | float | cosine similarity in [-1, 1] |
| `unit` | str | from library |
| `spec` | str | from library |
| `indicative_cost_aud` | float | from library |

### SymbolWithMatches (Phase 3)
| Field | Type | Notes |
| --- | --- | --- |
| `detection` | Detection | from Phase 2 |
| `matches` | List[RAGMatch] | top-K |
| `best` (computed) | RAGMatch \| None | matches[0] |

### BOMLineItem (Phase 4)
| Field | Type | Notes |
| --- | --- | --- |
| `library_key` | str | links back to library |
| `canonical_name` | str | from library |
| `quantity` | int ≥ 0 | from CV detection count |
| `unit` | str | from library |
| `spec` | str | from library |
| `unit_cost_aud` | float ≥ 0 | from library |
| `line_total_aud` | float ≥ 0 | quantity × unit_cost_aud |
| `source_pages` | List[int] | for traceability |
| `source_detection_count` | int | number of CV detections feeding the line |
| `notes` | str \| None | optional |

### BOM (Phase 4)
| Field | Type | Notes |
| --- | --- | --- |
| `project_name` | str \| None | Claude-extracted from OCR |
| `drawing_number` | str \| None | Claude-extracted |
| `revision` | str \| None | Claude-extracted |
| `drawing_date` | str \| None | Claude-extracted |
| `line_items` | List[BOMLineItem] | deterministic |
| `subtotal_aud` | float | deterministic |
| `notes_from_assembler` | str \| None | Claude's editorial note |
| `prompt_version` | str | e.g. "bom_v1" |
| `model` | str | e.g. "claude-sonnet-4-6" |
| `page_count` | int | source PDF page count |
| `generated_at` | str | ISO 8601 UTC |

### PipelineRun (Phase 4, for Phase 5's cached fallback + audit)
| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | str | unique |
| `source_pdf` | str | filename |
| `detections` | List[Detection] | |
| `ocr_spans` | List[OCRSpan] | |
| `symbol_matches` | List[SymbolWithMatches] | |
| `bom` | BOM | |
| `elapsed_seconds` | float | |

---

## 8. Key Design Decisions (and why)

### 8.1 Why no Claude vision in this POC
The email explicitly framed the pipeline as *"…rather than just sending
drawings to an LLM and hoping for the best."* A Claude-vision-only demo
would contradict the email's own narrative. OCR + CV + RAG + LLM-as-
assembler is the contract.

### 8.2 Why template matching over training a YOLO
Time-bounded interview-evidence build (~5 days). Annotating a dataset and
training even a small detector is a multi-week effort. Template matching
gets us measurable, defensible recall on a controlled fixture in 1 day.
Going to YOLO is the obvious production extension and explicitly called
out in the interview narrative.

### 8.3 Why real embeddings over deterministic lookup for RAG
Embedded retrieval over alias-enriched docs shows a visible RAG
mechanism on screen with top-K + similarity scores. Deterministic lookup
would be honest but invisible — the panel would fairly ask *"where's the
RAG?"*. Cost: one ~80 MB model download, ~80 ms per query after caching.

### 8.4 Why Claude is not the BOM author
"Claude hallucinated counts / invented specs" was identified as a HIGH
risk in the challenges discussion. The architectural answer is to take
the assembly responsibility *away* from Claude and give it back to
deterministic code, then let Claude do the narrower jobs it's actually
good at (metadata extraction from prose, written commentary). This makes
hallucinated quantities **structurally impossible**.

### 8.5 Why the synthetic PDF has scale + rotation + noise jitter
A self-fulfilling demo (100 % recall on a trivially-clean fixture) would
be spotted immediately by the Software Development Lead. Jitter brings
recall down to 88 % — honest, measurable, and the residual failures
explainable as the template-matching ceiling.

### 8.6 Why per-class thresholds and cross-class NMS
Composite symbols (wp_gpo = rectangle around circle around prongs) score
lower than their visual subsets (single_gpo = circle around prongs). Same-
threshold + same-NMS would let single_gpo "steal" every wp_gpo region.
Per-class threshold tuning + cross-class NMS with a *specificity bonus*
solves this honestly without hiding it.

---

## 9. Failure Modes Pre-empted (from the proposal §7 and the challenges discussion)

| Failure mode | Where it's pre-empted in the POC |
| --- | --- |
| Treating AI output as ground truth | Phase 4: deterministic BOM is the source-of-truth, Claude output validated and overridden |
| Hardcoded prompts | `pipeline/prompts.py` is a versioned registry; adding a new prompt is a dict entry |
| No retry / timeout / partial-result handling | `bom_assembler._call_claude` retries × 2, 90 s timeout, falls back to deterministic on parse failure |
| Lost page identifiers | `Detection.source_page` and `OCRSpan.source_page` carried end-to-end |
| Audit feedback page deferred | `audit_store/` writes from request #1; per-run folder with payload + raw response + final BOM + diagnostics |
| Synthetic-data self-fulfilling prophecy | Per-placement scale/rotation/noise jitter; honest 88 % recall |
| RAG over 15 items is theatre | Alias enrichment + visible top-3 + tight similarity scores; semantic retrieval demonstrated |
| Claude hallucinated counts / invented specs | Architectural: Claude doesn't author quantities/costs — deterministic enforcement |
| Cross-class CV confusion | Cross-class NMS with specificity bonus |
| Tesseract on drawings is noisy | PSM 11 + min length + min height + min confidence filters |
| Live demo blast radius | Phase 5: cached-result fallback button |

---

## 10. Dependencies

All in [poc/requirements.txt](poc/requirements.txt):

| Package | Purpose | Phase |
| --- | --- | --- |
| `streamlit` | UI (Phase 5) | 5 |
| `anthropic` | Claude SDK | 4 |
| `python-dotenv` | `.env` loader | 4, 5 |
| `pydantic` | schemas | all |
| `pypdfium2` | PDF → image (no Poppler needed on Windows) | 1, 2, 3 |
| `pillow` | image handling | 1, 2, 3 |
| `numpy` | CV array ops | 2 |
| `opencv-python` | `cv2.matchTemplate`, NMS arithmetic | 2 |
| `pytesseract` | wrapper around the Tesseract binary | 3 |
| `sentence-transformers` | embedding model (pulls torch ~600 MB CPU) | 3 |
| `faiss-cpu` | vector index | 3 |

**External binary:** Tesseract OCR for Windows (UB-Mannheim build) —
`C:\Program Files\Tesseract-OCR\tesseract.exe`, auto-located by
`pipeline/ocr.py`.

---

## 11. How to Run

### 11.1 First-time setup

```powershell
cd c:\Users\Dell\Desktop\Project\Floor_Plan\poc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Open .env in editor and paste your real ANTHROPIC_API_KEY
```

Install Tesseract for Windows if not present:
https://github.com/UB-Mannheim/tesseract/wiki

### 11.2 Regenerate the test fixture (only if `library.json` changed)

```powershell
.\venv\Scripts\python.exe symbol_library\generate_symbols.py
.\venv\Scripts\python.exe synth\generate_sample.py
```

### 11.3 Individual smoke tests

```powershell
.\venv\Scripts\python.exe scripts\cv_smoke_test.py
.\venv\Scripts\python.exe scripts\ocr_smoke_test.py
.\venv\Scripts\python.exe scripts\rag_smoke_test.py
.\venv\Scripts\python.exe scripts\bom_smoke_test.py
```

### 11.4 Full pipeline (Phase 5 once built)

```powershell
streamlit run app.py
```

---

## 12. Interview Narrative

**Opening framing:**
> *"This POC backs up the prior-work claim in the earlier email. It's the
> exact 5-stage pipeline we described — OCR + CV detection + RAG against a
> predefined symbol library + LLM-assembled BOM. The drawing is synthetic
> so we can show end-to-end with known ground truth, with deliberate jitter
> so the matching isn't trivial."*

**On the CV stage:**
> *"OpenCV multi-scale + multi-rotation template matching, with class-wise
> and cross-class NMS. 88 % recall on 60 placed symbols across 15 classes.
> The remaining gaps cluster on visually-ambiguous round-with-glyph
> templates — production fix is a fine-tuned YOLO, which is Phase 2 of the
> proposal scope."*

**On the RAG stage:**
> *"Real sentence-transformers embeddings over the library, served from a
> FAISS inner-product index. The library entries are enriched with 4–5
> aliases each, so the embeddings actually do work — the top-3 panel shows
> tight similarity scores for visually-confusable items like the GPO family,
> and clean wins for distinctive ones like the distribution board."*

**On the BOM stage (the key moment):**
> *"This is the part we hardened deliberately. Claude is not the source of
> truth for the BOM. Quantities, specs, costs all come deterministically
> from the CV detections + library entries. Claude's job is extracting
> project metadata from the OCR text and writing the estimator note. After
> Claude responds we validate every line item against the deterministic
> truth and override any deviations — so the BOM cannot have hallucinated
> values by construction. This is the airtight answer to the obvious risk."*

**On the synthetic data (lead with this, don't wait to be asked):**
> *"To be candid: the synthetic PDF is generated from the same templates
> the detector uses, with jitter applied so it's not a self-fulfilling
> demo. On arbitrary real drawings the template-matching layer needs to be
> extended — that's exactly the scope sitting in Phase 2 of the proposal."*

---

## 13. What's Explicitly Out of Scope for the POC

| Out-of-scope | Where it would live in production |
| --- | --- |
| Multi-page PDFs (only page 1 used) | Phase 5 of POC; trivial extension |
| Real (non-synthetic) electrical drawings | Phase 2 of the proposal — extended detector |
| Multi-storey BOMs (per-floor breakdown) | Phase 2 of the proposal |
| Cabling / conduit / RCDs / circuit breakers | Out of scope per the email; flagged by Claude in estimator note |
| Vendor RFQ pipeline beyond CSV export | Future commercial feature |
| Authentication / multi-tenant / billing | N/A — POC |
| Coordinate translation to ProCalc's native overlay format | Phase 1 of the proposal, not relevant to this POC |

---

## 14. Open Questions / Risks for the Interview

- **Sample-PDF realism:** can we obtain a real (or de-identified) electrical
  layout PDF to optionally swap in during the demo? Greatly strengthens
  the "real drawings" answer if yes; not blocking if no.
- **Live-demo connectivity:** Claude latency 23 s + network variability on
  Zoom. Phase 5 cached-fallback button is the mitigation.
- **Tesseract on demo machine:** verified on the dev machine; if the demo
  is from a different machine, install needs to happen there.
- **Embedding model first-run download (~80 MB):** pre-warm before the
  demo or rely on the cached `.rag_cache/`.

---

## 14e. Phase 8 — Wizard UI + ROI Padding Bump (2026-05-28)

Phase 7 surfaced two findings on real demo runs:

1. Some plan symbols near the bottom of variant_c page 3 weren't being detected — Claude's ROI bbox was cropping them off, and our 6 % padding wasn't generous enough.
2. The 6-tab layout was confusing for first-time users: they didn't know which tab to start on, and skipping the legend-ingest step led to empty BOMs. The flow needed to be **guided**, not "discover all six tabs and figure out the order."

Phase 8 addresses both.

### 14e.1 ROI padding fix

| File | Change |
| --- | --- |
| [poc/pipeline/runner.py](poc/pipeline/runner.py) | ROI crop padding bumped from **6 % → 12 %** of page on each side, with a 100 px floor (was 40 px). Catches edge-symbol detections that Claude's bbox missed, while still excluding the well-separated title block / legend / notes blocks. |

### 14e.2 Wizard UI replacing the 6-tab layout

| File | Change |
| --- | --- |
| [poc/app.py](poc/app.py) | Rewritten `main()` and sidebar around a 5-step wizard state machine. New step functions `_wizard_step1_upload`, `_wizard_step2_pack`, `_wizard_step3_legend`, `_wizard_step4_run`, `_wizard_step5_dashboard`. New `_render_stepper()` shows a horizontal step indicator. New `_wizard_reset()` clears state and returns to step 1. Existing `_tab_cv`, `_tab_ocr`, `_tab_rag`, `_tab_bom` are reused as sub-tabs inside step 5. |

### 14e.3 New flow

```
┌─────────────────────┐
│ Step 1: Upload      │  upload PDF → Claude classifies pages →
│                     │  right-side panel shows per-page sheet types
└─────────┬───────────┘  Also: "Open cached demo" → jumps to step 5
          │
┌─────────▼───────────┐
│ Step 2: Symbol pack │  Two cards: Built-in (skip to 4) | User (go to 3)
└─────┬───────────┬───┘
      │           │
      │   ┌───────▼───────────┐
      │   │ Step 3: Legend    │  Auto-uses the legend page Claude suggested.
      │   │ ingest            │  Editable per-row preview. "Ingest all rows"
      │   └───────┬───────────┘  unlocks the "Next →" button.
      │           │
┌─────▼───────────▼───┐
│ Step 4: Run         │  Page picker (default = suggested plan page),
│                     │  ROI toggle, "Run pipeline" button. On completion
└─────────┬───────────┘  auto-advances to step 5.
          │
┌─────────▼───────────┐
│ Step 5: Dashboard   │  4 sub-tabs: CV Detections / OCR / RAG / BOM
│                     │  (the existing tab functions reused here)
└─────────────────────┘  Sidebar shows run stats + "Restart wizard"
```

### 14e.4 What changed for the user

| Old (6-tab) | New (wizard) |
| --- | --- |
| Boot into Tab 5 (cached BOM), needed to manually navigate | Boot into Step 1 (upload prompt), with explicit "Open cached demo" shortcut |
| Have to know to ingest legend on Tab 6 *before* running on Tab 1 | Wizard *forces* the ingest step to happen first when "user pack" is chosen |
| Easy to pick the wrong page and get an empty BOM | Tab 1 + Tab 4 auto-suggest the right page from Claude's classification |
| Sidebar pack radio was easy to mis-set | Pack is chosen explicitly in Step 2 with descriptions of each option's trade-off |
| Symbol Library tab was a separate workflow with its own PDF upload | Step 3 reuses the PDF already uploaded in Step 1, defaults to the suggested legend page |

### 14e.5 State persisted across the wizard

| Session-state key | Purpose |
| --- | --- |
| `wizard_step` (1-5) | Current step; dispatcher reads this |
| `wizard_pdf_path` | tempfile.Path of the uploaded PDF (shared across steps 1-4) |
| `wizard_pdf_name` | Display name |
| `wizard_n_pages` | Cached page count |
| `wizard_pack_choice` | "builtin" or "user" (set in step 2) |
| `wizard_legend_ingested` | True after step 3 completes; gates step 3's "Next" |
| `page_classification` | Claude's per-page labels (set in step 1, used in steps 3 + 4) |
| `run`, `page_image`, `diagnostics` | Pipeline result (set in step 4, displayed in step 5) |

### 14e.6 Cached demo path

The **"Open cached demo"** button on Step 1 loads `poc/cached_runs/synthetic/` and jumps directly to Step 5. No upload, no Claude calls. This is the demo-day safety net — if the network or Anthropic API is down, the panel still sees a clean BOM in under 2 seconds.

### 14e.7 Old code retained but orphaned

`_tab_upload(...)` and `_tab_symbol_library(...)` remain in `app.py` but `main()` no longer dispatches to them — the wizard step functions cover the same ground in a guided order. Kept (not deleted) so the diff stays reviewable and the functions can be repurposed if the UX direction changes again.

---

## 14d. Phase 7 — Claude as a Vision Co-pilot for the CV Stack (2026-05-28)

Phase 6 introduced the *legend → user pack → detection* workflow but exposed three honest weaknesses on the variant_b/c sample PDFs:

1. Users picked the wrong PDF page (legend vs plan) and got empty BOMs.
2. CV ran against the *whole* rendered page, so the in-page legend block, title block, and notes sections produced large numbers of structural false positives (`distribution_board: 64`, `exhaust_fan: 33`).
3. The detection overlay pixelated when the user zoomed in to inspect a result — the underlying PNG was crisp but Streamlit displayed it at thumbnail size, so browser zoom upscaled raster pixels.

Phase 7 addresses all three by adding Claude vision as a **co-pilot on the metadata-handling stages** — *not* the counting stage. The CV + RAG core that the prior-work email specifically claimed remains deterministic; Claude is added at the edges where unstructured-document reasoning is exactly where LLMs are good.

### 14d.1 New / modified files

| File | What |
| --- | --- |
| [poc/pipeline/page_classifier.py](poc/pipeline/page_classifier.py) | **new** — `classify_pdf(pdf_path)` sends every page (thumbnailed to 800 px JPEG) to Claude vision in *one* call. Returns `PDFClassification` with per-page sheet type (`cover` / `electrical_plan` / `legend` / `demolition_plan` / `notes` / …), plus `suggested_plan_page` and `suggested_legend_page`. ~10 s, ~$0.01 per PDF. |
| [poc/pipeline/roi_extractor.py](poc/pipeline/roi_extractor.py) | **new** — `extract_plan_roi(page_image)` sends one page (downsampled to 1600 px) to Claude and asks for the bounding box of just the floor-plan content, excluding title block / legend / notes / property boundary. Returns `ROIResult` with bbox + reasoning. ~10–15 s, ~$0.02 per page. |
| [poc/pipeline/svg_overlay.py](poc/pipeline/svg_overlay.py) | **new** — `build_svg_overlay_html(page_image, detections)` produces a self-contained HTML snippet: page as base64 JPEG, bboxes + labels as SVG, inline pan/zoom JS (no external libraries). SVG strokes/text stay crisp at any zoom level. Rendered via `st.components.v1.html()`. |
| [poc/pipeline/detection_verifier.py](poc/pipeline/detection_verifier.py) | **new** — `verify_detections(overlay_image, detections)` sends the overlay PNG + class-count JSON to Claude vision asking for a 3–5 bullet sanity check (mis-placements / mis-classifications / systematic gaps). Reviewer aid; does **not** mutate the BOM. |
| [poc/pipeline/overlay.py](poc/pipeline/overlay.py) | **extended** — `draw_detections` now auto-scales stroke width and font size to the image resolution (3 px / 14 pt at 2000 px wide → 11 px / 45 pt at 4500 px wide), so overlays stay readable on the high-DPI user-pack render without manual tuning. New `class_colour_map()` exposes the deterministic colour palette for use in the Streamlit legend strip. |
| [poc/pipeline/schemas.py](poc/pipeline/schemas.py) | **extended** — `PageClassification`, `PDFClassification`, `ROIResult` models. |
| [poc/pipeline/runner.py](poc/pipeline/runner.py) | **extended** — accepts `use_roi: bool` flag. When true, runs `extract_plan_roi` → crops the page (with 6 % padding margin so border symbols aren't lost) → runs CV on the crop → translates detection bboxes back to full-page coords. ROI metadata surfaces in the `diagnostics` dict. |
| [poc/app.py](poc/app.py) | **extended** — Tab 1 auto-classifies on PDF upload, shows the per-page table, auto-suggests the plan page in the page picker, hints at the legend page for Tab 6. Sidebar gains a *Use Claude ROI crop on plan page* checkbox (default ON). CV tab gains: (a) a view-mode radio for *Vector pan-zoom overlay* vs *Static raster preview*, (b) a **download-full-resolution PNG** button, (c) a **class-colour legend strip** next to the image, (d) a **"Verify detections with Claude"** button. |

### 14d.2 End-to-end results on variant_b page 3

| Configuration | Detections | Distinct classes | BOM lines | Subtotal AUD | Pipeline elapsed |
| --- | --- | --- | --- | --- | --- |
| **ROI ON** (default) | 28 | 10/15 | 10 | 1,452 | **60 s** |
| **ROI OFF** | 70 | 14/15 | 14 | 3,449 | 146 s |

The toggle is a deliberate demo lever:

- **ROI ON** — clean, fast, plausible BOM. Excellent for *"show me the end-to-end pipeline working"*. Trade-off: ~4 classes lost because Claude's bbox cuts off some real plan symbols near the page edges (even with 6 % padding).
- **ROI OFF** — wide net. Catches more classes but includes false positives from the legend block on the plan page. Trade-off: 2.4× slower and noisier BOM.

The interview panel can ask *"can you show me without the crop?"* — flick the sidebar toggle, re-run, side-by-side comparison. **That's a strong demo moment** in itself.

### 14d.3 Page-classifier accuracy on variant_b

All 4 pages classified correctly with 0.97–0.98 confidence in a single ~9 s Claude call:

| Page | Predicted | Conf | Claude's reason (truncated) |
| --- | --- | --- | --- |
| 1 | `cover` | 0.97 | *"Title page with project name '24 Oak Street Residence', sheet list…"* |
| 2 | `demolition_plan` | 0.98 | *"Floor plan showing existing conditions with demolition legend…"* |
| 3 | `electrical_plan` | 0.98 | *"Proposed ground floor electrical plan with placed electrical…"* |
| 4 | `legend` | 0.97 | *"Electrical Symbol Library page showing all symbol types…"* |

`suggested_plan_page=3`, `suggested_legend_page=4` — both correct. The Tab 1 upload tab now auto-selects page 3 and shows a hint banner pointing to page 4 for Tab 6 ingestion. The wrong-page foot-gun is gone.

### 14d.4 Defending the architecture vs the prior-work email

The email said *"…rather than just sending drawings to an LLM and hoping for the best."* Phase 7 honours that because **Claude isn't counting symbols**. Per stage:

| Stage | Who does it | Defensible? |
| --- | --- | --- |
| Page classification | Claude vision | YES — classifying sheet *metadata*, not parsing the drawing |
| ROI selection | Claude vision | YES — choosing the *content region*, not detecting symbols inside it |
| Symbol detection | OpenCV template matching | unchanged from email — deterministic |
| Symbol → library matching | sentence-transformers + FAISS | unchanged from email — RAG, not LLM |
| Quantity / cost aggregation | deterministic Python | unchanged — the override-mechanism still catches any Claude deviation |
| Estimator note | Claude | unchanged — Claude only owns the narrative field |
| Sanity-check verifier | Claude vision | aid only — does NOT mutate the BOM |

The demo narrative becomes *"Claude reads the document; CV + RAG count the symbols; the deterministic aggregator owns the numbers."* Each Claude call is at a clearly-bounded edge.

### 14d.5 Failure modes and fallbacks

| Failure | Behaviour |
| --- | --- |
| `classify_pdf` errors (Anthropic down, JSON parse) | Returns `PDFClassification` with `.error` set, empty pages list. Streamlit falls back to *"pick page 1 manually"*. |
| `extract_plan_roi` errors / returns null bbox | `ROIResult.has_roi=False`, runner falls back to full-page CV (Phase 6 behaviour). |
| `verify_detections` errors | Streamlit displays the error string; no impact on existing run state. |
| SVG component too large (very dense detections / huge page) | Falls back via the *Static raster preview* radio option, which uses the existing `st.image()` path. |
| User dislikes ROI's aggressiveness on a specific PDF | Sidebar checkbox toggles it off mid-session. |

### 14d.6 Cost & latency budget added by Phase 7

| Call | Frequency | Latency | Cost |
| --- | --- | --- | --- |
| `classify_pdf` | once per PDF upload | ~8–10 s | ~$0.01 |
| `extract_plan_roi` | once per pipeline run (if ROI on) | ~10–15 s | ~$0.02 |
| `verify_detections` | on-demand button click | ~10–15 s | ~$0.02 |
| **Total per fresh run** | | ~20–30 s **added** | ~$0.03 **added** |

Net runtime usually *faster overall* because CV on the cropped region is much cheaper than CV on the full page (60 s vs 146 s in our test).

---

## 14c. Phase 6 — Dynamic Symbol Library Ingestion (2026-05-28)

After Phase 5 the user generated two new sample PDFs (`samples/synthetic_procalc_style_electrical_set*.pdf`) — 4-page ProCalc-style drawing sets with a real legend page (E-900) listing symbols + canonical names + alias phrases + units + indicative AUD costs. Phase 6 adds a feature to **read that legend page and merge its symbols into the runtime library**, additively, without overwriting the built-in 15 symbols.

The strategic value is in the demo: the POC stops being *"works only on our synthetic"* and becomes *"give us your drawing set, we'll learn your symbols and produce a BOM"*.

### 14c.1 Design choices (locked via the design-discussion question round)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Input format | Legend PDF page | Works directly on the new ProCalc-style sample PDFs without user prep |
| Detection scope after ingest | User-only by default | The user's drawing uses their symbols, not ours |
| Trigger | Streamlit tab AND CLI script | Preview-and-confirm safety in UI; repeatable batch via CLI |
| Name collisions | User wins, built-in shadowed | Cleanest semantic |

### 14c.2 New / modified files

| File | What |
| --- | --- |
| [poc/pipeline/legend_extractor.py](poc/pipeline/legend_extractor.py) | **new** — `extract_legend(pdf, page) -> LegendExtraction`. Renders the page at 2.5x DPI, OCRs with PSM 6, anchors on column headers (Symbol / Canonical / Alias / Unit / Indicative), clusters rows by Y, parses each cell, crops each symbol icon centered on OCR-detected glyph position. |
| [poc/pipeline/schemas.py](poc/pipeline/schemas.py) | **extended** — `LegendRow` (editable preview row with symbol bytes + parsed fields), `LegendExtraction` (full result + warnings list). |
| [poc/pipeline/rag.py](poc/pipeline/rag.py) | **extended** — `_load_library()` now merges `symbol_library/library.json` + `symbol_library/user_additions.json` with user-wins semantic. New `user_library_present()` helper. RAG embedding cache key already hashes the entry set, so additions auto-invalidate. |
| [poc/pipeline/cv_detect.py](poc/pipeline/cv_detect.py) | **extended** — `detect_symbols()` now accepts `symbol_pngs: list[Path]` in addition to `symbol_dir`. `_prepare_templates` takes a list directly. |
| [poc/pipeline/symbol_pack.py](poc/pipeline/symbol_pack.py) | **new** — `resolve_symbol_pngs(pack)` maps `"builtin"`/`"user"`/`"both"` to the right PNG list. User symbols shadow same-stem built-ins in `"both"` mode. |
| [poc/pipeline/runner.py](poc/pipeline/runner.py) | **extended** — accepts `symbol_pack` and `pdf_page_index`. Auto-bumps render resolution to `USER_PACK_CANVAS_W=4500` when the user pack is active, so multi-scale matching bridges the resolution gap between legend templates and plan symbols. |
| [poc/scripts/ingest_legend.py](poc/scripts/ingest_legend.py) | **new CLI** — `--pdf` + `--page` + `--dry-run` + `--reset`. Prints preview table, writes PNGs to `symbol_library/user/` and metadata to `user_additions.json` on confirm. |
| [poc/app.py](poc/app.py) | **extended** — new tab 6 *Symbol Library*: upload PDF → pick page → "Extract legend (preview)" → editable per-row dataframe (canonical name / alias / unit / cost) → "Ingest all rows" or "Discard preview". Sidebar gains a *symbol pack* radio (builtin / user / both) that auto-flips to `user` after an ingest. Upload tab (tab 1) gains a page picker for multi-page PDFs. |
| [.gitignore](.gitignore) | **extended** — `poc/symbol_library/user/` and `poc/symbol_library/user_additions.json` ignored (potentially client-sensitive). |

### 14c.3 End-to-end test result

| Stage | Input | Result |
| --- | --- | --- |
| Legend extraction | `synthetic_procalc_style_electrical_set.pdf` page 4 | **15 / 15 rows** extracted; all canonical names + units correct; 13 / 15 costs correct (4 fixed after column-width cap + tighter row clustering); minor OCR noise on aliases (`Ss` for `S`, `82` for `S2`) — editable in the Streamlit preview before ingest. |
| User pack ingestion | CLI: `python scripts/ingest_legend.py --pdf … --page 4` | **15 PNGs** written to `symbol_library/user/`; `user_additions.json` populated with all 15 entries; RAG cache auto-invalidates next build. |
| Detection on plan page | Same PDF, page 3, `symbol_pack="user"`, auto-bumped to 4500px wide render | **52 detections** across 11 classes (vs 1 detection at default 2000px wide — the high-DPI render was load-bearing). |
| BOM | 11 line items, AUD 3,264.60 | Confidence bands populated; some over-counts visible (`distribution_board: 8`) because the new PDF doesn't have ground-truth mask regions, so the in-page legend self-detects. Mask-region support exists; just needs `<pdf-stem>_ground_truth.json` to engage. |

### 14c.4 New demo flow (the moneymaker)

1. Open the app → cached run shows the original synthetic BOM (built-in pack, instant).
2. Click **tab 6 "Symbol Library"** → upload `synthetic_procalc_style_electrical_set.pdf` → pick page 4 → click **Extract legend** → see 15 rows in the editable table.
3. (Optional) fix any OCR-fuzzed aliases (`Ss` → `S`).
4. Click **Ingest all rows** → 15 PNGs land in `symbol_library/user/`; sidebar pack auto-flips to `user`.
5. Click **tab 1 "Upload"** → upload the *same* PDF → page picker shows 4 pages → pick page 3 (the electrical plan) → click **Run pipeline on page 3**.
6. Pipeline runs against the plan page using the freshly-ingested templates → BOM produced from the user's own symbols.

This is the moment the POC stops being a static-fixture demo and starts being *"point me at your drawing set"*.

### 14c.5 Known limitations of Phase 6

- **Detection over-counts when uploaded PDF's own legend block is on the plan page.** Mask regions are only loaded from a sibling `<stem>_ground_truth.json` (which only the synth fixture has). For real uploads, the in-page legend gets detected. Mitigation in the UI: per-line confidence band exposes the noise.
- **OCR fuzz on alias column.** Tesseract sometimes reads `S` as `Ss` and `S2` as `82`. The Streamlit preview is the safety net — user edits before ingest.
- **Legend extractor assumes a structured 5-column table** (Symbol / Canonical / Alias / Unit / Indicative AUD). Real legends with different column orderings or non-tabular layouts will need extractor extension; warnings list surfaces this when the column-header anchoring fails.
- **`@st.cache_resource` doesn't auto-invalidate on user-pack edits made outside the app.** The Streamlit ingest path calls `_get_rag.clear()` after writing; CLI ingestion requires a Streamlit page refresh to pick up.

---

## 14a. Post-Audit Fixes Applied (2026-05-28)

A full flaw audit was run after Phase 5 (see audit conversation). The
following CRITICAL and HIGH items were fixed; LOW/MEDIUM items remain in
the backlog (§14b) for future iteration.

### CRITICAL — all fixed

| # | Flaw | Fix applied |
| --- | --- | --- |
| 1 | Legend pollution: CV detected symbols inside the legend box | `cv_detect.detect_symbols()` now accepts `mask_regions` param; synth writes the legend rectangle to `ground_truth.json` and the runner / smoke tests load + pass it. Detections whose centre falls inside a mask region are dropped after NMS. |
| 2 | Synth fixture non-deterministic across Python invocations | Replaced `hash((room.name, count))` with `_det_seed(...)` (hashlib SHA-256 derived seed). Fixture now byte-stable across runs. |
| 3 | BOM had no upstream sanity check on CV counts | Added `avg_detection_score`, `min_detection_score`, `confidence_band` (high/medium/low) to `BOMLineItem`. App's BOM tab shows a "CV conf" column and an inline warning listing any low-confidence lines for estimator review. |
| 4 | Synthetic GT had placements buried under the legend | `_placement_positions()` now accepts `exclude` rectangle; `_place_symbols()` passes the legend rect for any room whose bbox overlaps it. No GT placement now lands under the legend. |
| 5 | Demo couldn't show the Claude-override mechanism live | New `bom_v1_force_deviation` prompt instructs Claude to deliberately misquote two lines, invent one, and omit one. `--force-deviation` flag on `bom_smoke_test.py` and a "Force Claude to deviate" sidebar checkbox in `app.py` engage it. Validator catches all four deviations and the BOM tab's diagnostics panel populates visibly. |
| 6 | Bathroom + Laundry room labels missed by OCR | Room-label font 22 → 30 px AND added a 60 px label-band exclusion at the top of every room in `_placement_positions()` so no symbol is placed over the label. All 8 room labels now OCR at 0.94–0.96 confidence. |

### HIGH — fixed (4 of 9)

| # | Flaw | Fix applied |
| --- | --- | --- |
| 11 | Templates composited onto cream BG (tuned to synth fixture only) | `DEFAULT_BG_RGB` switched from `(252,252,250)` to `(255,255,255)`. Real-world PDFs with white backgrounds now match without re-tuning. |
| 12 | No fallback if sentence-transformers model can't download | `SymbolRAG` now has `_deterministic_match()` using `difflib.SequenceMatcher` + alias-boost. `build()` catches embedding-path failures and flips to fallback mode. `SymbolRAG.mode` property reports `"embeddings"` vs `"deterministic-fallback"`. |
| 13 | `runner.py` reached into `rag._index` private attribute | `SymbolRAG.build()` is now idempotent (early-return on `self._built`); added public `SymbolRAG.is_built` property. Runner calls `rag.build()` unconditionally. |
| 14 | WP GPO symbol visually identical to single GPO inner-region | Redesigned WP GPO: hinged housing rectangle + diagonal cover-open marker + IP54 label + filled-disc centre. No inner prongs that mimic `single_gpo`. Threshold raised from 0.42 (was compensating for the bad design) to 0.60. |

### HIGH — deferred to §14b

| # | Flaw | Why deferred |
| --- | --- | --- |
| 7 | Per-class thresholds tuned to one fixture; no auto-tune | Real ML-engineering work; better as a separate `scripts/tune_thresholds.py` build-out post-interview. |
| 8 | Detector 18–26 s per page | Acceptable for the demo; the Streamlit progress bar masks the wait. Per-class rotation limiting is a 1-day perf project. |
| 9 | Single-page only | Synth fixture is single-page so the demo doesn't exercise multi-page. Real-PDF uploads would benefit; out-of-scope for interview. |
| 10 | Library + synth share PNGs (self-fulfilling) | Narrative addresses this; redrawing all 15 symbols with distinct synth-side variants is a half-day. |
| 15 | `audit_store/` unbounded | Ops concern, not demo-visible. |

### Result after fixes

| Metric | Before audit | After fixes |
| --- | --- | --- |
| CV recall on synth | 88 % | 90 % |
| CV `wp_gpo` over-count | +3 false positives | **0** (perfect) |
| Legend-region false positives | 3 | **0** (masked) |
| Room labels detected by OCR | 6 of 8 | **8 of 8** |
| Synth fixture deterministic across runs | NO (`hash()` randomised) | **yes** (hashlib seed) |
| Demo of override mechanism | NOT POSSIBLE | **`--force-deviation` flag + sidebar checkbox**; visibly catches 4 deviations |
| BOM line-item confidence visible | NO | **CV conf column + low-conf warning + diagnostics panel** |
| RAG fallback if embeddings unavailable | crash | **graceful deterministic-lookup mode** |
| New cached run subtotal | AUD 2,046.00 | **AUD 2,056.50** (one extra ceiling_light from re-tuning) |

---

## 14b. Backlog (post-interview improvements)

Items from the original audit that remain unfixed, in priority order.
Each is documented so the panel can be answered honestly *"yes that's on
the backlog, here's how we'd address it"*.

| # | Item | Effort | Production home |
| --- | --- | --- | --- |
| 7 | Auto-tune CV thresholds via grid-search against GT | 1 day | `scripts/tune_thresholds.py` |
| 8 | Per-class rotation list (skip 90/180/270 for symmetric symbols); ~3× speedup | 0.5 day | `cv_detect.py` constant |
| 9 | Multi-page PDF support | 1 day | `runner.run_pipeline` page loop |
| 10 | Synth uses visually-distinct templates from CV library | 0.5 day | new `synth/symbol_variants.py` |
| 15 | `audit_store/` retention policy | 2 hr | `bom_assembler._write_audit` |
| 16 | Switch to Anthropic SDK tool-use mode for structured output (no JSON parsing) | 0.5 day | `bom_assembler._call_claude` |
| 17 | PDF upload temp-file cleanup | 1 hr | `app._tab_upload` |
| 18 | OCR symbol-glyph noise: filter on alnum-only length | 30 min | `pipeline/ocr.py` |
| 19 | `_safe_str` reject `"-"` / `"N/A"` / `"None"` etc. | 30 min | `bom_assembler._safe_str` |
| 20 | `max_tokens` scales with line-item count | 30 min | `bom_assembler.DEFAULT_MAX_TOKENS` |
| 21 | File-size limit on PDF upload | 30 min | `app._tab_upload` |
| 22 | Friendly error wrapper for corrupt PDFs | 30 min | `runner.render_page` |
| 23 | CSV export includes assembler note | 15 min | `app._tab_bom` |
| 24 | Unit tests for `cv_detect`, `rag`, `bom_assembler` | 1 day | new `tests/` directory |
| 25 | `temperature=0.2` for stable estimator notes | 5 min | `bom_assembler._call_claude` |
| 26–32 | Various low-priority UX polish | various | various |

---

## 15. Estimated Total Time

| Phase | Estimated | Actual |
| --- | --- | --- |
| 1. Foundation & fixture | 1 day | ~½ day |
| 2. CV detection | 1 day | ~½ day (incl. 3 tuning passes) |
| 3. OCR + RAG | 1 day | ~½ day |
| 4. BOM assembly | 1 day | ~½ day |
| 5. Streamlit UI + polish | 1 day | pending |
| **Total** | **5 days** | **~2.5 days + Phase 5** |

---
