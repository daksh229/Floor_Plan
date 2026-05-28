# Electrical Layout → BOM — Implementation & Demo Guide

The pipeline-level docs (architecture, schemas, phases, design decisions)
live one level up in [implementation_plan.md](../implementation_plan.md).
This README covers **how to run the POC** and **how to demo it on the
interview call**.

---

## 1. Setup

### 1.1 Python venv

```powershell
cd c:\Users\Dell\Desktop\Project\Floor_Plan\poc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Open .env, paste your real ANTHROPIC_API_KEY after ANTHROPIC_API_KEY=
```

### 1.2 Tesseract binary (Windows)

Download `tesseract-ocr-w64-setup-v5.x.x.exe` from
https://github.com/UB-Mannheim/tesseract/wiki and install with defaults.
`pipeline/ocr.py` auto-locates `tesseract.exe` from `C:\Program Files\…`,
`C:\Program Files (x86)\…`, `%LOCALAPPDATA%\…`, or the `TESSERACT_CMD`
env var if set.

### 1.3 Generate the synthetic fixture (only needed once, or after
`library.json` edits)

```powershell
.\venv\Scripts\python.exe symbol_library\generate_symbols.py
.\venv\Scripts\python.exe synth\generate_sample.py
```

Outputs: `samples/synthetic_layout.pdf`, `samples/synthetic_layout_ground_truth.json`.

### 1.4 Capture one cached run for the Streamlit demo fallback

```powershell
.\venv\Scripts\python.exe scripts\capture_cached_run.py
```

Writes `poc/cached_runs/synthetic/{run.json, page.png, source.pdf}`.
The Streamlit app loads this on startup so the demo always opens to a
known-good state, even if Anthropic is down or the network is flaky.

---

## 2. Run the Streamlit demo

```powershell
streamlit run app.py
```

On first launch:
- The five tabs populate from the cached run instantly (no Claude call).
- Sidebar shows source = `cached: synthetic_layout.pdf`, stats, model,
  prompt version, and a "failure modes mitigated" checklist.

To re-run the live pipeline:
- **"Re-run on synthetic fixture"** (in tab 1) — runs CV + OCR + RAG +
  Claude against the synthetic PDF; takes ~70s.
- **Upload your own PDF** — runs the pipeline against any uploaded PDF.
  *Caveat: the symbol library is built for the synthetic fixture; on real
  drawings recall will be low.*

---

## 3. Individual smoke tests (CLI)

Each pipeline stage has a standalone smoke test for development /
debugging without launching Streamlit:

```powershell
.\venv\Scripts\python.exe scripts\cv_smoke_test.py     # CV detection + recall vs GT
.\venv\Scripts\python.exe scripts\ocr_smoke_test.py    # OCR span dump + overlay
.\venv\Scripts\python.exe scripts\rag_smoke_test.py    # RAG queries: exact / alias / free-text
.\venv\Scripts\python.exe scripts\bom_smoke_test.py    # full pipeline end-to-end
```

Outputs land in `samples/synthetic_layout_*.png` / `*_bom.json`.

---

## 4. Interview demo script

The whole arc is ~5 minutes. **Always open in cached mode** so the panels
are populated from the start.

### 4.1 Opening (30 sec)

> *"This POC backs up the prior-work claim in the earlier email — it's the
> exact five-stage pipeline I described to Richard: OCR + computer vision
> detection + RAG against a predefined symbol library + LLM-assembled BOM.
> The drawing is synthetic so we can show end-to-end against known ground
> truth, with deliberate scale/rotation/noise jitter so the matching isn't
> trivial."*

Switch to the browser. Show the **Upload tab** briefly to explain the
cached vs live mode.

### 4.2 CV Detections tab (60 sec)

Point to the page with bboxes.

> *"OpenCV multi-scale and multi-rotation template matching against the
> symbol library — five scales, four rotations, fifteen classes. After
> matching there's a per-class NMS to dedupe multi-scale hits of the same
> physical symbol, then a cross-class NMS so a `single_gpo` template
> doesn't claim a region that's actually a `wp_gpo` — the more-specific
> class wins via a specificity bonus."*

Point to the counts table.

> *"88% overall recall against ground truth on 60 placed symbols across
> 15 classes — twelve classes at 100%. The remaining gaps cluster on
> visually-ambiguous round-with-glyph shapes. Production fix is a
> fine-tuned YOLO, which is exactly Phase 2 in the proposal scope."*

### 4.3 OCR tab (30 sec)

> *"Tesseract sparse-text mode over the rendered page, with confidence
> and minimum-height filters to suppress symbol-glyph noise. Used
> downstream only for project metadata extraction — sheet number, project
> name, date — not for symbol detection."*

Show that the title block and most room labels came through.

### 4.4 RAG Matches tab (60 sec — the moment that sells this isn't a toy)

Expand the `wp_gpo` card.

> *"Real sentence-transformers embeddings over the alias-enriched library
> docs, served from a FAISS inner-product index — cosine similarity. The
> top-3 panel is the visible RAG mechanism: the GPO family is tightly
> contested because they share the prong glyph — `wp_gpo` 0.49, then
> `double_gpo` 0.43 and `single_gpo` 0.42 — while distinct items like
> `distribution_board` win cleanly. The BOM assembler commits to match
> #1 for each detected class."*

### 4.5 BOM tab (90 sec — the key technical moment)

Show the line items, then point to the estimator note.

> *"And this is the part we hardened deliberately. Claude is not the
> source of truth for the BOM. Quantities, specs, and costs all come
> deterministically from the CV detections plus library entries. Claude's
> job is narrowed to extracting project metadata from the OCR text and
> writing the estimator note you see here. After Claude responds we
> validate every line item against the deterministic truth and silently
> override any deviations. The BOM cannot have hallucinated values, by
> construction. This is the airtight answer to the obvious 'what if
> Claude makes up numbers' risk."*

Show the estimator note:

> *"And the note is what an estimator actually wants — Claude flagged
> that the 12-pole distribution board may be undersized given the six
> weatherproof GPOs, dedicated exhaust fan, and ceiling fans; that
> cabling and circuit breakers aren't included and need to be scoped
> separately; and that 6 weatherproof GPOs is high for a residential
> ground floor. That's where the model adds genuine value — not in
> reciting numbers it could get wrong."*

Open the **Raw BOM JSON** expander.

> *"And here's the strict JSON contract — every line traceable back to a
> library key and source pages, which is what makes it commercial-grade."*

Click the **Download BOM as CSV** button briefly.

### 4.6 Closing (30 sec — lead with the honesty, don't wait to be asked)

> *"To be candid about the limits: the synthetic PDF is generated from
> the same templates the detector uses, with jitter applied so it's not
> a perfectly self-fulfilling demo. On arbitrary real drawings the
> template-matching layer would need to be extended — that's exactly
> the scope sitting in Phase 2 of the proposal, where we'd train a
> small YOLO on a few hundred labelled plans. The pipeline shape and the
> Claude-as-editorialiser architecture is what carries forward."*

### 4.7 If anything goes wrong on the live call

- The Streamlit app **boots in cached mode by default**, so even if the
  network is dead, all five panels populate from disk.
- If you accidentally trigger a live run and Claude is slow, the status
  box shows each stage with elapsed time so the panel isn't staring at a
  blank spinner.
- If the live run errors mid-pipeline, click **"Reload cached run"** in
  the Upload tab to get back to the known-good demo state instantly.

---

## 5. File map (quick reference)

| File | Phase | Purpose |
| --- | --- | --- |
| [pipeline/ingestion.py](pipeline/ingestion.py) | 1 | PDF → page-numbered PIL images |
| [pipeline/cv_detect.py](pipeline/cv_detect.py) | 2 | Multi-scale + multi-rotation template matching, NMS within + across class |
| [pipeline/ocr.py](pipeline/ocr.py) | 3 | Tesseract pass with auto-located binary + filters |
| [pipeline/rag.py](pipeline/rag.py) | 3 | `SymbolRAG` class: alias-enriched embedding + FAISS index + cache |
| [pipeline/prompts.py](pipeline/prompts.py) | 4 | Versioned BOM-assembly prompt store |
| [pipeline/bom_assembler.py](pipeline/bom_assembler.py) | 4 | Deterministic aggregation + Claude as editorialiser + validation |
| [pipeline/runner.py](pipeline/runner.py) | 5 | End-to-end orchestration + cached-run save/load |
| [pipeline/overlay.py](pipeline/overlay.py) | 2, 3 | Detection + OCR bbox renderers |
| [pipeline/schemas.py](pipeline/schemas.py) | all | All Pydantic models |
| [symbol_library/generate_symbols.py](symbol_library/generate_symbols.py) | 1 | Produces the 15 symbol PNGs |
| [symbol_library/library.json](symbol_library/library.json) | 1 | Canonical metadata + aliases + cost per symbol |
| [synth/generate_sample.py](synth/generate_sample.py) | 1 | Synthesises the multi-room electrical layout PDF |
| [app.py](app.py) | 5 | Streamlit entry point — 5 visible panels |
| [scripts/cv_smoke_test.py](scripts/cv_smoke_test.py) | 2 | CLI: detection + recall vs ground truth |
| [scripts/ocr_smoke_test.py](scripts/ocr_smoke_test.py) | 3 | CLI: OCR span dump + overlay |
| [scripts/rag_smoke_test.py](scripts/rag_smoke_test.py) | 3 | CLI: RAG queries (exact / alias / free-text) |
| [scripts/bom_smoke_test.py](scripts/bom_smoke_test.py) | 4 | CLI: full pipeline end-to-end |
| [scripts/capture_cached_run.py](scripts/capture_cached_run.py) | 5 | One-shot: writes the demo fallback to `cached_runs/synthetic/` |

---

## 6. Known limitations & honest disclaimers

- **Synthetic data only.** The library and the test PDF share a visual
  vocabulary by design (so the pipeline is end-to-end demoable). Real
  drawings will see degraded CV recall.
- **Single-page only.** The synth fixture is one page; multi-page support
  would be a trivial loop in `runner.py` but isn't built.
- **Symbol-glyph noise leaks through OCR filters** as 2–3 character
  artefacts like `(2)`, `fs)`. Harmless — only the BOM assembler reads
  OCR text and it only uses high-confidence prose for metadata.
- **`Bathroom` and `Laundry` room labels are not detected** by Tesseract
  on the synthetic fixture (small font, crowded by symbols). Production
  fix: higher-DPI render or region-of-interest OCR seeded by room
  polygons.
- **Embedding model first-run download (~80 MB).** `pre-warmed in the
  capture step; cached in `~/.cache/huggingface/`.
