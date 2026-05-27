# ProCalc AI Measurement — Interview POC

A Streamlit demonstration of the five-step feature scoped in §2 of
`ProCalc_AI_Measurement_Response.docx`: ingest plan PDF → Claude vision
extraction into strict JSON with overlay coordinates → reviewer approval
gate → audit trail.

The POC targets the interview question:
*"Can the AI/ML lead demonstrate prior work involving architectural plans,
electrical takeoffs, construction drawings, or spatial/coordinate extraction?"*

## What it shows the panel

| Proposal section | Where it lives in the POC |
| --- | --- |
| §2 step 1 — PDF split | [pipeline/ingestion.py](pipeline/ingestion.py) |
| §2 step 2 — Claude agent returns strict JSON | [pipeline/extractor.py](pipeline/extractor.py), [pipeline/schemas.py](pipeline/schemas.py) |
| §2 step 3 — Calculator field population + overlay | [app.py](app.py), [pipeline/overlay.py](pipeline/overlay.py) |
| §2 step 4 — Approval gate before Calculate/Save | `_review_section` + `_calculate_section` in [app.py](app.py) |
| §2 step 5 — Audit store from request #1 | `_write_audit` in [pipeline/extractor.py](pipeline/extractor.py) |
| §3 — Prompts as versioned data | [pipeline/prompts.py](pipeline/prompts.py) |
| §7 — Three detection states (not collapsed) | `DetectionState` in [pipeline/schemas.py](pipeline/schemas.py) |
| §7 — Per-field state machine | `ApprovalState` + `ReviewableField` in [pipeline/schemas.py](pipeline/schemas.py) |
| §7 — Stable page identifiers | `PageRender.page_number` survives end-to-end |
| §7 — Retry / timeout / partial-result handling | `extract_from_page` in [pipeline/extractor.py](pipeline/extractor.py) |

## Setup

```powershell
cd c:\Users\Dell\Desktop\Project\Floor_Plan\poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

Drop a plan PDF into `..\samples\` (any residential plan set works; the
prompt is tuned for Ground Floor variables).

## Run

```powershell
streamlit run app.py
```

Then in the browser:

1. **Upload** the plan PDF.
2. **Pick** the Ground Floor page from the thumbnail strip.
3. **Run AI extraction**. The overlay panel renders Claude's bounding boxes
   on the page — solid green for `detected_actual`, dashed amber for
   `detected_placeholder`.
4. **Review** each field — `Sight on plan`, `Save correction`, or
   `Reject`. `Calculate` stays disabled until every detected field is
   resolved (the §2 step-4 hard block).
5. **Raw JSON** expander shows the exact §2 contract — open this on the
   interview call to show the structure.

## Variables extracted in this POC

Phase 1, Ground Floor only (proposal §4). Three were chosen so the JSON
contract gets exercised in three different shapes:

| Key | Shape | Why |
| --- | --- | --- |
| `ground_floor_area_m2` | Single measurement, one box | The Phase 1 headline variable |
| `room_count` | Integer + one box per named room | Demonstrates per-region extraction |
| `door_openings` | Count + per-door bounding boxes | Mirrors Techuz's prior electrical-symbol BOM experience |

Adding a fourth variable is a dict edit in
[pipeline/prompts.py](pipeline/prompts.py) — no code change, no deploy
(proposal §3).

## What this POC deliberately doesn't do

These are scoped out so the demo stays focused; each maps to scope in the
full proposal.

- Sheet classification (Phase 1, §3 "Plan segmentation").
- Multi-page reasoning across plan + schedule + specs (Phase 2, §5
  Legend & general-notes).
- Renovation existing-vs-new hatching reasoning (Phase 2, §5).
- Admin / feedback page UI on top of the audit store (Phase 1 §4.1
  "Admin feedback page").
- DPI / scale handling for ProCalc's native overlay format (Phase 1 §4.1
  "Coordinate translation layer" — POC works in rendered-image pixel
  space; production layer adds the transform).
