# Electrical Layout → BOM POC

Interview-evidence POC for the **ProCalc** engagement (residential construction
estimating, Australia). The POC backs up the prior-work claim in Techuz's
response email to Richard:

> _"A BOM (Bill of Materials) generation system for electrical construction
> layouts… OCR + computer vision pipeline processes the drawings to detect and
> locate all symbols, annotations, and elements on the blueprint. Symbol
> recognition via RAG — each detected element is matched against a predefined
> symbol library… AI-powered BOM generation — all extracted elements,
> quantities, and specifications are passed to the LLM, which assembles a
> structured Bill of Materials… rather than just sending drawings to an LLM
> and hoping for the best."_

The POC demonstrates that exact pipeline end-to-end, in pure Python with a
Streamlit UI. Each stage is a **separately visible panel** — the
email-promised architecture is observable, not collapsed behind a single LLM
call.

## Pipeline (mirrors the email)

| # | Stage | Technology |
| --- | --- | --- |
| 1 | Upload electrical layout PDF | `pypdfium2` |
| 2 | OCR + computer vision symbol detection | `pytesseract` + OpenCV multi-scale / multi-rotation template matching |
| 3 | RAG symbol → library matching | `sentence-transformers` + FAISS over a predefined symbol library |
| 4 | LLM-assembled structured BOM | Claude (Anthropic SDK) — assembly only, **no** count/spec invention |
| 5 | BOM output for vendor quotation | Streamlit table + strict JSON + CSV download |

## Why this POC and not a forward-looking ProCalc demo

Richard's Q2 to Vaibhav was:
> _"Can the AI/ML lead demonstrate prior work involving architectural plans,
> electrical takeoffs, construction drawings, or spatial/coordinate
> extraction?"_

The earlier email described an electrical-BOM pipeline as Techuz's prior work.
The POC is a working artifact of that exact claim, ready to demo on the final
interview.

The full ProCalc proposal sits separately at
[ProCalc_AI_Measurement_Response .docx](ProCalc_AI_Measurement_Response%20.docx)
and remains the contract for the actual engagement.

## Repository layout

```
Floor_Plan/
  ProCalc_AI_Measurement_Response .docx   # proposal context (reference only)
  README.md                                # you are here
  .gitignore
  samples/                                 # input PDFs (gitignored — drop locally)
  poc/
    app.py                                 # Streamlit entry point
    pipeline/                              # ingestion, cv_detect, ocr, rag, bom_assembler, overlay, schemas, prompts
    symbol_library/                        # ~15 symbol PNGs + library.json
    synth/                                 # generate_sample.py — synthesises a non-trivial electrical layout PDF
    audit_store/                           # per-request audit trail (gitignored)
    requirements.txt
    .env.example                           # copy to .env, fill ANTHROPIC_API_KEY
    README.md                              # implementation details + interview demo script (rewritten in Phase 5)
```

## Build phases

| Phase | Deliverables | Visible end state |
| --- | --- | --- |
| **1. Foundation & fixture** | symbol library (PNGs + `library.json` with aliases & specs); `synth/generate_sample.py` → `samples/synthetic_layout.pdf` with scale/rotation/noise jitter; updated `requirements.txt`; Tesseract verified on Windows | Realistic synthetic electrical drawing on disk |
| **2. CV symbol detection** | `pipeline/cv_detect.py` (multi-scale + multi-rotation template matching, NMS); `Detection` schema; overlay reused | CLI prints per-class counts + saves overlay PNG |
| **3. OCR + RAG** | `pipeline/ocr.py` (pytesseract + single-char filter); `pipeline/rag.py` (sentence-transformers + FAISS over alias-enriched library) | Two CLI demos: OCR text dump, RAG top-3 lookup |
| **4. BOM assembly** | versioned `pipeline/prompts.py`; `pipeline/bom_assembler.py` (Claude call with strict no-invention constraints, retry/timeout/audit); final `BOM` schema with traceability back to detections + library entries | Structured BOM JSON on disk, every line traceable |
| **5. Streamlit UI + polish** | `app.py` rewritten with 5 visible panels (Upload → CV → OCR → RAG → BOM); cached-result fallback button for demo safety; CSV download; `poc/README.md` rewritten with interview demo script | Full interview-ready demo with safety net |

Each phase ends in something demoable. If we get squeezed for time, stopping
after Phase 4 still produces a complete CLI-driven pipeline.

## Honesty stance on the synthetic data

The sample PDF is synthesised from the same symbol library used for detection.
This is **disclosed in the demo**, not hidden: the synthetic PDF exists to
prove the pipeline shape end-to-end; on arbitrary real drawings the
template-matching layer would be extended (additional scales, fine-tuned YOLO,
or vector-PDF symbol extraction) — that scope sits in the proposal's Phase 2,
not in the POC.

This pre-empts the predictable interview question ("does it work on a real
PDF?") with a candid answer instead of a hand-wave.

## Setup

```powershell
cd poc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY
```

Then:

```powershell
streamlit run app.py
```

Full implementation notes and the on-screen demo script live in
[poc/README.md](poc/README.md) (gets rewritten in Phase 5; the current copy
documents the superseded measurement POC).

## Related context

- **Proposal:** [ProCalc_AI_Measurement_Response .docx](ProCalc_AI_Measurement_Response%20.docx) — what Techuz will build for ProCalc if the engagement is won.
- **Prior email referenced:** the electrical-BOM description Vaibhav sent Richard (in the client thread); this POC is the working artifact of that description.
