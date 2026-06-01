"""Electrical Layout -> BOM POC — Streamlit demo.

Five-stage pipeline visible across five tabs:
  1. Upload     — pick the cached run or upload a PDF
  2. CV         — OpenCV multi-scale + multi-rotation template matching
  3. OCR        — Tesseract --psm 11 sparse text
  4. RAG        — sentence-transformers + FAISS over alias-enriched library
  5. BOM        — deterministic aggregation + Claude as editorialiser

The app starts in cached mode so the demo always opens to a known-good
state, even if Anthropic is down or the network is flaky.
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

# Imports kept after load_dotenv so the Anthropic SDK sees ANTHROPIC_API_KEY
# at module-import time on first request.
from pipeline.bom_assembler import DEFAULT_MODEL                # noqa: E402
from pipeline.detection_verifier import verify_detections       # noqa: E402
from pipeline.legend_extractor import extract_legend            # noqa: E402
from pipeline.overlay import (                                  # noqa: E402
    draw_detections,
    draw_ocr_spans,
)
from pipeline.page_classifier import classify_pdf               # noqa: E402
from pipeline.prompts import DEFAULT_VERSION                    # noqa: E402
from pipeline.svg_overlay import build_svg_overlay_html         # noqa: E402
from pipeline.rag import (                                      # noqa: E402
    SymbolRAG,
    USER_ADDITIONS_PATH,
    USER_SYMBOL_DIR,
    user_library_present,
)
from pipeline.runner import (                                   # noqa: E402
    load_pipeline_run,
    pdf_page_count,
    run_pipeline,
    save_pipeline_run,
)
from pipeline.schemas import PipelineRun                        # noqa: E402
from pipeline.symbol_pack import SymbolPack, user_pack_size     # noqa: E402


PROJECT_ROOT = HERE.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
SYMBOL_DIR = HERE / "symbol_library"
CACHE_DIR_SYNTH = HERE / "cached_runs" / "synthetic"
AUDIT_DIR = HERE / "audit_store"


# ---------- helpers ----------

@st.cache_resource(show_spinner="Loading sentence-transformers embedding model (first run only)...")
def _get_rag() -> SymbolRAG:
    """Load the embedding model + FAISS index once across reruns."""
    rag = SymbolRAG()
    rag.build()
    return rag


def _init_state() -> None:
    # Streamlit forbids writing to a widget-bound session_state key AFTER
    # that widget has rendered in the same run. So when an ingest / clear
    # handler wants to switch the symbol pack, it parks its intent in
    # `_pending_symbol_pack` and triggers a rerun. Here, at the very top of
    # the next run (before _sidebar() instantiates the radio widget), we
    # consume the pending value and write it to the bound key.
    pending = st.session_state.pop("_pending_symbol_pack", None)
    if pending is not None:
        st.session_state["symbol_pack"] = pending

    st.session_state.setdefault("run", None)            # PipelineRun
    st.session_state.setdefault("page_image", None)     # PIL.Image
    st.session_state.setdefault("source_label", None)   # str shown in sidebar
    st.session_state.setdefault("diagnostics", None)    # dict from latest run
    st.session_state.setdefault("force_deviation", False)

    # Phase 6 — symbol-library ingestion
    st.session_state.setdefault("legend_extraction", None)  # LegendExtraction or None
    st.session_state.setdefault("legend_pdf_path", None)    # tempfile path
    st.session_state.setdefault("legend_pdf_name", None)
    # default symbol pack: "user" if a user pack is present, else "builtin"
    default_pack = "user" if user_library_present() else "builtin"
    st.session_state.setdefault("symbol_pack", default_pack)
    st.session_state.setdefault("pdf_page_index", 1)

    # Phase 7 — Claude page classifier + ROI extractor
    st.session_state.setdefault("page_classification", None)  # PDFClassification keyed by upload
    st.session_state.setdefault("page_classification_for", None)  # which file we classified
    st.session_state.setdefault("use_roi", True)  # opt-in Claude ROI crop, default ON
    # CV backend: "template" = multi-scale template matching (cv_detect),
    # "yolo" = trained YOLOv8 ONNX (yolo_detect). Auto-selects yolo when
    # best.onnx is present in project root.
    from pathlib import Path as _P
    _default_backend = "yolo" if (_P(__file__).resolve().parent.parent / "best.onnx").exists() else "template"
    st.session_state.setdefault("cv_backend", _default_backend)
    # Claude vision fallback for room extraction — kicks in when OCR
    # finds fewer than 5 rooms. Default ON since it materially improves
    # the by-room BOM quality on plans where Tesseract misses labels.
    st.session_state.setdefault("enable_claude_room_fallback", True)
    st.session_state.setdefault("cv_verify_findings", None)
    st.session_state.setdefault("cv_verify_error", None)

    # Phase 8 — wizard state machine
    # 1=upload, 2=pack, 3=legend (skipped for builtin), 4=run, 5=dashboard
    st.session_state.setdefault("wizard_step", 1)
    st.session_state.setdefault("wizard_pdf_path", None)        # tempfile.Path of uploaded PDF
    st.session_state.setdefault("wizard_pdf_name", None)
    st.session_state.setdefault("wizard_n_pages", 0)
    st.session_state.setdefault("wizard_pack_choice", None)     # "builtin" | "user"
    st.session_state.setdefault("wizard_legend_ingested", False)


def _load_cached_into_state() -> bool:
    if not (CACHE_DIR_SYNTH / "run.json").exists():
        return False
    run, page = load_pipeline_run(CACHE_DIR_SYNTH)
    st.session_state["run"] = run
    st.session_state["page_image"] = page
    st.session_state["source_label"] = f"cached: {run.source_pdf}"
    st.session_state["diagnostics"] = None
    return True


def _run_against_pdf(pdf_path: Path, source_label: str) -> None:
    """Live pipeline run with a Streamlit status bar."""
    rag = _get_rag()
    progress_box = st.status("Running pipeline...", expanded=True)
    stage_log: list[tuple[str, float, float]] = []
    t0 = time.perf_counter()
    force_dev = bool(st.session_state.get("force_deviation", False))
    pack: SymbolPack = st.session_state.get("symbol_pack", "builtin")  # type: ignore[assignment]
    page_index = int(st.session_state.get("pdf_page_index", 1))
    use_roi = bool(st.session_state.get("use_roi", True))
    cv_backend = st.session_state.get("cv_backend", "template")
    enable_claude_room_fallback = bool(
        st.session_state.get("enable_claude_room_fallback", True)
    )

    def _progress(stage: str, frac: float) -> None:
        elapsed = time.perf_counter() - t0
        stage_log.append((stage, frac, elapsed))
        progress_box.update(label=f"{stage} — {elapsed:.1f}s", state="running")
        with progress_box:
            st.write(f"[{frac:>5.0%}]  {stage}  (t+{elapsed:.1f}s)")

    try:
        run, page, diagnostics = run_pipeline(
            pdf_path,
            rag=rag,
            audit_dir=AUDIT_DIR,
            progress=_progress,
            force_demo_deviation=force_dev,
            symbol_pack=pack,
            pdf_page_index=page_index,
            use_roi=use_roi,
            cv_backend=cv_backend,
            enable_claude_room_fallback=enable_claude_room_fallback,
        )
    except Exception as exc:  # noqa: BLE001
        progress_box.update(label=f"Pipeline error: {exc}", state="error")
        st.exception(exc)
        return
    progress_box.update(
        label=f"Pipeline complete in {run.elapsed_seconds}s"
              + (f"  ·  pack: {pack}  ·  page: {page_index}")
              + (" (DEMO MODE: override active)" if force_dev else ""),
        state="complete",
        expanded=False,
    )
    st.session_state["run"] = run
    st.session_state["page_image"] = page
    label_extras = []
    if pack != "builtin":
        label_extras.append(f"pack={pack}")
    if page_index != 1:
        label_extras.append(f"p{page_index}")
    if cv_backend != "template":
        label_extras.append(f"cv={cv_backend}")
    if force_dev:
        label_extras.append("demo-mode")
    suffix = (" [" + ", ".join(label_extras) + "]") if label_extras else ""
    st.session_state["source_label"] = source_label + suffix
    st.session_state["diagnostics"] = diagnostics


def _check_api_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        st.error(
            "ANTHROPIC_API_KEY is not set in `poc/.env`. The cached run still "
            "works, but a fresh pipeline run will fail."
        )
        return False
    return True


# ---------- tabs ----------

def _tab_upload(api_key_ok: bool) -> None:
    st.subheader("1. Upload the electrical layout")
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("**Demo defaults to the cached run** so the panels are populated immediately. Use the controls below to either re-run against the synthetic fixture or supply your own PDF.")
        if st.button("Re-run on synthetic fixture", use_container_width=True, disabled=not api_key_ok):
            pdf = CACHE_DIR_SYNTH / "source.pdf"
            if not pdf.exists():
                pdf = SAMPLES_DIR / "synthetic_layout.pdf"
            st.session_state["pdf_page_index"] = 1
            _run_against_pdf(pdf, source_label=f"live: {pdf.name}")
            st.rerun()
        if st.button("Reload cached run", use_container_width=True):
            if _load_cached_into_state():
                st.success("Cached run reloaded.")
            else:
                st.error("No cached run available at " + str(CACHE_DIR_SYNTH))
        st.caption(
            "Cached runs are saved by "
            "`scripts/capture_cached_run.py` (run once after pipeline changes)."
        )

    with col1:
        uploaded = st.file_uploader(
            "Or upload your own electrical layout PDF",
            type=["pdf"],
            help="If the PDF has a legend page, use tab 6 'Symbol Library' to "
                 "ingest it first, then come back here and pick the plan page.",
        )
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            try:
                n_pages = pdf_page_count(tmp_path)
            except Exception:  # noqa: BLE001
                n_pages = 1

            # Auto-classify the pages on first encounter of this upload
            pc = st.session_state.get("page_classification")
            if api_key_ok and (
                pc is None or st.session_state.get("page_classification_for") != uploaded.name
            ):
                with st.spinner(f"Classifying {n_pages} page(s) with Claude vision..."):
                    pc = classify_pdf(tmp_path)
                st.session_state["page_classification"] = pc
                st.session_state["page_classification_for"] = uploaded.name

            # Show what Claude thinks
            default_page = 1
            if pc is not None and not pc.error and pc.pages:
                st.markdown("**Detected sheets:**")
                rows = [
                    {
                        "page": p.page_index,
                        "sheet_class": p.sheet_class,
                        "conf": p.confidence,
                        "reason": p.reasoning,
                    }
                    for p in sorted(pc.pages, key=lambda p: p.page_index)
                ]
                st.dataframe(rows, hide_index=True, use_container_width=True)
                if pc.suggested_plan_page:
                    default_page = pc.suggested_plan_page
                    st.success(
                        f"Suggested plan page: **{default_page}** "
                        "(auto-selected below; change if you disagree)"
                    )
                if pc.suggested_legend_page:
                    st.info(
                        f"Suggested legend page: **{pc.suggested_legend_page}** "
                        "(ingest it on Tab 6 'Symbol Library')"
                    )
            elif pc is not None and pc.error:
                st.warning(f"Page classifier failed: {pc.error} — picking page 1 by default.")

            page_index = default_page
            if n_pages > 1:
                page_index = st.number_input(
                    f"Page to process (this PDF has {n_pages} pages)",
                    min_value=1, max_value=n_pages,
                    value=min(default_page, n_pages), step=1,
                )
                st.session_state["pdf_page_index"] = int(page_index)

            run_btn = st.button(
                f"Run pipeline on page {page_index}",
                type="primary",
                disabled=not api_key_ok,
            )
            if run_btn:
                if not api_key_ok:
                    st.error("Need ANTHROPIC_API_KEY to run the pipeline.")
                else:
                    st.session_state["pdf_page_index"] = int(page_index)
                    _run_against_pdf(tmp_path, source_label=f"live: {uploaded.name}")
                    st.rerun()


def _tab_cv() -> None:
    st.subheader("2. CV symbol detection")
    run: PipelineRun = st.session_state["run"]
    page: Image.Image = st.session_state["page_image"]

    st.caption(
        "OpenCV `cv2.matchTemplate` (TM_CCOEFF_NORMED) across multi-scale × "
        "4 rotations × N symbol classes. Class-wise NMS dedupes multi-scale "
        "duplicates; cross-class NMS with a specificity bonus resolves "
        "structural collisions (e.g. `single_gpo` inside `wp_gpo`)."
    )

    overlaid = draw_detections(page, run.detections)
    # Cache the rendered overlay bytes on the session so the download button
    # doesn't re-encode on every rerun.
    buf = io.BytesIO()
    overlaid.save(buf, format="PNG", optimize=False)
    overlay_bytes = buf.getvalue()

    view_mode = st.radio(
        "Overlay view",
        options=["Vector (pan/zoom, crisp at any zoom)", "Static raster preview"],
        index=0,
        horizontal=True,
        key="cv_view_mode",
    )

    left, right = st.columns([2, 1])
    with left:
        if view_mode.startswith("Vector"):
            html = build_svg_overlay_html(page, run.detections, component_height_px=720)
            st.components.v1.html(html, height=820, scrolling=False)
            st.caption(
                f"{len(run.detections)} detections · drag to pan · scroll to "
                "zoom · double-click to reset · SVG bboxes stay sharp at any zoom."
            )
        else:
            st.image(
                overlaid,
                caption=f"{len(run.detections)} detections · page {run.detections[0].source_page if run.detections else '-'}",
                use_container_width=True,
            )
        st.download_button(
            label=f"\U0001F4E5 Download full-resolution overlay PNG ({len(overlay_bytes)//1024} KB)",
            data=overlay_bytes,
            file_name=f"detections_{run.run_id}_p{run.detections[0].source_page if run.detections else 1}.png",
            mime="image/png",
            help="Right-click → Save, or click to download. Native image viewers "
                 "handle zoom much better than the in-browser preview.",
        )
    with right:
        # NOTE: the per-class colour legend that used to live here has moved
        # *into* the vector-overlay component, where it's interactive
        # (hover-to-isolate / click-to-lock). The detection-details table
        # below still lives here as the sortable data-view.
        st.caption(
            "Tip: the colour legend now sits beside the plan — hover a class to "
            "isolate it, click to lock, Esc to clear."
        )

        with st.expander("Detection details (sortable)", expanded=False):
            st.dataframe(
                [
                    {
                        "class": d.symbol_class,
                        "score": d.score,
                        "rot": d.rotation_deg,
                        "scale": d.scale,
                        "bbox": str(d.bbox),
                    }
                    for d in run.detections
                ],
                hide_index=True,
                use_container_width=True,
                height=400,
            )

        # ROI / padding diagnostics, if applicable
        diagnostics = st.session_state.get("diagnostics") or {}
        roi_info = diagnostics.get("roi") or {}
        pad_info = diagnostics.get("roi_padding") or {}
        if roi_info.get("bbox") and pad_info.get("pad_pct") is not None:
            with st.expander(
                f"\U0001F50D  Claude ROI crop · auto-pad {pad_info['pad_pct']*100:.1f}%"
            ):
                st.write(f"**Claude's bbox:** `{roi_info['bbox']}`  "
                         f"(conf {roi_info['confidence']:.2f})")
                st.write(f"**Padded bbox:** `{pad_info.get('padded_bbox')}`  "
                         f"(+{pad_info['pad_x_px']} px x / +{pad_info['pad_y_px']} px y)")
                st.caption(
                    "Padding is dynamic: scales with Claude's confidence and "
                    "how tight the bbox is vs. the page. See "
                    "`pipeline/runner._compute_roi_padding_pct()`."
                )
                if roi_info.get("reasoning"):
                    st.caption(f"_Claude's reasoning:_ {roi_info['reasoning']}")

        st.divider()
        st.markdown("**Claude vision sanity-check**")
        st.caption(
            "Send the overlay + class counts to Claude vision. Returns 3-5 "
            "bullet points flagging obvious mis-placements, mis-classifications "
            "or systematic gaps. Reviewer aid only — does NOT mutate the BOM."
        )
        if st.button("Verify detections with Claude", key="cv_verify_btn",
                     use_container_width=True):
            with st.spinner("Claude reviewing the overlay (~15s)..."):
                findings, err = verify_detections(overlaid, run.detections)
            st.session_state["cv_verify_findings"] = findings
            st.session_state["cv_verify_error"] = err

        findings = st.session_state.get("cv_verify_findings")
        err = st.session_state.get("cv_verify_error")
        if err:
            st.error(f"Verifier error: {err}")
        elif findings:
            st.info(findings)


def _tab_ocr() -> None:
    st.subheader("3. OCR")
    run: PipelineRun = st.session_state["run"]
    page: Image.Image = st.session_state["page_image"]

    st.caption(
        "Tesseract `--psm 11` (sparse text) with min-confidence, "
        "min-length and min-height filters to suppress symbol-glyph noise. "
        "Used downstream by the BOM assembler for project metadata only."
    )

    if not run.ocr_spans:
        st.warning("No OCR spans captured. Tesseract may not be installed on "
                   "this machine — install from UB-Mannheim and restart.")
        return

    left, right = st.columns([2, 1])
    with left:
        overlaid = draw_ocr_spans(page, run.ocr_spans)
        st.image(overlaid, caption=f"{len(run.ocr_spans)} OCR spans (blue)",
                 use_container_width=True)
    with right:
        st.markdown("**Text spans (top → bottom)**")
        sorted_spans = sorted(run.ocr_spans, key=lambda s: (s.bbox[1], s.bbox[0]))
        st.dataframe(
            [{"text": s.text, "conf": s.confidence, "bbox": str(s.bbox)} for s in sorted_spans],
            hide_index=True,
            use_container_width=True,
            height=600,
        )


def _tab_rag() -> None:
    st.subheader("4. RAG: symbol → library matching")
    run: PipelineRun = st.session_state["run"]

    st.caption(
        "sentence-transformers (`all-MiniLM-L6-v2`) over the alias-enriched "
        "library document for each symbol; FAISS `IndexFlatIP` over "
        "L2-normalised vectors = cosine similarity. The top-3 panel below is "
        "what makes the retrieval mechanism *visible* on screen — the BOM "
        "assembler commits to match #1 for each detected class."
    )

    # Group by class — pick one canonical SymbolWithMatches per class (matches
    # are identical within a class because the query is the class name).
    by_class: dict[str, list] = {}
    for sm in run.symbol_matches:
        by_class.setdefault(sm.detection.symbol_class, []).append(sm)

    for cls in sorted(by_class):
        sm_list = by_class[cls]
        any_sm = sm_list[0]
        with st.expander(
            f"**{cls}**  —  {len(sm_list)} detection(s)  →  "
            f"best match: `{any_sm.best.library_key}`  "
            f"(sim {any_sm.best.similarity:+.3f})"
            if any_sm.best else f"**{cls}**  (no matches)",
            expanded=False,
        ):
            for rank, m in enumerate(any_sm.matches, start=1):
                cols = st.columns([1, 4, 2])
                cols[0].markdown(f"**#{rank}**")
                cols[1].markdown(
                    f"**{m.canonical_name}**  \n"
                    f"`{m.library_key}`  ·  unit `{m.unit}`  ·  AUD {m.indicative_cost_aud:.2f}  \n"
                    f"*{m.spec}*"
                )
                cols[2].progress(
                    max(0.0, min(1.0, (m.similarity + 1) / 2)),
                    text=f"sim {m.similarity:+.3f}",
                )


def _tab_symbol_library() -> None:
    st.subheader("6. Symbol Library — ingest a legend page (Phase 6)")
    st.caption(
        "Drop a drawing PDF that contains a legend page. The system "
        "renders that page at high DPI, locates the table columns, "
        "crops every symbol icon, and reads the canonical name + alias + "
        "unit + indicative cost from the table. Review and edit the "
        "extracted rows before clicking **Ingest** — once ingested, the "
        "symbols are merged into `symbol_library/user_additions.json` and "
        "available to the CV detector when *symbol pack: user* is selected "
        "in the sidebar. Built-in symbols are never overwritten."
    )

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("**Current user pack**")
        if user_library_present():
            n = user_pack_size()
            st.metric("User symbols ingested", n)
            st.code(f"file: {USER_ADDITIONS_PATH.name}\npngs: {USER_SYMBOL_DIR.name}/", language="text")
            if st.button("🗑️  Clear all user symbols", use_container_width=True):
                _reset_user_pack()
                _get_rag.clear()
                # Defer the pack switch to next rerun — see _init_state() for why.
                st.session_state["_pending_symbol_pack"] = "builtin"
                st.success("User pack cleared. Sidebar pack reverted to 'builtin'.")
                st.rerun()
        else:
            st.info("No user symbols ingested yet. Built-in library (15 symbols) is active.")

    with col1:
        uploaded = st.file_uploader(
            "Upload PDF containing the legend page",
            type=["pdf"],
            key="legend_upload",
        )
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)
            try:
                n_pages = pdf_page_count(tmp_path)
            except Exception:  # noqa: BLE001
                n_pages = 1
            page_pick = st.number_input(
                f"Legend page (this PDF has {n_pages} pages)",
                min_value=1, max_value=max(n_pages, 1), value=n_pages, step=1,
                help="Most ProCalc-style drawing sets put the legend on the LAST page.",
            )
            if st.button("Extract legend (preview)", type="primary"):
                with st.spinner("OCR + crop legend rows..."):
                    try:
                        extraction = extract_legend(tmp_path, int(page_pick))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Extraction error: {exc}")
                        return
                st.session_state["legend_extraction"] = extraction
                st.session_state["legend_pdf_path"] = tmp_path
                st.session_state["legend_pdf_name"] = uploaded.name
                st.rerun()

    extraction = st.session_state.get("legend_extraction")
    if extraction is None:
        return

    st.divider()
    st.markdown(f"### Preview — {len(extraction.rows)} row(s) from `{st.session_state['legend_pdf_name']}` page {extraction.source_page}")
    if extraction.warnings:
        with st.expander(f"⚠️ {len(extraction.warnings)} warning(s) from extraction"):
            for w in extraction.warnings:
                st.write(f"- {w}")

    # Show each row with its cropped icon thumbnail + editable text inputs
    st.caption(
        "Edit any cell before ingesting — OCR fuzz on aliases ('Ss' for 'S', "
        "'82' for 'S2') is common. The library_key is auto-derived from the "
        "canonical name."
    )
    edited_rows: list[dict] = []
    for i, row in enumerate(extraction.rows):
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1, 3, 3, 1, 1])
            with c1:
                img_bytes = base64.b64decode(row.symbol_image_b64)
                st.image(img_bytes, width=64)
            with c2:
                canonical = st.text_input("Canonical name", value=row.canonical_name, key=f"can_{i}")
            with c3:
                alias = st.text_input("Alias / spec phrase", value=row.alias, key=f"ali_{i}")
            with c4:
                unit = st.text_input("Unit", value=row.unit, key=f"uni_{i}")
            with c5:
                cost = st.number_input(
                    "Cost AUD", min_value=0.0, value=float(row.indicative_cost_aud),
                    step=0.5, format="%.2f", key=f"cst_{i}",
                )
            edited_rows.append({
                "library_key": row.library_key,
                "canonical_name": canonical,
                "alias": alias,
                "unit": unit,
                "indicative_cost_aud": cost,
                "symbol_image_b64": row.symbol_image_b64,
                "source_pdf": row.source_pdf,
                "source_page": row.source_page,
            })

    cta_cols = st.columns([1, 1, 3])
    with cta_cols[0]:
        if st.button("Ingest all rows", type="primary", use_container_width=True):
            n_added, n_overwritten = _ingest_edited_rows(edited_rows)
            _get_rag.clear()
            # Defer the pack switch to next rerun — see _init_state() for why.
            st.session_state["_pending_symbol_pack"] = "user"
            st.session_state["legend_extraction"] = None
            st.success(
                f"Ingested {len(edited_rows)} symbols "
                f"({n_added} new, {n_overwritten} overwritten). "
                "Sidebar pack switched to 'user'."
            )
            st.rerun()
    with cta_cols[1]:
        if st.button("Discard preview", use_container_width=True):
            st.session_state["legend_extraction"] = None
            st.rerun()


def _ingest_edited_rows(rows: list[dict]) -> tuple[int, int]:
    """Mirror of scripts/ingest_legend._merge_into_additions, but consumes the
    edited-in-the-UI dicts (which may differ from the raw LegendRow values)."""
    USER_SYMBOL_DIR.mkdir(parents=True, exist_ok=True)

    if USER_ADDITIONS_PATH.exists():
        try:
            existing = json.loads(USER_ADDITIONS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    existing.setdefault("symbols", {})
    existing.setdefault("_meta", {})

    added = 0
    overwritten = 0
    for row in rows:
        key = row["library_key"]
        aliases = [a.strip() for a in row["alias"].split(",") if a.strip()]
        if not aliases:
            aliases = [row["canonical_name"].lower()]
        entry = {
            "canonical_name": row["canonical_name"],
            "aliases": aliases,
            "unit": row["unit"] or "ea",
            "spec": (f"Ingested from legend; alias phrase: '{row['alias']}'."
                     if row["alias"] else "Ingested from legend."),
            "indicative_cost_aud": float(row["indicative_cost_aud"]),
            "user_added": True,
            "source_pdf": row["source_pdf"],
            "source_page": row["source_page"],
        }
        if key in existing["symbols"]:
            overwritten += 1
        else:
            added += 1
        existing["symbols"][key] = entry
        png_dest = USER_SYMBOL_DIR / f"{key}.png"
        png_dest.write_bytes(base64.b64decode(row["symbol_image_b64"]))

    existing["_meta"].update({
        "last_ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_symbols": len(existing["symbols"]),
    })
    USER_ADDITIONS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return added, overwritten


def _reset_user_pack() -> None:
    if USER_SYMBOL_DIR.exists():
        shutil.rmtree(USER_SYMBOL_DIR, ignore_errors=True)
    if USER_ADDITIONS_PATH.exists():
        USER_ADDITIONS_PATH.unlink()


def _tab_bom() -> None:
    st.subheader("5. Bill of Materials")
    run: PipelineRun = st.session_state["run"]
    bom = run.bom

    st.caption(
        "Deterministic aggregation produces quantities/specs/costs from "
        "(CV detections, library.json). Claude is constrained to extracting "
        "project metadata from the OCR text + writing the estimator note. "
        "Any Claude value that conflicts with the deterministic source-of-truth "
        "is silently overridden — hallucinated quantities are structurally "
        "impossible."
    )

    meta = st.columns(4)
    meta[0].metric("Project", bom.project_name or "—")
    meta[1].metric("Drawing No.", bom.drawing_number or "—")
    meta[2].metric("Revision", bom.revision or "—")
    meta[3].metric("Date", bom.drawing_date or "—")

    _band_icon = {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low",
                  "unknown": "⚪ unknown"}

    def _row_dict(li, *, include_room: bool):
        row = {
            "library_key": li.library_key,
            "canonical_name": li.canonical_name,
            "qty": li.quantity,
            "unit": li.unit,
            "unit AUD": li.unit_cost_aud,
            "line total AUD": li.line_total_aud,
            "spec": li.spec,
            "CV conf": _band_icon.get(li.confidence_band, li.confidence_band),
            "avg score": round(li.avg_detection_score, 3),
            "src pages": ", ".join(map(str, li.source_pages)),
        }
        if include_room:
            return {"room": li.room or "Unassigned", **row}
        return row

    # Surface the room-distribution sanity warning at the top of the BOM
    # tab so it's the first thing the estimator sees before drilling in.
    diag = st.session_state.get("diagnostics") or {}
    dist_warn = diag.get("room_distribution_warning")
    if dist_warn:
        st.warning(dist_warn)
    # Also surface the room-extraction-itself warning (zero rooms, fallback
    # added N rooms, etc.) — separate signal, distinct from distribution.
    re_diag = diag.get("room_extraction") or {}
    re_warn = re_diag.get("warning")
    if re_warn and re_warn != dist_warn:
        st.info(re_warn)

    has_rooms = bool(bom.rooms_detected)
    tab_room, tab_sym = st.tabs([
        f"By Room ({len(bom.rooms_detected)})" if has_rooms else "By Room (n/a)",
        f"By Symbol ({len(bom.by_symbol())})",
    ])

    with tab_room:
        if not has_rooms:
            st.info(
                "Room extraction was disabled or produced no labels — no per-room "
                "breakdown available. The By-Symbol tab has the global view."
            )
            diag_re = (st.session_state.get("diagnostics") or {}).get("room_extraction")
            if diag_re and diag_re.get("warning"):
                st.warning(diag_re["warning"])
        else:
            grouped = bom.by_room()
            # Decide which rooms open by default — only the largest 3 by qty
            room_totals = [
                (room, sum(li.quantity for li in items),
                 sum(li.line_total_aud for li in items), items)
                for room, items in grouped.items()
                if room != "Unassigned"  # Unassigned is hidden when empty (req: option A)
            ]
            top3_open = {
                r for r, _, _, _ in sorted(room_totals, key=lambda x: -x[1])[:3]
            }
            for room, qty, rtot, items in sorted(room_totals, key=lambda x: -x[1]):
                header = f"**{room}** — {qty} item{'s' if qty != 1 else ''} · AUD {rtot:,.2f}"
                with st.expander(header, expanded=(room in top3_open)):
                    st.dataframe(
                        [_row_dict(li, include_room=False) for li in items],
                        hide_index=True, use_container_width=True,
                    )
            unassigned = grouped.get("Unassigned") or []
            if unassigned:
                u_qty = sum(li.quantity for li in unassigned)
                u_tot = sum(li.line_total_aud for li in unassigned)
                with st.expander(
                    f"**Unassigned** — {u_qty} item · AUD {u_tot:,.2f} "
                    "_(could not pin to a room)_",
                    expanded=False,
                ):
                    st.dataframe(
                        [_row_dict(li, include_room=False) for li in unassigned],
                        hide_index=True, use_container_width=True,
                    )

    with tab_sym:
        st.dataframe(
            [_row_dict(li, include_room=False) for li in bom.by_symbol()],
            hide_index=True, use_container_width=True,
        )

    low_conf = [li for li in bom.line_items if li.confidence_band == "low"]
    if low_conf:
        st.warning(
            "**Low CV confidence on:** "
            + ", ".join(f"`{li.library_key}` (avg {li.avg_detection_score:.2f})" for li in low_conf)
            + " — counts on these lines may include CV false positives. "
            "Estimator should sight-check before quoting."
        )

    diagnostics = st.session_state.get("diagnostics")
    if diagnostics and any(
        diagnostics.get(k) for k in (
            "qty_overrides", "cost_overrides", "spec_overrides",
            "spurious_line_items_dropped", "missing_line_items_added",
        )
    ):
        with st.expander("🛡️ Override mechanism findings (Claude vs deterministic truth)", expanded=True):
            st.caption(
                "Each finding here was Claude attempting to mutate a value that "
                "the deterministic source-of-truth disagreed with. The BOM you "
                "see above already has the override applied."
            )
            if diagnostics["qty_overrides"]:
                st.markdown("**Quantity overrides:**")
                st.dataframe(diagnostics["qty_overrides"], hide_index=True, use_container_width=True)
            if diagnostics["cost_overrides"]:
                st.markdown("**Cost overrides:**")
                st.dataframe(diagnostics["cost_overrides"], hide_index=True, use_container_width=True)
            if diagnostics["spec_overrides"]:
                st.markdown(f"**Spec overrides:** {', '.join(diagnostics['spec_overrides'])}")
            if diagnostics["spurious_line_items_dropped"]:
                st.markdown(f"**Spurious items dropped:** {', '.join(diagnostics['spurious_line_items_dropped'])}")
            if diagnostics["missing_line_items_added"]:
                st.markdown(f"**Missing items reinstated:** {', '.join(diagnostics['missing_line_items_added'])}")

    totals = st.columns([3, 1])
    totals[0].markdown(" ")
    totals[1].metric("Subtotal (AUD)", f"{bom.subtotal_aud:,.2f}")

    if bom.notes_from_assembler:
        st.markdown("### Estimator note (from Claude)")
        st.info(bom.notes_from_assembler)

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow([
        "room", "library_key", "canonical_name", "quantity", "unit",
        "unit_cost_aud", "line_total_aud", "spec", "source_pages",
    ])
    for li in bom.line_items:
        writer.writerow([
            li.room or "Unassigned",
            li.library_key, li.canonical_name, li.quantity, li.unit,
            li.unit_cost_aud, li.line_total_aud, li.spec,
            "|".join(map(str, li.source_pages)),
        ])
    st.download_button(
        "Download BOM as CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"bom_{run.run_id}.csv",
        mime="text/csv",
    )

    with st.expander("Raw BOM JSON (the §2-style contract)"):
        st.code(bom.model_dump_json(indent=2), language="json")


# ---------- sidebar ----------

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## ProCalc Interview POC")
        st.caption("Electrical Layout → BOM, end-to-end.")

        run: PipelineRun | None = st.session_state.get("run")
        src = st.session_state.get("source_label", "—")
        st.markdown(f"**Source:** `{src}`")

        if run is not None:
            st.markdown("### Run stats")
            stats = st.columns(2)
            stats[0].metric("Detections", len(run.detections))
            stats[1].metric("OCR spans", len(run.ocr_spans))
            stats2 = st.columns(2)
            stats2[0].metric("BOM lines", len(run.bom.line_items))
            stats2[1].metric("Subtotal AUD", f"{run.bom.subtotal_aud:,.0f}")
            st.metric("Pipeline elapsed", f"{run.elapsed_seconds}s")

        st.divider()
        st.markdown("### Model / prompt")
        st.code(f"model:  {DEFAULT_MODEL}\nprompt: {DEFAULT_VERSION}", language="text")

        st.divider()
        st.markdown("### Symbol pack")
        user_count = user_pack_size()
        pack_options = ["builtin", "user", "both"] if user_count else ["builtin"]
        current_pack = st.session_state.get("symbol_pack", "builtin")
        if current_pack not in pack_options:
            current_pack = pack_options[0]
            st.session_state["symbol_pack"] = current_pack
        st.radio(
            f"Templates used by the CV detector (user pack: {user_count} symbol(s))",
            options=pack_options,
            index=pack_options.index(current_pack),
            key="symbol_pack",
            help="`builtin` = the 15 PNGs shipped with the POC. `user` = "
                 "symbols ingested from a legend page on tab 6. `both` = "
                 "union, with user shadowing built-in on name collision.",
        )
        if user_count == 0:
            st.caption(
                "_To enable `user` and `both`: ingest a legend page on tab 6._"
            )

        st.divider()
        st.markdown("### Demo controls")
        st.checkbox(
            "Use Claude ROI crop on plan page",
            key="use_roi",
            help="Before CV runs, Claude crops the page to just the floor-plan "
                 "region (excluding title block, in-page legend, notes). "
                 "Dramatically reduces false positives from non-plan content. "
                 "Adds ~10-15s per run; usually net-faster because CV runs on a "
                 "smaller image. Toggle off to disable.",
        )
        st.checkbox(
            "Force Claude to deviate (demo override)",
            key="force_deviation",
            help="Next live pipeline run uses the bom_v1_force_deviation prompt. "
                 "Claude is instructed to inflate 2 quantities and round 2 costs up; "
                 "the override mechanism catches and corrects, populating the "
                 "diagnostics panel in the BOM tab.",
        )

        st.divider()
        st.markdown(
            "### Failure modes mitigated\n"
            "- ✓ Three-stage detection (CV + OCR + RAG), not LLM-vision-only  \n"
            "- ✓ Claude never authors quantities, specs, or costs  \n"
            "- ✓ Versioned, data-driven prompts  \n"
            "- ✓ Stable page identifiers across stages  \n"
            "- ✓ Retry / timeout / partial-result handling  \n"
            "- ✓ Audit store written for every Claude call  \n"
            "- ✓ Cached-fallback path for live-demo safety"
        )

        st.divider()
        st.caption(
            "Architecture details in `implementation_plan.md` (project root); "
            "demo script in `poc/README.md`."
        )


# ---------- entry ----------

###############################################################################
# Phase 8 — Wizard UI
###############################################################################

_WIZARD_STEPS = [
    (1, "Upload"),
    (2, "Symbol pack"),
    (3, "Ingest legend"),
    (4, "Run pipeline"),
    (5, "Dashboard"),
]


def _wizard_goto(step: int) -> None:
    st.session_state["wizard_step"] = step


def _wizard_reset() -> None:
    """Reset wizard state but preserve user pack + cached run on disk."""
    for k in (
        "wizard_step", "wizard_pdf_path", "wizard_pdf_name", "wizard_n_pages",
        "wizard_pack_choice", "wizard_legend_ingested",
        "page_classification", "page_classification_for",
        "legend_extraction", "legend_pdf_path", "legend_pdf_name",
        "run", "page_image", "source_label", "diagnostics",
        "cv_verify_findings", "cv_verify_error",
    ):
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["wizard_step"] = 1


def _render_stepper(current: int) -> None:
    """Horizontal step indicator at the top of the page."""
    # On the user-pack path, step 3 is active; on builtin path, step 3 is skipped (greyed).
    pack = st.session_state.get("wizard_pack_choice")
    cols = st.columns(len(_WIZARD_STEPS))
    for i, (num, label) in enumerate(_WIZARD_STEPS):
        with cols[i]:
            is_current = num == current
            is_done = num < current
            # Skip step 3 visually if user picked builtin
            is_skipped = (num == 3 and pack == "builtin")
            if is_current:
                colour = "#1f77b4"
                bg = "#e3f0fa"
                badge = "●"
            elif is_done:
                colour = "#16a34a"
                bg = "#e6f7ec"
                badge = "✓"
            elif is_skipped:
                colour = "#aaa"
                bg = "#f5f5f5"
                badge = "—"
            else:
                colour = "#888"
                bg = "#f5f5f5"
                badge = "○"
            st.markdown(
                f"""<div style="text-align:center;padding:8px;border-radius:6px;
                        background:{bg};border:1px solid {colour};">
                    <div style="font-size:18px;color:{colour};font-weight:700;">
                        {badge} Step {num}
                    </div>
                    <div style="font-size:13px;color:{colour};">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------- step 1: upload ----------------

def _wizard_step1_upload(api_key_ok: bool) -> None:
    st.subheader("Step 1 — Upload a construction-drawing PDF")
    st.caption(
        "Drop in your drawing set. Claude will classify each page (cover, "
        "electrical plan, legend, …) so the right page gets used at each "
        "later step. Or, jump straight to the cached demo run for the "
        "synthetic fixture."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        uploaded = st.file_uploader(
            "Drawing PDF",
            type=["pdf"],
            key="wizard_upload",
        )
        if uploaded is not None and uploaded.name != st.session_state.get("wizard_pdf_name"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)
            st.session_state["wizard_pdf_path"] = tmp_path
            st.session_state["wizard_pdf_name"] = uploaded.name
            try:
                st.session_state["wizard_n_pages"] = pdf_page_count(tmp_path)
            except Exception:  # noqa: BLE001
                st.session_state["wizard_n_pages"] = 1
            # invalidate previous classification
            st.session_state["page_classification"] = None
            st.session_state["page_classification_for"] = None
            st.rerun()

        if st.session_state.get("wizard_pdf_path") is not None:
            st.success(
                f"Loaded **{st.session_state['wizard_pdf_name']}** · "
                f"{st.session_state['wizard_n_pages']} page(s)"
            )
            # Auto-classify on first encounter
            pc = st.session_state.get("page_classification")
            if api_key_ok and (
                pc is None
                or st.session_state.get("page_classification_for") != st.session_state["wizard_pdf_name"]
            ):
                with st.spinner("Claude is classifying the pages..."):
                    pc = classify_pdf(st.session_state["wizard_pdf_path"])
                st.session_state["page_classification"] = pc
                st.session_state["page_classification_for"] = st.session_state["wizard_pdf_name"]

    with col_side:
        st.markdown("### Detected sheets")
        pc = st.session_state.get("page_classification")
        if pc is None:
            st.info("Upload a PDF (left) to see Claude's per-page classification here.")
        elif pc.error:
            st.error(f"Classifier failed: {pc.error}")
        elif not pc.pages:
            st.warning("Classifier returned no pages.")
        else:
            for p in sorted(pc.pages, key=lambda p: p.page_index):
                bar = "🟢" if p.confidence > 0.85 else ("🟡" if p.confidence > 0.6 else "🔴")
                st.markdown(
                    f"{bar} **p{p.page_index}** · `{p.sheet_class}` "
                    f"<span style='color:#888;font-size:0.85em'>conf {p.confidence:.2f}</span>",
                    unsafe_allow_html=True,
                )
            if pc.suggested_plan_page:
                st.success(f"Plan page: **{pc.suggested_plan_page}**")
            if pc.suggested_legend_page:
                st.info(f"Legend page: **{pc.suggested_legend_page}**")

    st.divider()
    nav_cols = st.columns([3, 1, 1])
    with nav_cols[1]:
        if st.button("Open cached demo", use_container_width=True):
            if _load_cached_into_state():
                _wizard_goto(5)
                st.rerun()
            else:
                st.error("No cached run available.")
    with nav_cols[2]:
        next_ok = st.session_state.get("page_classification") is not None and \
                  not (st.session_state["page_classification"].error)
        if st.button("Next →", type="primary", use_container_width=True, disabled=not next_ok):
            _wizard_goto(2)
            st.rerun()


# ---------------- step 2: pack picker ----------------

def _wizard_step2_pack() -> None:
    st.subheader("Step 2 — Pick the symbol library")
    st.caption(
        "Which symbol templates should the CV detector use against your "
        "plan page? You can change this later from the sidebar."
    )
    pc = st.session_state.get("page_classification")
    has_legend = bool(pc and pc.suggested_legend_page)
    pdf_name = (st.session_state.get("wizard_pdf_name") or "").lower()
    # Built-in templates were drawn by symbol_library/generate_symbols.py
    # to match symbols placed by synth/generate_sample.py. They only match
    # the synth fixture; on any other PDF the built-in pack returns ~0
    # detections. Detect that case and steer the user.
    is_synth_fixture = "synthetic_layout" in pdf_name and "procalc" not in pdf_name

    if not is_synth_fixture:
        st.warning(
            "⚠️ **Heads-up — your uploaded PDF is not the synthetic test fixture.** "
            "The Built-in pack contains 15 templates I drew specifically to match "
            "the symbols on `synthetic_layout.pdf`. On *any other PDF* (including "
            "this one), the built-in templates won't visually match the drawing's "
            "symbol style and detection will return **~0 detections**. "
            "**Use the User pack** (right card) — it ingests your PDF's own "
            "legend so detection matches your drawing's actual symbol style."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "### 🔵 Built-in (15 symbols)\n"
            "Use the POC's 15 default templates (single/double GPO, switches, "
            "lights, fans, DB, …). **Best for the synthetic test fixture** — "
            "the templates match its style exactly."
        )
        if is_synth_fixture:
            st.caption("✅ This PDF *is* the synth fixture — built-in pack will work.")
        else:
            st.caption("🔴 Not recommended for this PDF — see warning above.")
        if st.button(
            "Use built-in (skip to Step 4)",
            use_container_width=True,
            key="pack_builtin_btn",
            type="primary" if is_synth_fixture else "secondary",
        ):
            st.session_state["wizard_pack_choice"] = "builtin"
            st.session_state["_pending_symbol_pack"] = "builtin"
            _wizard_goto(4)
            st.rerun()
    with col2:
        st.markdown(
            "### 🟢 User pack (from your PDF's legend)\n"
            "Extract the symbols from your PDF's legend page (Claude found "
            f"page **{pc.suggested_legend_page if has_legend else '?'}** if any), "
            "preview them, and ingest into the user pack. Detection will use "
            "those templates instead."
        )
        if is_synth_fixture:
            st.caption("(Optional for this PDF — built-in works fine.)")
        else:
            st.caption("✅ Recommended for this PDF.")
        if st.button(
            "Use user pack (go to Step 3 to ingest)",
            type="primary" if not is_synth_fixture else "secondary",
            use_container_width=True,
            key="pack_user_btn",
            disabled=not has_legend,
        ):
            st.session_state["wizard_pack_choice"] = "user"
            _wizard_goto(3)
            st.rerun()
        if not has_legend:
            st.caption("_(disabled: no legend page detected on this PDF)_")

    st.divider()
    nav = st.columns([1, 5, 1])
    with nav[0]:
        if st.button("← Back", use_container_width=True, key="step2_back"):
            _wizard_goto(1)
            st.rerun()


# ---------------- step 3: legend ingest ----------------

def _wizard_step3_legend() -> None:
    st.subheader("Step 3 — Ingest the legend")
    pc = st.session_state.get("page_classification")
    pdf_path: Path = st.session_state.get("wizard_pdf_path")
    legend_page = pc.suggested_legend_page if pc else None

    st.caption(
        f"Extracting symbols from **{st.session_state['wizard_pdf_name']}** "
        f"page **{legend_page}** (the page Claude classified as the legend). "
        "Review the rows below, fix any OCR noise, then click *Ingest all rows*."
    )

    extraction = st.session_state.get("legend_extraction")
    # Auto-run extraction the first time we hit this step
    if extraction is None or st.session_state.get("legend_pdf_name") != st.session_state["wizard_pdf_name"]:
        with st.spinner(f"Extracting legend rows from page {legend_page}..."):
            try:
                extraction = extract_legend(pdf_path, int(legend_page))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Extraction error: {exc}")
                extraction = None
        st.session_state["legend_extraction"] = extraction
        st.session_state["legend_pdf_path"] = pdf_path
        st.session_state["legend_pdf_name"] = st.session_state["wizard_pdf_name"]

    if extraction is None:
        nav = st.columns([1, 5, 1])
        with nav[0]:
            if st.button("← Back", use_container_width=True, key="step3_back_err"):
                _wizard_goto(2); st.rerun()
        return

    st.markdown(f"### Preview — {len(extraction.rows)} row(s)")
    if extraction.warnings:
        with st.expander(f"⚠️ {len(extraction.warnings)} warning(s)"):
            for w in extraction.warnings:
                st.write(f"- {w}")

    edited_rows: list[dict] = []
    for i, row in enumerate(extraction.rows):
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1, 3, 3, 1, 1])
            with c1:
                img_bytes = base64.b64decode(row.symbol_image_b64)
                st.image(img_bytes, width=64)
            with c2:
                canonical = st.text_input("Canonical name", value=row.canonical_name,
                                          key=f"wcan_{i}")
            with c3:
                alias = st.text_input("Alias / spec phrase", value=row.alias,
                                      key=f"wali_{i}")
            with c4:
                unit = st.text_input("Unit", value=row.unit, key=f"wuni_{i}")
            with c5:
                cost = st.number_input("Cost AUD", min_value=0.0,
                                       value=float(row.indicative_cost_aud),
                                       step=0.5, format="%.2f", key=f"wcst_{i}")
            edited_rows.append({
                "library_key": row.library_key,
                "canonical_name": canonical,
                "alias": alias,
                "unit": unit,
                "indicative_cost_aud": cost,
                "symbol_image_b64": row.symbol_image_b64,
                "source_pdf": row.source_pdf,
                "source_page": row.source_page,
            })

    st.divider()
    nav = st.columns([1, 3, 1, 1])
    with nav[0]:
        if st.button("← Back", use_container_width=True, key="step3_back"):
            _wizard_goto(2); st.rerun()
    with nav[2]:
        if st.button("Ingest all rows", type="primary", use_container_width=True,
                     key="step3_ingest"):
            n_added, n_overwritten = _ingest_edited_rows(edited_rows)
            _get_rag.clear()
            st.session_state["_pending_symbol_pack"] = "user"
            st.session_state["wizard_legend_ingested"] = True
            st.session_state["legend_extraction"] = None
            st.success(
                f"Ingested {len(edited_rows)} symbols "
                f"({n_added} new, {n_overwritten} overwritten)."
            )
    with nav[3]:
        next_ok = bool(st.session_state.get("wizard_legend_ingested"))
        if st.button("Next →", type="primary", use_container_width=True,
                     disabled=not next_ok, key="step3_next"):
            _wizard_goto(4); st.rerun()
        if not next_ok:
            st.caption("_(Ingest first)_")


# ---------------- step 4: run pipeline ----------------

def _wizard_step4_run(api_key_ok: bool) -> None:
    st.subheader("Step 4 — Run the pipeline")
    pdf_path: Path = st.session_state.get("wizard_pdf_path")
    pc = st.session_state.get("page_classification")
    n_pages = st.session_state.get("wizard_n_pages", 1)
    pack = st.session_state.get("wizard_pack_choice", "builtin")

    default_page = pc.suggested_plan_page if (pc and pc.suggested_plan_page) else 1

    st.caption(
        f"Running on **{st.session_state.get('wizard_pdf_name', '?')}** with "
        f"the **{pack}** symbol pack. The plan page is auto-suggested below; "
        "you can override before running."
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        if n_pages > 1:
            page_index = st.number_input(
                f"Plan page (PDF has {n_pages} pages)",
                min_value=1, max_value=n_pages,
                value=min(default_page, n_pages), step=1,
                key="wizard_run_page",
            )
        else:
            page_index = 1
            st.write("Single-page PDF — using page 1.")
        st.session_state["pdf_page_index"] = int(page_index)

        st.checkbox(
            "Use Claude ROI crop on plan page",
            key="use_roi",
            help="Before CV runs, Claude crops the page to just the floor-plan "
                 "region. Faster + cleaner BOM. Disable for max recall on "
                 "PDFs where the auto-crop misses edge symbols.",
        )

        run_clicked = st.button(
            f"▶ Run pipeline on page {page_index}",
            type="primary",
            use_container_width=True,
            disabled=not api_key_ok,
            key="step4_run_btn",
        )
        if run_clicked:
            label_pack = pack if pack else "builtin"
            _run_against_pdf(pdf_path,
                             source_label=f"live: {st.session_state['wizard_pdf_name']} (pack={label_pack})")
            # On successful run, advance to dashboard
            if st.session_state.get("run") is not None:
                _wizard_goto(5)
                st.rerun()

    with col_right:
        st.markdown("### Run settings")
        st.markdown(f"- **Symbol pack:** `{pack}`")
        st.markdown(f"- **Plan page:** `{page_index}`")
        st.markdown(f"- **Claude ROI crop:** {'on' if st.session_state.get('use_roi') else 'off'}")
        if pack == "user":
            st.caption("(User pack ingested in step 3 is now active.)")
        elif pack == "builtin":
            st.caption("(Built-in 15 symbols. Use only with the synthetic fixture for clean results.)")

    st.divider()
    nav = st.columns([1, 5, 1])
    with nav[0]:
        if st.button("← Back", use_container_width=True, key="step4_back"):
            _wizard_goto(3 if pack == "user" else 2); st.rerun()


# ---------------- step 5: dashboard ----------------

def _wizard_step5_dashboard() -> None:
    if st.session_state.get("run") is None:
        st.warning("No run loaded. Start a new run or open the cached demo.")
        if st.button("Restart wizard"):
            _wizard_reset(); st.rerun()
        return

    st.subheader("Results dashboard")
    st.caption(
        "Drill into the four pipeline-stage panels below. CV detections show "
        "what the matcher found; OCR shows text spans; RAG shows symbol-to-"
        "library matching; BOM shows the final assembled bill."
    )

    tab_cv, tab_ocr, tab_rag, tab_bom = st.tabs(
        ["CV Detections", "OCR", "RAG Matches", "BOM"]
    )
    with tab_cv:
        _tab_cv()
    with tab_ocr:
        _tab_ocr()
    with tab_rag:
        _tab_rag()
    with tab_bom:
        _tab_bom()


# ---------------- wizard sidebar ----------------

def _wizard_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ProCalc Interview POC")
        st.caption("Electrical Layout → BOM, end-to-end.")
        step = st.session_state.get("wizard_step", 1)

        st.markdown(f"### Step {step} of 5")
        st.progress(min(1.0, step / 5))

        run: PipelineRun | None = st.session_state.get("run")
        src = st.session_state.get("source_label", "—")
        if run is not None and step == 5:
            st.markdown(f"**Source:** `{src}`")
            st.metric("Detections", len(run.detections))
            st.metric("BOM lines", len(run.bom.line_items))
            st.metric("Rooms found", len(run.bom.rooms_detected))
            st.metric("Subtotal AUD", f"{run.bom.subtotal_aud:,.0f}")
            st.metric("Pipeline elapsed", f"{run.elapsed_seconds}s")

        st.divider()
        st.markdown("### Demo controls")
        roi_on = bool(st.session_state.get("use_roi", True))
        st.markdown(
            f"**Claude ROI crop:** {'🟢 on' if roi_on else '⚪ off'}  "
            "_(toggle on Step 4 before running)_"
        )
        st.checkbox(
            "Force Claude to deviate (demo override)",
            key="force_deviation",
            help="Next live pipeline run uses the bom_v1_force_deviation prompt.",
        )
        st.checkbox(
            "Claude vision room fallback",
            key="enable_claude_room_fallback",
            help="When OCR finds fewer than 5 rooms, ask Claude vision to "
                 "enumerate any missed labels. Costs ~1 extra Claude call (~$0.01) "
                 "but significantly improves room recall on plans with small/stylised "
                 "labels Tesseract misses.",
        )

        # CV backend selector (only when best.onnx is available)
        from pathlib import Path as _P
        _yolo_available = (_P(__file__).resolve().parent.parent / "best.onnx").exists()
        if _yolo_available:
            backend_options = ["template", "yolo"]
            current_backend = st.session_state.get("cv_backend", "template")
            if current_backend not in backend_options:
                current_backend = "template"
                st.session_state["cv_backend"] = current_backend
            st.radio(
                "CV backend",
                options=backend_options,
                index=backend_options.index(current_backend),
                key="cv_backend",
                help="template = multi-scale matching against symbol PNGs. "
                     "yolo = trained YOLOv8 (best.onnx, 15-class library). "
                     "YOLO ignores the active symbol pack.",
            )

        user_count = user_pack_size()
        if user_count > 0:
            st.divider()
            st.markdown(f"**User pack:** {user_count} symbol(s)")
            pack_options = ["builtin", "user", "both"]
            current_pack = st.session_state.get("symbol_pack", "builtin")
            if current_pack not in pack_options:
                current_pack = pack_options[0]
                st.session_state["symbol_pack"] = current_pack
            st.radio(
                "Active symbol pack (advanced)",
                options=pack_options,
                index=pack_options.index(current_pack),
                key="symbol_pack",
            )

        st.divider()
        st.markdown("### Model / prompt")
        st.code(f"model:  {DEFAULT_MODEL}\nprompt: {DEFAULT_VERSION}", language="text")

        st.divider()
        if st.button("🔄 Restart wizard", use_container_width=True):
            _wizard_reset()
            st.rerun()

        st.caption(
            "Architecture details in `implementation_plan.md`; "
            "demo script in `poc/README.md`."
        )


# ---------------- main dispatcher ----------------

def main() -> None:
    st.set_page_config(
        page_title="ProCalc POC — Electrical BOM",
        layout="wide",
    )
    _init_state()
    api_key_ok = _check_api_key()

    _wizard_sidebar()

    st.title("ProCalc Interview POC — Electrical Layout → BOM")
    st.caption(
        "Wizard flow: upload → pick pack → (ingest legend) → run pipeline → "
        "results dashboard. Each Claude touch is at a clearly-bounded edge; "
        "CV + RAG remain deterministic."
    )

    _render_stepper(st.session_state.get("wizard_step", 1))
    st.divider()

    step = st.session_state.get("wizard_step", 1)
    if step == 1:
        _wizard_step1_upload(api_key_ok)
    elif step == 2:
        _wizard_step2_pack()
    elif step == 3:
        _wizard_step3_legend()
    elif step == 4:
        _wizard_step4_run(api_key_ok)
    elif step == 5:
        _wizard_step5_dashboard()
    else:
        st.error(f"Unknown wizard step: {step}")
        if st.button("Restart"):
            _wizard_reset(); st.rerun()


if __name__ == "__main__":
    main()
