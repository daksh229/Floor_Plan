"""ProCalc AI Measurement — interview POC.

Demonstrates the five-step feature from §2 of the proposal end-to-end on one page:
  1. PDF ingestion (page-numbered images)
  2. Claude vision extraction into strict §2 JSON contract
  3. Calculator-field population (results table)
  4. Approval gate before Calculate/Save unlocks
  5. Audit trail written for every request/response
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from pipeline.extractor import DEFAULT_MODEL, extract_from_page
from pipeline.ingestion import PageRender, render_pdf, thumbnail
from pipeline.overlay import draw_fields
from pipeline.prompts import DEFAULT_VERSION, PROMPT_REGISTRY
from pipeline.schemas import (
    ApprovalState,
    DetectionState,
    ExtractedField,
    ExtractionResult,
    ReviewableField,
)


load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(page_title="ProCalc AI Measurement — POC", layout="wide")


AUDIT_DIR = Path(__file__).parent / "audit_store"


def _check_api_key() -> None:
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        st.error(
            "ANTHROPIC_API_KEY is not set. Open `poc/.env` and paste your "
            "Anthropic key after `ANTHROPIC_API_KEY=`, then restart Streamlit."
        )
        st.stop()


_STATE_BADGE = {
    DetectionState.DETECTED_ACTUAL: "🟢 actual",
    DetectionState.DETECTED_PLACEHOLDER: "🟡 placeholder",
    DetectionState.NOT_DETECTED: "⚪ not detected",
}

_APPROVAL_BADGE = {
    ApprovalState.PENDING_REVIEW: "⏳ pending",
    ApprovalState.SIGHTED: "✅ sighted",
    ApprovalState.CORRECTED: "✏️ corrected",
    ApprovalState.REJECTED: "❌ rejected",
}


# ---------- session helpers ----------

def _init_state() -> None:
    st.session_state.setdefault("pages", [])               # List[PageRender]
    st.session_state.setdefault("pdf_name", None)
    st.session_state.setdefault("selected_page", None)     # 1-indexed
    st.session_state.setdefault("result", None)            # ExtractionResult
    st.session_state.setdefault("reviewables", {})         # variable -> ReviewableField


def _reset_extraction() -> None:
    st.session_state["result"] = None
    st.session_state["reviewables"] = {}


def _all_reviewed(reviewables: Dict[str, ReviewableField]) -> bool:
    """Approval gate: every variable that the model returned must be sighted/corrected/rejected."""
    if not reviewables:
        return False
    for r in reviewables.values():
        if r.extracted.detection_state == DetectionState.NOT_DETECTED:
            continue  # not_detected variables fall back to manual entry, no sighting needed
        if r.approval == ApprovalState.PENDING_REVIEW:
            return False
    return True


# ---------- UI sections ----------

def _sidebar() -> None:
    with st.sidebar:
        st.header("Configuration")
        st.text_input("Model", value=DEFAULT_MODEL, disabled=True, key="model_display")
        prompt_version = st.selectbox(
            "Prompt version",
            options=list(PROMPT_REGISTRY.keys()),
            index=list(PROMPT_REGISTRY.keys()).index(DEFAULT_VERSION),
        )
        st.session_state["prompt_version"] = prompt_version
        st.caption(
            "Prompts are versioned data, not code (proposal §3). Adding a project "
            "type or variable does not require a deploy."
        )
        st.divider()
        st.markdown(
            "**Failure modes mitigated** (proposal §7):\n"
            "- Per-field state machine, not boolean\n"
            "- Three distinct detection states\n"
            "- Stable page identifiers preserved end-to-end\n"
            "- Versioned prompts\n"
            "- Retry, timeout, partial-result handling\n"
            "- Audit store written from request #1"
        )


def _upload_section() -> None:
    st.subheader("1. Upload plan set")
    uploaded = st.file_uploader("PDF plan set", type=["pdf"])
    if uploaded is None:
        return
    if st.session_state["pdf_name"] == uploaded.name and st.session_state["pages"]:
        return

    with st.spinner("Rendering pages…"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        pages = render_pdf(tmp_path)

    st.session_state["pages"] = pages
    st.session_state["pdf_name"] = uploaded.name
    st.session_state["selected_page"] = pages[0].page_number if pages else None
    _reset_extraction()
    st.success(f"Loaded {len(pages)} page(s) from {uploaded.name}")


def _page_picker() -> None:
    pages: List[PageRender] = st.session_state["pages"]
    if not pages:
        return
    st.subheader("2. Pick the Ground Floor page")
    st.caption(
        "In a Phase 1 build, sheet classification would auto-select this. For the "
        "POC, the reviewer picks it — same JSON contract either way."
    )
    cols = st.columns(min(6, len(pages)))
    for idx, page in enumerate(pages):
        with cols[idx % len(cols)]:
            st.image(thumbnail(page.image), caption=f"p. {page.page_number}")
            if st.button(f"Use page {page.page_number}", key=f"pick_{page.page_number}"):
                st.session_state["selected_page"] = page.page_number
                _reset_extraction()


def _extraction_section() -> None:
    pages: List[PageRender] = st.session_state["pages"]
    selected = st.session_state["selected_page"]
    if not pages or selected is None:
        return

    page = next((p for p in pages if p.page_number == selected), None)
    if page is None:
        return

    st.subheader(f"3. Extract from page {selected}")
    left, right = st.columns([1, 1])
    with left:
        st.image(page.image, caption=f"Page {selected} (rendered)")
    with right:
        if st.button("Run AI extraction", type="primary"):
            with st.spinner("Calling Claude…"):
                result = extract_from_page(
                    page_image=page.image,
                    source_page=selected,
                    prompt_version=st.session_state.get("prompt_version", DEFAULT_VERSION),
                    audit_dir=AUDIT_DIR,
                )
            st.session_state["result"] = result
            st.session_state["reviewables"] = {
                f.variable: ReviewableField(extracted=f) for f in result.fields
            }
            if result.error:
                st.error(f"Extraction error: {result.error}")
            else:
                st.success("Extraction complete.")

        result: ExtractionResult | None = st.session_state["result"]
        if result is not None and not result.error:
            overlaid = draw_fields(page.image, result.fields)
            st.image(overlaid, caption="Overlay (solid=actual, dashed=placeholder)")


def _review_section() -> None:
    result: ExtractionResult | None = st.session_state["result"]
    if result is None:
        return

    st.subheader("4. Reviewer approval gate")
    st.caption(
        "Per §2 step 4 of the proposal: the builder must sight every AI-populated "
        "field on the plan before Calculate/Save unlocks."
    )

    reviewables: Dict[str, ReviewableField] = st.session_state["reviewables"]
    for variable, r in reviewables.items():
        with st.container(border=True):
            top = st.columns([3, 1, 1, 1])
            with top[0]:
                st.markdown(f"**{r.extracted.display_name}** (`{variable}`)")
            with top[1]:
                st.markdown(_STATE_BADGE[r.extracted.detection_state])
            with top[2]:
                st.markdown(_APPROVAL_BADGE[r.approval])
            with top[3]:
                st.markdown(f"page {r.extracted.source_page}")

            cols = st.columns([2, 2, 3])
            with cols[0]:
                ai_value = "" if r.extracted.value is None else str(r.extracted.value)
                st.text_input(
                    "AI value",
                    value=ai_value,
                    disabled=True,
                    key=f"ai_{variable}",
                )
            with cols[1]:
                corrected = st.text_input(
                    "Reviewer value (override)",
                    value="" if r.corrected_value is None else str(r.corrected_value),
                    key=f"corr_{variable}",
                )
                r.corrected_value = corrected or None
            with cols[2]:
                st.text_input(
                    "Unit",
                    value=r.extracted.unit or "",
                    disabled=True,
                    key=f"unit_{variable}",
                )
                if r.extracted.confidence_note:
                    st.caption(f"_AI note:_ {r.extracted.confidence_note}")

            actions = st.columns(4)
            if r.extracted.detection_state == DetectionState.NOT_DETECTED:
                actions[0].caption("Manual entry only — no sighting required.")
            else:
                if actions[0].button("Sight on plan", key=f"sight_{variable}"):
                    r.approval = ApprovalState.SIGHTED
                if actions[1].button("Save correction", key=f"savecorr_{variable}"):
                    r.approval = ApprovalState.CORRECTED
                if actions[2].button("Reject", key=f"reject_{variable}"):
                    r.approval = ApprovalState.REJECTED


def _calculate_section() -> None:
    reviewables: Dict[str, ReviewableField] = st.session_state.get("reviewables", {})
    if not reviewables:
        return
    st.subheader("5. Calculate / Save")
    unlocked = _all_reviewed(reviewables)
    if unlocked:
        st.button("Calculate", type="primary", disabled=False)
        st.success("All AI fields sighted — gate unlocked.")
    else:
        st.button("Calculate", type="primary", disabled=True)
        pending = [
            r.extracted.display_name
            for r in reviewables.values()
            if r.extracted.detection_state != DetectionState.NOT_DETECTED
            and r.approval == ApprovalState.PENDING_REVIEW
        ]
        st.warning(
            "Calculate/Save is blocked until every AI field is sighted, corrected, "
            f"or rejected. Pending: {', '.join(pending) if pending else '—'}"
        )


def _raw_section() -> None:
    result: ExtractionResult | None = st.session_state.get("result")
    if result is None:
        return
    with st.expander("Raw JSON (the §2 contract — show this on the interview call)"):
        st.code(result.model_dump_json(indent=2), language="json")
    if result.raw_response:
        with st.expander("Raw Claude response (pre-validation)"):
            st.code(result.raw_response, language="json")


def main() -> None:
    _init_state()
    _check_api_key()
    st.title("ProCalc AI Measurement — POC")
    st.caption(
        "Streamlit demo of the five-step feature in proposal §2: ingest → Claude "
        "vision → strict JSON with overlay coordinates → reviewer approval gate → "
        "audit trail. Built for interview demonstration of spatial / coordinate "
        "extraction from construction plans."
    )
    _sidebar()
    _upload_section()
    _page_picker()
    _extraction_section()
    _review_section()
    _calculate_section()
    _raw_section()


if __name__ == "__main__":
    main()
