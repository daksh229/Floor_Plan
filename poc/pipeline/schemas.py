"""Domain models for the electrical-BOM pipeline.

Schemas are added incrementally across phases:
  - Phase 2: Detection
  - Phase 3: OCRSpan, RAGMatch, SymbolWithMatches
  - Phase 4 (here): BOMLineItem, BOM, PipelineRun
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """A single template-match hit after NMS.

    `bbox` is (x0, y0, x1, y1) in source-page-image pixel space.
    `score` is the normalized template-match correlation in [-1, 1]; after
    thresholding it's effectively in [threshold, 1].
    """
    model_config = ConfigDict(frozen=True)

    symbol_class: str = Field(..., description="Library key, e.g. 'double_gpo'")
    bbox: Tuple[int, int, int, int] = Field(..., description="x0, y0, x1, y1 in image px")
    score: float = Field(..., ge=-1.0, le=1.0)
    rotation_deg: int = Field(..., description="Template rotation that produced the hit")
    scale: float = Field(..., gt=0.0, description="Template scale that produced the hit")
    source_page: int = Field(..., ge=1, description="1-indexed page in source PDF")

    @property
    def center(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)


class OCRSpan(BaseModel):
    """A single text span returned by Tesseract after filtering.

    `confidence` is normalised to [0, 1] from Tesseract's native 0-100.
    """
    model_config = ConfigDict(frozen=True)

    text: str
    bbox: Tuple[int, int, int, int] = Field(..., description="x0, y0, x1, y1 in image px")
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_page: int = Field(..., ge=1)


class RAGMatch(BaseModel):
    """One ranked candidate from the symbol-library RAG layer."""
    model_config = ConfigDict(frozen=True)

    library_key: str = Field(..., description="library.json symbol key, e.g. 'wp_gpo'")
    canonical_name: str
    similarity: float = Field(..., ge=-1.0, le=1.0, description="cosine similarity")
    unit: str
    spec: str
    indicative_cost_aud: float


class SymbolWithMatches(BaseModel):
    """A detection plus its top-K library candidates.

    The single best candidate (matches[0]) is what Phase 4's BOM assembler
    will commit to. The full top-K is preserved for the UI panel (Phase 5)
    so the panel can show the RAG mechanism on screen.
    """
    detection: Detection
    matches: List[RAGMatch]

    @property
    def best(self) -> RAGMatch | None:
        return self.matches[0] if self.matches else None


class BOMLineItem(BaseModel):
    """One row of the assembled Bill of Materials.

    Quantities and cost values are authoritative from the deterministic
    aggregator over (Detections, library.json). Claude is constrained to
    NOT mutate these; if it does, post-validation overrides Claude's values
    back to the deterministic source of truth.

    `avg_detection_score` and `min_detection_score` carry forward the CV
    confidence so the BOM UI / estimator note can flag low-confidence lines
    (a hallucinated CV count is the one failure mode the deterministic
    contract can't protect against on its own).
    """
    model_config = ConfigDict(frozen=False)

    library_key: str = Field(..., description="library.json symbol key")
    canonical_name: str
    quantity: int = Field(..., ge=0)
    unit: str
    spec: str
    unit_cost_aud: float = Field(..., ge=0.0)
    line_total_aud: float = Field(..., ge=0.0)
    source_pages: List[int] = Field(default_factory=list, description="1-indexed pages of source detections")
    source_detection_count: int = Field(0, description="how many CV detections fed this line")
    avg_detection_score: float = Field(0.0, ge=0.0, le=1.0, description="mean template-match score for source detections")
    min_detection_score: float = Field(0.0, ge=0.0, le=1.0, description="min template-match score across source detections")
    confidence_band: str = Field("unknown", description="high / medium / low — derived from avg_detection_score")
    notes: str | None = None


class BOM(BaseModel):
    """The final assembled Bill of Materials.

    `notes_from_assembler` is Claude's editorial / sanity-check commentary —
    the only free-text field the model controls. Everything else is sourced
    from deterministic computation or extracted from OCR.
    """
    project_name: str | None = None
    drawing_number: str | None = None
    revision: str | None = None
    drawing_date: str | None = None
    line_items: List[BOMLineItem] = Field(default_factory=list)
    subtotal_aud: float = 0.0
    notes_from_assembler: str | None = None
    prompt_version: str
    model: str
    page_count: int = 1
    generated_at: str = Field(..., description="ISO 8601 timestamp")


class PipelineRun(BaseModel):
    """One end-to-end pipeline invocation, captured for the audit store."""
    run_id: str
    source_pdf: str
    detections: List[Detection]
    ocr_spans: List[OCRSpan]
    symbol_matches: List[SymbolWithMatches]
    bom: BOM
    elapsed_seconds: float


# ---------- Phase 6: legend ingestion ----------

class LegendRow(BaseModel):
    """One row extracted from a legend table.

    Editable in the Streamlit preview before ingest, so the user can correct
    any OCR noise without re-running extraction.
    """
    model_config = ConfigDict(frozen=False)

    library_key: str = Field(..., description="snake_case slug derived from canonical_name")
    canonical_name: str
    alias: str = Field("", description="content of the 'Alias/spec phrase' column")
    unit: str = "ea"
    indicative_cost_aud: float = 0.0
    symbol_bbox: Tuple[int, int, int, int] = Field(..., description="crop region in page-image px")
    source_pdf: str
    source_page: int = Field(..., ge=1)
    row_index: int = Field(..., ge=0)
    symbol_image_b64: str = Field(
        "", description="PNG bytes of the cropped symbol, base64-encoded (for UI preview)"
    )
    ingested: bool = Field(False, description="True after the row has been written to user_additions.json")


class LegendExtraction(BaseModel):
    """Output of legend_extractor.extract_legend()."""
    source_pdf: str
    source_page: int
    page_image_width: int
    page_image_height: int
    rows: List[LegendRow]
    warnings: List[str] = Field(default_factory=list)


# ---------- Phase 7: Claude-vision page classifier + ROI extractor ----------

class PageClassification(BaseModel):
    """Claude's classification of one PDF page as a sheet type."""
    model_config = ConfigDict(frozen=True)

    page_index: int = Field(..., ge=1, description="1-indexed PDF page")
    sheet_class: str = Field(
        ...,
        description="cover | demolition_plan | electrical_plan | floor_plan | "
                    "legend | title_block | notes | other",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field("", description="One-sentence why")


class PDFClassification(BaseModel):
    """Per-page classification for an entire PDF, with derived suggestions."""
    source_pdf: str
    pages: List[PageClassification]
    suggested_plan_page: Optional[int] = Field(
        None, description="1-indexed page Claude thinks is the electrical plan"
    )
    suggested_legend_page: Optional[int] = Field(
        None, description="1-indexed page Claude thinks is the legend"
    )
    error: Optional[str] = None


class ROIResult(BaseModel):
    """Claude's bounding box of the main floor-plan content on a page,
    excluding title block / legend / notes / boundary diagrams.

    `bbox` is in source-page-image px (x0, y0, x1, y1). Whoever consumes
    this should crop the page to bbox before running CV, then translate
    detection coords back to the full-page space.
    """
    model_config = ConfigDict(frozen=True)

    page_index: int = Field(..., ge=1)
    page_width: int = Field(..., gt=0)
    page_height: int = Field(..., gt=0)
    bbox: Optional[Tuple[int, int, int, int]] = Field(
        None, description="(x0, y0, x1, y1) plan region; None if Claude couldn't find one"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    error: Optional[str] = None

    @property
    def has_roi(self) -> bool:
        return self.bbox is not None and self.error is None

    @property
    def offset_xy(self) -> Tuple[int, int]:
        """The (x_offset, y_offset) that needs to be ADDED to a detection's
        bbox after CV runs on the cropped image, to bring it back into
        full-page coordinates. (0, 0) if no ROI."""
        return (self.bbox[0], self.bbox[1]) if self.bbox else (0, 0)
