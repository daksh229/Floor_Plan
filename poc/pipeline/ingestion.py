"""PDF -> page-numbered images. Mirrors ProCalc's "wrapping the existing splitter" (§3, §4.1).

pypdfium2 chosen over pdf2image because it's a self-contained pure-Python wheel
on Windows (no Poppler install). Page identifiers are preserved as the 1-indexed
page number, addressing the §7 "lost page identifiers" failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pypdfium2 as pdfium
from PIL import Image


@dataclass
class PageRender:
    page_number: int  # 1-indexed; survives end-to-end so overlays replay correctly
    image: Image.Image
    width: int
    height: int


def render_pdf(pdf_path: Path, max_long_edge_px: int = 1568, dpi_hint: int = 200) -> List[PageRender]:
    """Render every page. max_long_edge_px caps the image so Claude vision stays under its limit.

    Claude vision recommends <= 1568px on the long edge for best speed/cost.
    """
    pages: List[PageRender] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            # scale = points-per-inch / 72; start from dpi_hint then clamp
            scale = dpi_hint / 72.0
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()
            img = _cap_long_edge(img, max_long_edge_px)
            pages.append(
                PageRender(page_number=i + 1, image=img, width=img.width, height=img.height)
            )
    finally:
        pdf.close()
    return pages


def _cap_long_edge(img: Image.Image, max_px: int) -> Image.Image:
    long_edge = max(img.width, img.height)
    if long_edge <= max_px:
        return img
    ratio = max_px / long_edge
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def thumbnail(img: Image.Image, max_px: int = 220) -> Image.Image:
    t = img.copy()
    t.thumbnail((max_px, max_px), Image.LANCZOS)
    return t
