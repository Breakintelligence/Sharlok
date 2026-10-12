"""Multi-format architecture input ingestion for Sharlok.

Accepts an uploaded architecture artifact in whatever format it comes in and
extracts a plain-text architecture description that ``threat_modeling.py``
can feed straight into the review pipeline:

  - Images (.png, .jpg, .jpeg, .webp, .gif) -> vision extraction (Ollama gemma4)
  - Mermaid / plain text diagrams (.mmd, .md, .txt) -> read as-is (already text)
  - PDF (.pdf) -> extract embedded text (pypdf/PyPDF2)
  - PowerPoint (.pptx) -> extract slide titles/body/table text (python-pptx)

Any format works irrespective of domain (cloud, on-prem, mobile, AI/ML, etc.) -
this module only cares about turning the artifact into text a text model can
reason about; domain-specific framing lives in prompts.yaml.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("Sharlok.file_ingest")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_EXTS = {".mmd", ".md", ".markdown", ".txt", ".mermaid"}
PDF_EXTS = {".pdf"}
PPTX_EXTS = {".pptx"}

SUPPORTED_EXTS = IMAGE_EXTS | TEXT_EXTS | PDF_EXTS | PPTX_EXTS

_MIN_TEXT_CHARS = 100


def is_image(ext: str) -> bool:
    """Return True if the extension is a raster image handled via vision."""
    return ext.lower() in IMAGE_EXTS


def _extract_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()

    if len(text) < _MIN_TEXT_CHARS:
        raise RuntimeError(
            "This PDF has little or no extractable text (likely a scanned or "
            "image-only diagram). Export the diagram page as PNG/JPG and "
            "upload that instead."
        )
    return text


def _extract_pptx_text(path: str) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "PowerPoint support requires the 'python-pptx' package. "
            "Install it with: uv pip install python-pptx"
        ) from exc

    prs = Presentation(path)
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        slide_lines = [f"Slide {i}:"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        slide_lines.append(line)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        slide_lines.append(" | ".join(cells))
        if len(slide_lines) > 1:
            parts.append("\n".join(slide_lines))

    text = "\n\n".join(parts).strip()
    if len(text) < _MIN_TEXT_CHARS:
        raise RuntimeError(
            "This PowerPoint has little or no extractable text (likely diagram "
            "images with no text layer). Export the slide as PNG/JPG and "
            "upload that instead."
        )
    return text


def extract_architecture_description(file_path: str) -> str:
    """Extract a plain-text architecture description from any supported file.

    Images are handled separately by the vision step in ``threat_modeling.py``;
    this function only handles the non-image, text-extractable formats
    (PDF, PPTX, Mermaid/plain text). Callers should check ``is_image()`` first.

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the extension isn't supported
        RuntimeError: If extraction yields no usable text
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = p.suffix.lower()
    if ext in TEXT_EXTS:
        return _extract_text_file(file_path)
    if ext in PDF_EXTS:
        return _extract_pdf_text(file_path)
    if ext in PPTX_EXTS:
        return _extract_pptx_text(file_path)

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: images (png/jpg/webp/gif), "
        f"PDF, PPTX, Mermaid/text (mmd/md/txt)."
    )
