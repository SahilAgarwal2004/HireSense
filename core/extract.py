"""
core/extract.py
───────────────
Resume file extraction: PDF and DOCX → structured dict.

Public API
----------
extract_resume(path, metadata=False) -> dict
    Unified entry point used by the rest of the pipeline.
    Returns at minimum {"text": str, "filetype": str}.
    With metadata=True, also returns fonts, images, tables, pages.

extract_text_basic(path)        -> str   (raw text only)
extract_with_metadata(path)     -> dict  (text + structural metadata)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import docx
import fitz  # PyMuPDF


# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_resume(path: str, metadata: bool = False) -> dict:
    """
    Unified entry point used by the rest of the pipeline.

    Parameters
    ----------
    path     : str  – absolute or relative path to the resume file
    metadata : bool – include structural metadata (fonts, images, tables, pages)

    Returns
    -------
    dict
        Always contains:
            text     (str)  – full resume text, newlines preserved
            filetype (str)  – "pdf" or "docx"
        Additionally when metadata=True:
            fonts    (dict[str, int]) – {font_name: occurrence_count}
            images   (int)            – number of embedded images
            tables   (int)            – number of tables
            pages    (int | None)     – page count (None for DOCX)

    Raises
    ------
    FileNotFoundError  – file does not exist
    ValueError         – unsupported extension, encrypted PDF, or corrupt file
    """
    if metadata:
        return extract_with_metadata(path)
    return {
        "text": extract_text_basic(path),
        "filetype": _filetype(path),
    }


def extract_text_basic(path: str) -> str:
    """
    Extract plain text from a PDF or DOCX file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the resume file.

    Returns
    -------
    str
        Raw text with newlines preserved (required by sectioner).

    Raises
    ------
    FileNotFoundError  – file does not exist
    ValueError         – unsupported extension, encrypted PDF, or corrupt file
    """
    ext = _validate_and_get_ext(path)
    if ext == ".pdf":
        return _extract_text_from_pdf(path)
    return _extract_text_from_docx(path)


def extract_with_metadata(path: str) -> dict:
    """
    Extract text AND structural metadata (fonts, images, tables, page count).

    Parameters
    ----------
    path : str
        Absolute or relative path to the resume file.

    Returns
    -------
    dict with keys:
        text     (str)            – full resume text, newline-separated pages
        fonts    (dict[str, int]) – {font_name: occurrence_count}
        images   (int)            – number of embedded images
        tables   (int)            – number of tables (PDF always 0 — needs pdfplumber)
        pages    (int | None)     – page count (None for DOCX)
        filetype (str)            – "pdf" or "docx"

    Raises
    ------
    FileNotFoundError  – file does not exist
    ValueError         – unsupported extension, encrypted PDF, or corrupt file
    """
    ext = _validate_and_get_ext(path)
    if ext == ".pdf":
        return _extract_pdf_with_metadata(path)
    return _extract_docx_with_metadata(path)


# ── Validators ─────────────────────────────────────────────────────────────────

def _validate_and_get_ext(path: str) -> str:
    """Check that the file exists and has a supported extension; return lowercase ext."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return ext


def _filetype(path: str) -> str:
    """Return 'pdf' or 'docx' from a file path (no validation)."""
    return os.path.splitext(path)[1].lower().lstrip(".")


# ── PDF helpers ────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF, one newline between pages."""
    try:
        doc = fitz.open(path)
        if doc.is_encrypted:
            raise ValueError(f"PDF is password-protected and cannot be read: {path!r}")
        return "\n".join(page.get_text("text") for page in doc).strip()
    except fitz.FileDataError as exc:
        raise ValueError(
            f"Could not read PDF — file may be corrupted: {path!r}\n  {exc}"
        ) from exc


def _extract_pdf_with_metadata(path: str) -> dict:
    """Extract text + fonts + image count from a PDF."""
    try:
        doc = fitz.open(path)
        if doc.is_encrypted:
            raise ValueError(f"PDF is password-protected and cannot be read: {path!r}")

        text_pages: list[str] = []
        fonts: dict[str, int] = {}
        image_count = 0

        for page in doc:
            text_pages.append(page.get_text("text"))

            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font = span["font"]
                            fonts[font] = fonts.get(font, 0) + 1
                if block["type"] == 1:  # image block
                    image_count += 1

        return {
            "text": "\n".join(text_pages).strip(),
            "fonts": fonts,
            "images": image_count,
            "tables": 0,  # pdfplumber needed for real table detection
            "pages": len(doc),
            "filetype": "pdf",
        }

    except fitz.FileDataError as exc:
        raise ValueError(
            f"Could not read PDF — file may be corrupted: {path!r}\n  {exc}"
        ) from exc


# ── DOCX helpers ───────────────────────────────────────────────────────────────

def _extract_text_from_docx(path: str) -> str:
    """
    Extract plain text from a DOCX file.

    Includes both paragraph text and table cell text.
    Table-based resume layouts are common; silently skipping tables would
    drop large sections of candidate content.
    """
    doc = docx.Document(path)
    parts: list[str] = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    return "\n".join(parts).strip()


def _extract_docx_with_metadata(path: str) -> dict:
    """Extract text + table count from a DOCX file."""
    doc = docx.Document(path)
    parts: list[str] = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    return {
        "text": "\n".join(parts).strip(),
        "fonts": {},        # python-docx exposes run.font per run; add if needed
        "images": 0,        # python-docx inline image access is non-trivial
        "tables": len(doc.tables),
        "pages": None,      # python-docx has no reliable page-count API
        "filetype": "docx",
    }


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "Sahil_Agarwal_Resume_2025.pdf"

    print("── BASIC TEXT ──────────────────────────────────────────")
    print(extract_text_basic(sample))

    print("\n── TEXT + METADATA ─────────────────────────────────────")
    pprint.pprint(extract_with_metadata(sample))

    print("\n── extract_resume(metadata=False) ──────────────────────")
    pprint.pprint(extract_resume(sample, metadata=False))

    print("\n── extract_resume(metadata=True) ───────────────────────")
    pprint.pprint(extract_resume(sample, metadata=True))

    #Dhruv File