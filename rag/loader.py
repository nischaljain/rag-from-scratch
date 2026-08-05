"""Turn a PDF file into plain text, one entry per page.

This is the only module that knows what a PDF is. Everything downstream
works on strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    """One page of extracted text."""

    number: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str


class LoaderError(Exception):
    """Raised when a PDF cannot be turned into usable text."""


def load_pdf(path: str | Path) -> list[Page]:
    """Read `path` and return its pages as text.

    Pages that extract to nothing (images, blank pages) are dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"no such file: {path}")

    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise LoaderError(f"could not open {path.name}: {exc}") from exc

    if reader.is_encrypted:
        # An empty password unlocks many "protected" PDFs.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise LoaderError(f"{path.name} is password-protected") from exc

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Page(number=i, text=text))

    if not pages:
        raise LoaderError(
            f"{path.name} contains no extractable text. "
            "It is probably a scan, which would need OCR."
        )

    return pages
