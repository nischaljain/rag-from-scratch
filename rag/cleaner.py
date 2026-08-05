"""Normalise text extracted from a PDF.

PDFs store glyph positions, not sentences, so extracted text arrives full
of layout artefacts: padding spaces, words broken across lines, ligatures.
This module turns that back into ordinary prose.

Paragraph breaks (blank lines) are preserved deliberately — the chunker
splits on them.
"""

from __future__ import annotations

import re
import unicodedata

# A word broken across a line break: "informa-\ntion" -> "information"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    """Return `text` with PDF layout artefacts removed."""
    # NFKC folds ligatures (ﬁ -> fi), non-breaking spaces, and other
    # typographic variants down to their plain equivalents.
    text = unicodedata.normalize("NFKC", text)

    # Soft hyphens are invisible line-break hints; they corrupt word matching.
    text = text.replace("\xad", "")

    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # Three or more newlines means padding; two means a paragraph break.
    text = _BLANK_LINES.sub("\n\n", text)

    return text.strip()
