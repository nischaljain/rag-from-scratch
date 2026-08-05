"""Cut cleaned page text into retrievable passages.

Two forces pull in opposite directions:

  - Chunks must be small, so a search result points at the answer rather
    than at a whole document.
  - Chunks must be self-contained, because each one is retrieved alone,
    with no neighbours to give it context.

The compromise here is to pack whole paragraphs up to a size budget, and
to repeat a little text at the boundary so a sentence split across two
chunks still appears intact in one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Page

TARGET_CHARS = 1000
OVERLAP_CHARS = 200

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A passage of text, tagged with where it came from."""

    id: str  # e.g. "policy.pdf:p7:c12"
    text: str
    source: str
    page: int
    index: int  # position within the document, 0-based
    embedding: list[float] | None = None  # filled in by the embedder


def chunk_pages(
    pages: list[Page],
    source: str,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Split already-cleaned pages into chunks of roughly `target_chars`."""
    chunks: list[Chunk] = []

    for page in pages:
        for text in _split(page.text, target_chars, overlap_chars):
            chunks.append(
                Chunk(
                    id=f"{source}:p{page.number}:c{len(chunks)}",
                    text=text,
                    source=source,
                    page=page.number,
                    index=len(chunks),
                )
            )

    return chunks


def _split(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Pack paragraphs into chunks, carrying `overlap_chars` between them."""
    pieces = _paragraphs(text, target_chars)

    chunks: list[str] = []
    buffer = ""

    for piece in pieces:
        if buffer and len(buffer) + len(piece) + 2 > target_chars:
            chunks.append(buffer)
            buffer = _tail(buffer, overlap_chars)
        buffer = f"{buffer}\n\n{piece}".strip()

    if buffer:
        chunks.append(buffer)

    return chunks


def _paragraphs(text: str, target_chars: int) -> list[str]:
    """Paragraphs, with any that exceed the budget broken into sentences."""
    pieces: list[str] = []

    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            pieces.extend(s for s in _SENTENCE_END.split(para) if s)

    return pieces


def _tail(text: str, overlap_chars: int) -> str:
    """The last `overlap_chars` of `text`, trimmed to a word boundary."""
    if overlap_chars <= 0:
        return ""
    if len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail
