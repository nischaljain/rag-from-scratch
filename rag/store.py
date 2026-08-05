"""Save chunks with their vectors, and search them.

This is the seam between the two pipelines: indexing writes the file,
asking reads it. Nothing else is shared.

Search is brute force — every stored vector is compared against the
query. That is exact and fast enough for thousands of chunks. Vector
databases exist because approximate methods win at millions; at this
scale they would only add a dependency and hide the arithmetic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .chunker import Chunk
from .embedder import DIMENSIONS, MODEL

DEFAULT_PATH = Path("index.json")


class StoreError(Exception):
    """Raised when an index cannot be read or does not match the current model."""


def save(chunks: list[Chunk], path: str | Path = DEFAULT_PATH) -> None:
    """Write `chunks` and their vectors to `path`."""
    missing = [c.id for c in chunks if c.embedding is None]
    if missing:
        raise StoreError(f"{len(missing)} chunk(s) have no embedding, e.g. {missing[0]}")

    index = {
        "embedding_model": MODEL,
        "dimensions": DIMENSIONS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [asdict(c) for c in chunks],
    }

    Path(path).write_text(json.dumps(index), encoding="utf-8")


def load(path: str | Path = DEFAULT_PATH) -> list[Chunk]:
    """Read chunks from `path`, refusing an index built by a different model."""
    path = Path(path)
    if not path.is_file():
        raise StoreError(f"no index at {path}. Run `index` first.")

    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreError(f"{path} is not valid JSON: {exc}") from exc

    # Vectors from different models are not comparable. Mixing them produces
    # no error and no crash — only quietly meaningless search results.
    if index.get("embedding_model") != MODEL:
        raise StoreError(
            f"{path} was built with {index.get('embedding_model')!r}, "
            f"but the embedder now uses {MODEL!r}. Rebuild the index."
        )

    return [Chunk(**c) for c in index["chunks"]]


def search(
    query_vector: list[float],
    chunks: list[Chunk],
    k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Return the `k` chunks closest to `query_vector`, best first."""
    scored = [(c, cosine_similarity(query_vector, c.embedding)) for c in chunks]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How closely two vectors point in the same direction: -1 to 1.

    Direction carries the meaning, not length, so this divides the dot
    product by both magnitudes. Two vectors about the same topic score
    near 1 whether the passages were long or short.
    """
    dot = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot / (magnitude_a * magnitude_b)
