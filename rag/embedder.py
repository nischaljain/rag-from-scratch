"""Turn text into vectors using the Gemini embedding API.

An embedding is a fixed-length list of numbers positioned so that texts
with similar meaning land near each other. Once text is vectors, "find
relevant passages" becomes "find nearby points" — a geometry problem.

Documents and queries are embedded with *different* task types. A stored
passage and the question that should find it are not the same kind of
text, and telling the model which is which measurably improves results.
That asymmetry is easy to get wrong, so callers use embed_documents() or
embed_query() rather than passing a task type by hand.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 100  # texts per HTTP request

_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents"
)

Vector = list[float]


class EmbedderError(Exception):
    """Raised when the embedding API cannot be reached or returns an error."""


def embed_documents(texts: list[str]) -> list[Vector]:
    """Embed passages that will be stored and searched over."""
    return _embed(texts, task="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> Vector:
    """Embed a single question being asked of the index."""
    return _embed([text], task="RETRIEVAL_QUERY")[0]


def _embed(texts: list[str], task: str) -> list[Vector]:
    """Embed `texts`, one HTTP request per BATCH_SIZE of them."""
    if not texts:
        return []

    vectors: list[Vector] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(_request(batch, task))

    return vectors


def _request(batch: list[str], task: str) -> list[Vector]:
    """Send one batch to the API and return its vectors, in order."""
    payload = {
        "requests": [
            {
                "model": f"models/{MODEL}",
                "content": {"parts": [{"text": text}]},
                "taskType": task,
                "outputDimensionality": DIMENSIONS,
            }
            for text in batch
        ]
    }

    request = urllib.request.Request(
        _ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": _api_key(),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        # The API explains what was wrong in the response body, not the status.
        raise EmbedderError(f"embedding request failed: {exc.read().decode()}") from exc
    except urllib.error.URLError as exc:
        raise EmbedderError(f"could not reach the embedding API: {exc.reason}") from exc

    embeddings = body.get("embeddings")
    if embeddings is None or len(embeddings) != len(batch):
        raise EmbedderError(f"unexpected response shape: {body}")

    return [e["values"] for e in embeddings]


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EmbedderError(
            "GEMINI_API_KEY is not set. Create a key at "
            "https://aistudio.google.com/apikey and export it."
        )
    return key
