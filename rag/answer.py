"""Build a grounded prompt from retrieved chunks, and generate an answer.

This is the "A" and the "G" of RAG. The retrieval work is done; what is
left is to hand the model the passages and constrain it to use them.

The constraint is the whole point. Without it the model blends retrieved
text with its own recollection, and you lose the one property RAG buys:
knowing where the answer came from.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .chunker import Chunk

MODEL = "gemini-3.6-flash"

_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

NOT_FOUND = "I couldn't find that in the documents."

_PROMPT = """\
Answer the question using only the numbered sources below.

Rules:
- Use only what the sources say. Do not add outside knowledge.
- If the sources do not answer the question, reply exactly: {not_found}
- Cite the sources you used inline, like [1] or [2].
- Be brief. Do not restate the question.

Sources:
{context}

Question: {question}
Answer:"""


class AnswerError(Exception):
    """Raised when the generation API cannot be reached or returns an error."""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Assemble the prompt sent to the model. Pure — no network, no state."""
    context = "\n\n".join(
        f"[{n}] ({chunk.source}, p.{chunk.page})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    )
    return _PROMPT.format(not_found=NOT_FOUND, context=context, question=question)


def generate(prompt: str) -> str:
    """Send `prompt` to Gemini and return the text of its reply."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Near-zero temperature: we want the sources reported, not embellished.
        "generationConfig": {"temperature": 0.0},
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
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise AnswerError(f"generation request failed: {exc.read().decode()}") from exc
    except urllib.error.URLError as exc:
        raise AnswerError(f"could not reach the generation API: {exc.reason}") from exc

    return _text_of(body)


def _text_of(body: dict) -> str:
    """Pull the reply text out of Gemini's response envelope."""
    candidates = body.get("candidates")
    if not candidates:
        # No candidate at all usually means the prompt was blocked.
        raise AnswerError(f"no answer returned: {body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        reason = candidates[0].get("finishReason", "unknown")
        raise AnswerError(f"empty answer (finishReason: {reason})")

    return text


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise AnswerError(
            "GEMINI_API_KEY is not set. Create a key at "
            "https://aistudio.google.com/apikey and export it."
        )
    return key
