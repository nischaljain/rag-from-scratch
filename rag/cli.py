"""Command line entry point: index, search, ask.

This is the only file that prints. Every other module returns values and
lets this one decide how to show them — which is what keeps them usable
from a notebook, a test, or a web handler later.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .answer import AnswerError, build_prompt, generate
from .chunker import OVERLAP_CHARS, TARGET_CHARS, Chunk, chunk_pages
from .cleaner import clean
from .embedder import EmbedderError, embed_documents, embed_query
from .loader import LoaderError, Page, load_pdf
from .store import DEFAULT_PATH, StoreError, load, save, search

PipelineError = (LoaderError, EmbedderError, StoreError, AnswerError)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = _parse_args(argv)

    try:
        return args.run(args)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------
# commands


def _index(args: argparse.Namespace) -> int:
    """Run the offline pipeline over every PDF and write one index."""
    chunks: list[Chunk] = []

    for path in args.pdfs:
        pages = [Page(p.number, clean(p.text)) for p in load_pdf(path)]
        found = chunk_pages(
            pages,
            source=Path(path).name,
            target_chars=args.chunk_size,
            overlap_chars=args.overlap,
        )
        chunks.extend(found)
        print(f"{path}: {len(pages)} pages, {len(found)} chunks")

    if not chunks:
        print("nothing to index", file=sys.stderr)
        return 1

    # One call for every document, so batching works across files too.
    print(f"embedding {len(chunks)} chunks...")
    for chunk, vector in zip(chunks, embed_documents([c.text for c in chunks])):
        chunk.embedding = vector

    save(chunks, args.index)
    size_kb = Path(args.index).stat().st_size / 1024
    print(f"wrote {args.index} ({len(chunks)} chunks, {size_kb:.0f} KB)")
    return 0


def _search(args: argparse.Namespace) -> int:
    """Show what retrieval returns, without spending a generation call."""
    hits = search(embed_query(args.query), load(args.index), k=args.k)

    for rank, (chunk, score) in enumerate(hits, start=1):
        print(f"\n{rank}. {score:.4f}  {chunk.source} p.{chunk.page}  [{chunk.id}]")
        print(f"   {_preview(chunk.text)}")

    return 0


def _ask(args: argparse.Namespace) -> int:
    """Retrieve, then answer from what was retrieved."""
    hits = search(embed_query(args.question), load(args.index), k=args.k)
    chunks = [chunk for chunk, _ in hits]

    prompt = build_prompt(args.question, chunks)
    if args.show_prompt:
        print(prompt, "\n", "-" * 60, sep="")

    print(generate(prompt))
    print("\nsources:")
    for n, chunk in enumerate(chunks, start=1):
        print(f"  [{n}] {chunk.source} p.{chunk.page}")

    return 0


# --------------------------------------------------------------------------
# plumbing


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rag", description="Ask questions of PDFs.")
    parser.add_argument(
        "--index",
        default=str(DEFAULT_PATH),
        help=f"index file to read or write (default: {DEFAULT_PATH})",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_cmd = commands.add_parser("index", help="build an index from PDFs")
    index_cmd.add_argument("pdfs", nargs="+", help="PDF files to index")
    index_cmd.add_argument(
        "--chunk-size",
        type=int,
        default=TARGET_CHARS,
        help=f"target characters per chunk (default: {TARGET_CHARS})",
    )
    index_cmd.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP_CHARS,
        help=f"characters repeated between chunks (default: {OVERLAP_CHARS})",
    )
    index_cmd.set_defaults(run=_index)

    search_cmd = commands.add_parser("search", help="show retrieved chunks only")
    search_cmd.add_argument("query")
    search_cmd.add_argument("-k", type=int, default=5, help="how many chunks")
    search_cmd.set_defaults(run=_search)

    ask_cmd = commands.add_parser("ask", help="answer a question from the index")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("-k", type=int, default=5, help="how many chunks to use")
    ask_cmd.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the prompt sent to the model",
    )
    ask_cmd.set_defaults(run=_ask)

    return parser.parse_args(argv)


def _preview(text: str, width: int = 100) -> str:
    """One line of a chunk, for scanning search results."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width] + "..."


def _load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=value lines from .env, without overriding the real environment."""
    path = Path(path)
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    sys.exit(main())
