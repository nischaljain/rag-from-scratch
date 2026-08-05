# rag-from-scratch

A PDF question-answering CLI, built without a RAG framework to see what one
actually does.

Ask a question, get an answer drawn only from your documents, with the source
page cited. No LangChain, no LlamaIndex, no vector database. Every stage —
chunking, embedding, similarity search, prompt construction — is a readable
function in this repo.

**One dependency: `pypdf`.** PDF parsing is the only part that isn't
reasonably hand-writable. Embeddings and generation are HTTP calls to Gemini
made with `urllib` from the standard library; storage is a JSON file;
similarity search is a dot product over two square roots.

Total: seven modules, about 700 lines including comments.

## Setup

Python 3.9 or later.

```sh
python3 -m venv .venv
.venv/bin/pip install pypdf
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
(no credit card, ~1,500 requests/day). Then:

```sh
cp .env.example .env
# edit .env and paste the key
```

`.env` is gitignored. The CLI reads it automatically; a real `GEMINI_API_KEY`
environment variable takes precedence if one is set.

## Usage

Three commands. Build an index once, then query it.

### `index` — read PDFs, write `index.json`

```
$ python -m rag index handbook.pdf
handbook.pdf: 1 pages, 6 chunks
embedding 6 chunks...
wrote index.json (6 chunks, 64 KB)
```

Accepts multiple files. Chunks from all of them are embedded in one batched
call, which matters against a request-limited free tier.

### `ask` — answer a question from the index

```
$ python -m rag ask "how many holidays do I get?"
Employees accrue 20 days of paid annual leave per calendar year [3].

sources:
  [1] handbook.pdf p.1
  [2] handbook.pdf p.1
  [3] handbook.pdf p.1
```

When the documents don't contain the answer, it says so rather than guessing:

```
$ python -m rag ask "what is the parental leave policy?"
I couldn't find that in the documents.
```

### `search` — retrieval only, no generation

```
$ python -m rag search "how many holidays do I get" -k 3

1. 0.6723  handbook.pdf p.1  [handbook.pdf:p1:c1]
   at least two weeks in advance through the internal portal. Sick leave is...

2. 0.6522  handbook.pdf p.1  [handbook.pdf:p1:c2]
   than three consecutive working days. Section 2: Expenses Business travel...
```

This is the most useful command in the project and it costs one embedding
call. See [Debugging](#debugging).

### Options

```
--index PATH          index file to read or write (default: index.json)
                      goes before the subcommand: rag --index other.json ask "..."

index:   --chunk-size N   target characters per chunk (default: 1000)
         --overlap N      characters repeated between chunks (default: 200)
search:  -k N             chunks to return (default: 5)
ask:     -k N             chunks to put in the prompt (default: 5)
         --show-prompt    print the exact prompt sent to the model
```

## How it works

Two pipelines that share nothing but `index.json`.

```
INDEX (offline)   PDF → loader → cleaner → chunker → embedder → store.save()
                                                                     │
                                                                index.json
                                                                     │
ASK (per query)   question → embed_query → store.search() → build_prompt → generate
```

| Module | Input → output | Why it exists |
|---|---|---|
| `loader.py` | PDF → pages of text | The only module that knows the PDF format. Keeps page numbers, because this is the one moment they exist |
| `cleaner.py` | raw text → prose | Extraction returns page geometry as whitespace; on the sample PDF, 5,021 chars for 175 chars of actual text |
| `chunker.py` | prose → `Chunk` records | A whole document is too coarse to retrieve and too broad to embed well |
| `embedder.py` | text → 768-dim vectors | Turns meaning into geometry, so "find relevant" becomes "find nearby" |
| `store.py` | vectors → `index.json`, ranked search | Persistence, plus brute-force cosine similarity |
| `answer.py` | chunks → grounded answer | Passages aren't an answer. Constrains the model to use only what it was given |
| `cli.py` | modules → commands | The only file that prints |

### Why the retrieval works

The query *"how do I get my money back?"* shares no words with *"Customers may
request a refund within 30 days"* — not "refund", not "purchase", not "30".
Keyword search scores it zero. Embeddings rank it first, because they encode
meaning rather than spelling.

Documents and queries are embedded with different task types
(`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`). A stored passage states facts; a
question requests them. Telling the model which is which improves how well
questions find their answers, and getting it wrong produces no error — only
quietly worse results. That's why `embedder.py` exposes two functions rather
than one with a flag.

## Debugging

Wrong answers have two very different causes, and separating them is most of
the work.

**Did retrieval find the right passage?** `search` shows exactly what came
back, ranked, with no LLM involved. If the answer isn't in these results, no
prompt engineering will help.

**Did generation use it?** `ask --show-prompt` prints the exact string sent to
the model. If the passage is there and the answer is still wrong, retrieval was
never the problem.

Both failures happened while building this, on the same document:

- Retrieval ranked an expenses chunk above a leave chunk, because 1000-char
  chunks each spanned several topics and the embedding averaged them. Fixed by
  re-indexing at `--chunk-size 350`.
- Generation then refused a question whose answer was in the prompt verbatim.
  The instruction "do not add outside knowledge" was strict enough to also
  suppress mapping *holidays* to *annual leave*. The rule now separates facts,
  which must come from the sources, from wording, which need not match.

Neither bug was visible to the other command.

## Tuning

Two numbers matter most, both on `index`:

**`--chunk-size`** (default 1000). Too large and one embedding averages several
topics, so nothing ranks distinctly. Too small and an answer gets severed from
the context that qualifies it — a chunk reading "This applies only to
enterprise customers" is useless alone.

**`--overlap`** (default 200). Text repeated between adjacent chunks, so a
sentence landing on a boundary survives intact in one of them. Costs about 20%
extra storage.

Changing either requires re-indexing. `index.json` records which embedding
model built it, and `load()` refuses an index built by a different one —
vectors from two models aren't comparable, and mixing them fails silently
rather than loudly.

## Limits

Deliberate omissions, each a real technique left out to keep the base mechanism
visible:

- **Brute-force search.** Every stored vector is compared against the query.
  Exact, and fine into the tens of thousands of chunks. Vector databases exist
  because approximate methods win at millions.
- **No hybrid search.** Embeddings are weak at exact identifiers — product
  codes, error numbers, names — where keyword matching wins. Production systems
  combine both.
- **No reranking, no query rewriting, no conversation history.**
- **No incremental indexing.** Re-running `index` re-embeds everything.
- **No OCR.** Scanned PDFs contain images, not text; the loader detects this and
  says so rather than writing an empty index.
- **Vectors stored as JSON**, which is verbose — roughly 10 KB per chunk.

## The build log

[`conversation.jsonl`](conversation.jsonl) is the full record of building this,
one entry per exchange: the questions asked, the decisions made, the reasoning
behind them, and the two bugs above as they were found. It's the part that
doesn't survive in the finished code.
