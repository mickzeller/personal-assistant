# RAG Pipeline: How the Knowledge Base Works

How documents become searchable context, and how a question becomes an answer
grounded in my own notes. Config values live in `configs/config.py`; this doc
covers the reasoning behind them.

## Overview

Two separate flows: ingestion (one-time, per document) and retrieval (every question).

```
INGESTION (src/ingest.py)

  Your files          Load text       Split into      Embed each       Store in
  (.md, .txt,    ──►  from file  ──►  chunks     ──►  chunk       ──►  ChromaDB
   .pdf, .docx,                       (~512 chars)    (nomic-embed)
   .json, .jsonl)

RETRIEVAL (src/retriever.py + src/chat.py)

  Question ──► Embed ──► ChromaDB search ──► Rerank ──► Inject into ──► Ollama
                          (top 10, fast)     (top 5,     prompt          generates
                                             precise)
```

Each stored chunk carries three things: the text itself (what the LLM reads),
an embedding vector (what ChromaDB searches by), and metadata (source file,
chunk index, ingest date).

## Why chunking matters

If you embed an entire 50-page PDF as one vector, the embedding becomes a vague
"average" of everything in the document. It's too generic to match any specific
question. Chunking into ~512 character pieces means each vector represents one
focused idea, which makes retrieval much more precise.

The tradeoffs:

| Chunk size | Problem                                                   |
|------------|-----------------------------------------------------------|
| Too big    | Mixed topics per chunk, retrieval gets noisy              |
| Too small  | Chunks lose context, model can't make sense of them alone |
| Sweet spot | One coherent idea with enough context to stand on its own |

The 50-char overlap keeps sentences from getting cut in half at chunk
boundaries, which would confuse both the embedding model and the LLM. The
splitter also tries separators in order of preference (paragraphs, then lines,
then sentences, then words) so chunks break at natural reading boundaries.

## How embeddings work

An embedding model reads text and produces a fixed-size list of numbers (a
vector) that captures the *meaning* of the text. Similar meaning, similar vectors:

```
"The cat sat on the mat"   →  [0.12, -0.45, 0.78, ...]
"A kitten rested on a rug" →  [0.11, -0.43, 0.76, ...]   (close!)
"Stock markets crashed"    →  [-0.89, 0.22, -0.11, ...]  (way off)
```

That's the whole trick behind semantic search: turn the question into a vector,
find the stored vectors nearest to it.

Two implementation details worth knowing:

- **Everything runs locally through Ollama** (nomic-embed-text). No API calls,
  no cost, and my personal notes never leave the machine. We could use
  sentence-transformers directly, but Ollama is already running and handles
  model management, so it's one less thing to configure.
- **nomic-embed-text uses task prefixes.** Documents get embedded as
  `search_document: <text>` and questions as `search_query: <text>`. The model
  was trained to match queries to documents using these prefixes, so skipping
  them quietly hurts retrieval quality. This bit me once, hence the callout.

## Why retrieval has two stages

Stage 1 (embedding search) is like judging a book by its cover: fast but misses
nuance. Stage 2 (the cross-encoder reranker) actually reads the content: slower
but much more accurate.

| Stage              | What happens                                                                | Speed     |
|--------------------|-----------------------------------------------------------------------------|-----------|
| 1. Broad retrieval | Embed the question, pull top 10 nearest chunks from ChromaDB                | ~5-20ms   |
| 2. Reranking       | Cross-encoder reads question + each chunk *together*, rescores, keeps top 5 | ~50-200ms |

Concrete example. Ask "How do I handle errors in Python?" and embedding search
might return a chunk about Python error types (great) AND a chunk about Python
virtual environments (has the word "Python", totally irrelevant). The reranker
reads both chunks alongside the question and correctly scores the error
handling chunk way higher.

The combination gives us the speed of embeddings with the accuracy of a model
that actually understands the question-document relationship. The ~100ms is
worth it every time.

## ChromaDB storage details

- Local SQLite on disk (`data/chroma_db/`), persists across restarts, cosine
  similarity for search.
- **Chunk IDs are deterministic** (hash of source file + chunk index), so
  re-ingesting the same file upserts instead of creating duplicates. You can
  re-run ingestion whenever without worrying about it.
- **Batching:** ChromaDB caps batches at ~5,461 items, and a big Claude export
  can produce 7,000+ chunks, so ingestion processes 500 at a time.

## The chat flow

`src/chat.py` ties it together. Retrieved chunks get injected into the current
user message via a template, each labeled with its source file so the model can
cite where an answer came from. One subtle detail: conversation history stores
the *original* question, not the context-injected version, so history stays
clean across turns.

Deciding *whether* to retrieve is currently just the `/rag on|off` toggle. A
classifier that skips retrieval for small talk would be a nice upgrade someday.

## Further reading

- [ChromaDB docs](https://docs.trychroma.com/)
- [nomic-embed-text and task prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Cross-encoders vs bi-encoders (SBERT)](https://www.sbert.net/examples/applications/cross-encoder/README.html)
