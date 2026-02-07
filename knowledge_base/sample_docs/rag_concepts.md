# Understanding RAG (Retrieval Augmented Generation)

## What is RAG?

RAG is a technique that improves LLM responses by giving the model access to external knowledge at query time. Instead of relying solely on what the model "memorized" during training, we retrieve relevant documents and include them in the prompt.

Think of it like an open-book exam vs. a closed-book exam. The LLM's training data is the "closed book" — whatever it learned. RAG gives it an "open book" of your personal documents, current information, and domain-specific knowledge.

## Why RAG Instead of Fine-Tuning?

| Approach | Best For | Updates | Cost |
|----------|----------|---------|------|
| Fine-tuning | Teaching behavior, style, and reasoning patterns | Requires retraining | High (GPU time) |
| RAG | Adding factual knowledge, personal docs, current info | Just update the database | Low (embed + store) |

The golden rule: **Fine-tune for HOW the model responds. Use RAG for WHAT it knows.**

## The RAG Pipeline

### Step 1: Ingestion (Offline, One-Time Per Document)

1. **Load** your documents (PDFs, markdown, text files, etc.)
2. **Chunk** them into smaller pieces (typically 256-1024 tokens each)
3. **Embed** each chunk — convert text to a numerical vector using an embedding model
4. **Store** the vectors in a vector database (ChromaDB, FAISS, Pinecone, etc.)

### Step 2: Retrieval (Online, Every Query)

1. **Embed** the user's question using the same embedding model
2. **Search** the vector database for chunks most similar to the question
3. **Rerank** (optional but recommended) — use a cross-encoder to re-score the candidates
4. **Return** the top-k most relevant chunks

### Step 3: Generation (Online, Every Query)

1. **Inject** the retrieved chunks into the prompt as context
2. **Generate** a response using the LLM
3. The model can now reference specific facts from your documents

## Vector Similarity

At the heart of RAG is the idea that similar meanings produce similar vectors. When you embed "What's the capital of France?" and a chunk containing "Paris is the capital of France", their vectors will be close together in the embedding space.

The distance metric (usually cosine similarity) measures how "aligned" two vectors are:
- **1.0** = identical direction (very similar meaning)
- **0.0** = perpendicular (unrelated)
- **-1.0** = opposite direction (opposite meaning)

## Common RAG Pitfalls

### 1. Chunks Too Large
If your chunks are too big, embeddings become vague averages of mixed content. The vector represents "a little bit of everything" rather than one specific idea.

### 2. Chunks Too Small
If chunks are too small, they lack context. A chunk that says "It increased by 40%" is useless without knowing what "it" refers to.

### 3. No Reranking
Embedding similarity alone misses some relevant results. A cross-encoder reranker dramatically improves precision by reading query + document together.

### 4. Ignoring Metadata
Metadata filters (source type, date, topic) can eliminate irrelevant results before semantic search even runs. If the user asks about "meeting notes from January", filter by date first, then search.

## Embedding Models

Popular open-source embedding models:

- **nomic-embed-text** — excellent quality, supports task prefixes, 768 dimensions
- **all-MiniLM-L6-v2** — fast, lightweight, 384 dimensions, good for getting started
- **BGE-large-en-v1.5** — high quality, 1024 dimensions, more compute required
- **E5-large-v2** — strong on benchmarks, uses query/passage prefixes similar to Nomic

For a personal knowledge base, nomic-embed-text is an excellent choice — high quality, moderate size, and works great with Ollama.
