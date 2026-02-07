"""
RAG Retrieval Module — finds the most relevant chunks for a user's question.

THE TWO-STAGE PIPELINE:
───────────────────────
Stage 1 — BROAD RETRIEVAL (fast, approximate)
    Embed the user's question → search ChromaDB for similar vectors
    Returns top_k candidates (default: 10)
    Speed: ~5-20ms

Stage 2 — RERANKING (slow, precise)
    Feed each candidate + the question into a cross-encoder model
    The cross-encoder reads BOTH texts together (not just their embeddings)
    and produces a relevance score
    Returns the top rerank_top_k results (default: 5)
    Speed: ~50-200ms

WHY TWO STAGES?
───────────────
Embedding similarity (Stage 1) is like judging a book by its cover —
fast but misses nuance. The cross-encoder (Stage 2) actually reads
the content — slower but much more accurate.

Example: Query = "How do I handle errors in Python?"
    - Embedding search might return a chunk about "Python error types" (great)
      AND a chunk about "Python virtual environments" (has the word "Python" but irrelevant)
    - The reranker reads both chunks with the question and correctly scores
      the error handling chunk much higher than the venv chunk.

The combination gives us the speed of embeddings with the accuracy of
a model that actually understands the question-document relationship.
"""

from dataclasses import dataclass

from rich.console import Console
from sentence_transformers import CrossEncoder

from configs.config import retrieval_config
from src.embeddings import OllamaEmbedder
from src.ingest import get_chroma_collection

console = Console()


@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with its metadata and relevance scores.

    Attributes:
        text:            The actual text content of the chunk
        source:          Which file this chunk came from
        chunk_index:     Position within the original document
        distance:        Raw cosine distance from ChromaDB (lower = more similar)
        rerank_score:    Score from the cross-encoder (higher = more relevant)
                         None if reranking wasn't applied
    """
    text: str
    source: str
    chunk_index: int
    distance: float
    rerank_score: float | None = None


class Retriever:
    """
    Two-stage retriever: semantic search + cross-encoder reranking.

    Usage:
        retriever = Retriever()
        chunks = retriever.search("How do I use type hints in Python?")

        for chunk in chunks:
            print(f"[{chunk.source}] Score: {chunk.rerank_score:.3f}")
            print(chunk.text)
    """

    def __init__(self):
        self.embedder = OllamaEmbedder()
        self.collection = get_chroma_collection()

        # Load the cross-encoder reranker
        # This downloads ~80MB the first time, then caches locally
        console.print("[dim]Loading reranker model...[/dim]", end=" ")
        self.reranker = CrossEncoder(
            retrieval_config.reranker_model,
            max_length=512,  # Truncate very long inputs
        )
        console.print("[dim]done.[/dim]")

    def search(
        self,
        query: str,
        top_k: int = None,
        rerank_top_k: int = None,
        source_filter: str = None,
        use_reranking: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Search the knowledge base for chunks relevant to the query.

        Args:
            query:          The user's question
            top_k:          How many candidates to retrieve from ChromaDB
            rerank_top_k:   How many to keep after reranking
            source_filter:  Only search chunks from this source file
            use_reranking:  Set False to skip reranking (faster, less accurate)

        Returns:
            List of RetrievedChunk objects, sorted by relevance (best first)
        """
        top_k = top_k or retrieval_config.top_k
        rerank_top_k = rerank_top_k or retrieval_config.rerank_top_k

        # ── Stage 1: Semantic Search ──────────────────────────────
        query_embedding = self.embedder.embed_query(query)

        # Build ChromaDB query (with optional source filter)
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if source_filter:
            query_params["where"] = {"source": source_filter}

        results = self.collection.query(**query_params)

        # Unpack results into RetrievedChunk objects
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                distance=dist,
            ))

        if not chunks:
            return []

        # ── Stage 2: Reranking ────────────────────────────────────
        if use_reranking and len(chunks) > 1:
            # Cross-encoder scores query-document pairs
            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self.reranker.predict(pairs)

            # Attach scores and sort by reranker score (highest first)
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)

            chunks.sort(key=lambda c: c.rerank_score, reverse=True)
            chunks = chunks[:rerank_top_k]
        else:
            # Without reranking, sort by distance (lowest first = most similar)
            chunks.sort(key=lambda c: c.distance)
            chunks = chunks[:rerank_top_k]

        # ── Optional: Filter by minimum relevance ─────────────────
        if retrieval_config.min_relevance_score > 0 and use_reranking:
            chunks = [
                c for c in chunks
                if c.rerank_score and c.rerank_score >= retrieval_config.min_relevance_score
            ]

        return chunks

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.

        Each chunk is labeled with its source so the LLM can cite where
        information came from. This is important for transparency —
        you want to know WHICH document the answer is based on.
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            score_str = ""
            if chunk.rerank_score is not None:
                score_str = f" (relevance: {chunk.rerank_score:.2f})"

            context_parts.append(
                f"[Source: {chunk.source} | Chunk {chunk.chunk_index}]{score_str}\n"
                f"{chunk.text}"
            )

        return "\n\n---\n\n".join(context_parts)


# ──────────────────────────────────────────────────────────────
# Quick test — run this file directly to try a search
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How does RAG retrieval work?"

    console.print(f"\n[bold]Searching for:[/bold] {query}\n")

    retriever = Retriever()
    results = retriever.search(query)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
    else:
        for i, chunk in enumerate(results, 1):
            score = f"{chunk.rerank_score:.3f}" if chunk.rerank_score else "N/A"
            console.print(f"[bold cyan]Result {i}[/bold cyan] — {chunk.source} "
                          f"(score: {score}, distance: {chunk.distance:.3f})")
            console.print(f"  {chunk.text[:200]}...")
            console.print()
