"""
RAG retrieval — finds the most relevant chunks for a question.

Two stages: embed the query and grab top_k candidates from ChromaDB
(fast), then rerank with a cross-encoder that reads query + chunk
together. The reranker is ~100ms slower but catches relevance that
embedding similarity misses.
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
    """One retrieved chunk with its source and scores."""
    text: str
    source: str
    chunk_index: int
    distance: float                    # cosine distance (lower = more similar)
    rerank_score: float | None = None  # cross-encoder score (higher = better), None if not reranked


class Retriever:
    """Two-stage retriever: semantic search + cross-encoder reranking."""

    def __init__(self):
        self.embedder = OllamaEmbedder()
        self.collection = get_chroma_collection()

        # ~80MB download the first time, cached after that
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
            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self.reranker.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)

            chunks.sort(key=lambda c: c.rerank_score, reverse=True)
            chunks = chunks[:rerank_top_k]
        else:
            chunks.sort(key=lambda c: c.distance)
            chunks = chunks[:rerank_top_k]

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
        for idx, chk in enumerate(chunks, 1):
            score_str = ""
            if chk.rerank_score is not None:
                score_str = f" (relevance: {chk.rerank_score:.2f})"

            context_parts.append(
                f"[Source: {chk.source} | Chunk {chk.chunk_index}]{score_str}\n"
                f"{chk.text}"
            )

        return "\n\n---\n\n".join(context_parts)


# ──────────────────────────────────────────────────────────────
# Quick test — run directly to try a search
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
