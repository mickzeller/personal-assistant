"""
Tests for the ingestion pipeline's ChromaDB storage behavior.

Run with: pytest tests/ -v

No Ollama required: embeddings come from a tiny fake embedder and
ChromaDB runs in memory.
"""

import uuid

import chromadb
import pytest

from src.ingest import generate_chunk_id, ingest_file


class FakeEmbedder:
    """Deterministic stand-in for OllamaEmbedder. Vector length matches nothing real, and that's fine."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


@pytest.fixture
def collection():
    # EphemeralClient shares one in-process instance across tests, so each
    # test needs its own collection name to avoid "already exists" errors.
    client = chromadb.EphemeralClient()
    return client.create_collection(
        name=f"test_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )


def _write_paragraphs(path, n):
    """Write n paragraphs long enough that each lands in its own chunk (chunk_size=512)."""
    path.write_text("\n\n".join(f"Paragraph {i}. " + ("word " * 90) for i in range(n)))


def _chunks_for(collection, source):
    return collection.get(where={"source": source}, include=["metadatas", "documents"])


# ──────────────────────────────────────────────────────────────
# Re-ingest behavior (GitHub issue #2)
# ──────────────────────────────────────────────────────────────

def test_first_ingest_stores_all_chunks(tmp_path, collection):
    note = tmp_path / "note.md"
    _write_paragraphs(note, 6)

    n = ingest_file(note, collection, FakeEmbedder())

    assert n > 1
    assert collection.count() == n
    metas = _chunks_for(collection, "note.md")["metadatas"]
    assert all(m["total_chunks"] == n for m in metas)


def test_reingesting_shorter_file_removes_stale_chunks(tmp_path, collection):
    note = tmp_path / "note.md"
    _write_paragraphs(note, 10)
    n_before = ingest_file(note, collection, FakeEmbedder())

    _write_paragraphs(note, 3)
    n_after = ingest_file(note, collection, FakeEmbedder())

    assert n_after < n_before
    assert collection.count() == n_after

    result = _chunks_for(collection, "note.md")
    indices = sorted(m["chunk_index"] for m in result["metadatas"])
    assert indices == list(range(n_after))
    assert all(m["total_chunks"] == n_after for m in result["metadatas"])
    # The old tail chunks are gone, not just hidden
    stale_id = generate_chunk_id("note.md", n_before - 1)
    assert collection.get(ids=[stale_id])["ids"] == []


def test_reingest_leaves_other_sources_alone(tmp_path, collection):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    _write_paragraphs(a, 4)
    _write_paragraphs(b, 4)
    ingest_file(a, collection, FakeEmbedder())
    n_b = ingest_file(b, collection, FakeEmbedder())

    _write_paragraphs(a, 2)
    n_a = ingest_file(a, collection, FakeEmbedder())

    assert collection.count() == n_a + n_b
    assert len(_chunks_for(collection, "b.md")["ids"]) == n_b


def test_reingest_same_content_is_idempotent(tmp_path, collection):
    note = tmp_path / "note.md"
    _write_paragraphs(note, 5)
    n1 = ingest_file(note, collection, FakeEmbedder())
    n2 = ingest_file(note, collection, FakeEmbedder())

    assert n1 == n2
    assert collection.count() == n1
