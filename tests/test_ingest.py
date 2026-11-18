"""
Tests for the ingestion pipeline's ChromaDB storage behavior.

Run with: pytest tests/ -v

No Ollama required: embeddings come from a tiny fake embedder and
ChromaDB runs in memory.
"""

import uuid

import chromadb
import pytest

from src.ingest import (
    display_source,
    generate_chunk_id,
    ingest_file,
    prune_orphaned_sources,
    source_name_for,
)


class FakeEmbedder:
    """Deterministic stand-in for OllamaEmbedder. Vector length matches nothing real, and that's fine."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class ExplodingEmbedder:
    """Simulates Ollama dying mid-run."""

    def embed_documents(self, texts):
        raise RuntimeError("ollama went away")


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

def test_first_ingest_stores_all_chunks(kb, collection):
    note = kb / "note.md"
    _write_paragraphs(note, 6)

    n = ingest_file(note, collection, FakeEmbedder())

    assert n > 1
    assert collection.count() == n
    metas = _chunks_for(collection, "note.md")["metadatas"]
    assert len(metas) == n
    assert all(m["total_chunks"] == n for m in metas)


def test_reingesting_shorter_file_removes_stale_chunks(kb, collection):
    note = kb / "note.md"
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


def test_reingest_leaves_other_sources_alone(kb, collection):
    a = kb / "a.md"
    b = kb / "b.md"
    _write_paragraphs(a, 4)
    _write_paragraphs(b, 4)
    ingest_file(a, collection, FakeEmbedder())
    n_b = ingest_file(b, collection, FakeEmbedder())

    _write_paragraphs(a, 2)
    n_a = ingest_file(a, collection, FakeEmbedder())

    assert collection.count() == n_a + n_b
    assert len(_chunks_for(collection, "b.md")["ids"]) == n_b


def test_reingest_same_content_is_idempotent(kb, collection):
    note = kb / "note.md"
    _write_paragraphs(note, 5)
    n1 = ingest_file(note, collection, FakeEmbedder())
    n2 = ingest_file(note, collection, FakeEmbedder())

    assert n1 == n2
    assert collection.count() == n1


def test_file_that_became_empty_drops_its_chunks(kb, collection):
    note = kb / "note.md"
    _write_paragraphs(note, 4)
    ingest_file(note, collection, FakeEmbedder())

    note.write_text("")
    n = ingest_file(note, collection, FakeEmbedder())

    assert n == 0
    assert collection.count() == 0


def test_embedding_failure_keeps_previous_chunks(kb, collection):
    note = kb / "note.md"
    _write_paragraphs(note, 4)
    n = ingest_file(note, collection, FakeEmbedder())

    with pytest.raises(RuntimeError):
        ingest_file(note, collection, ExplodingEmbedder())

    assert collection.count() == n


def test_loader_failure_keeps_previous_chunks(kb, collection, monkeypatch):
    import src.ingest as ingest_module

    note = kb / "note.md"
    _write_paragraphs(note, 4)
    n = ingest_file(note, collection, FakeEmbedder())

    def boom(_path):
        raise ValueError("corrupt file")

    monkeypatch.setitem(ingest_module.LOADERS, ".md", boom)
    with pytest.raises(ValueError):
        ingest_file(note, collection, FakeEmbedder())

    assert collection.count() == n


# ──────────────────────────────────────────────────────────────
# Orphan pruning (files deleted or renamed on disk)
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def kb(tmp_path, monkeypatch):
    """Point the knowledge base root at tmp_path so sources get relative names."""
    import src.ingest as ingest_module
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_DIR", tmp_path)
    return tmp_path


def _ingest_all(directory, collection):
    files = sorted(directory.rglob("*.md"))
    for f in files:
        ingest_file(f, collection, FakeEmbedder())
    return {source_name_for(f) for f in files}


def test_prune_removes_deleted_file(kb, collection):
    keep = kb / "keep.md"
    gone = kb / "gone.md"
    _write_paragraphs(keep, 3)
    _write_paragraphs(gone, 3)
    _ingest_all(kb, collection)
    n_keep = len(_chunks_for(collection, "keep.md")["ids"])

    gone.unlink()
    live = _ingest_all(kb, collection)
    pruned = prune_orphaned_sources(collection, kb, [".md"], live)

    assert pruned == ["gone.md"]
    assert collection.count() == n_keep
    assert _chunks_for(collection, "gone.md")["ids"] == []


def test_prune_handles_rename_without_duplicating(kb, collection):
    old = kb / "old-notes.md"
    _write_paragraphs(old, 3)
    _ingest_all(kb, collection)

    old.rename(kb / "notes-2024.md")
    live = _ingest_all(kb, collection)
    pruned = prune_orphaned_sources(collection, kb, [".md"], live)

    assert pruned == ["old-notes.md"]
    n_new = len(_chunks_for(collection, "notes-2024.md")["ids"])
    assert n_new > 0
    assert collection.count() == n_new


def test_prune_scoped_to_subdirectory(kb, collection):
    docs = kb / "docs"
    elsewhere = kb / "elsewhere"
    docs.mkdir()
    elsewhere.mkdir()
    _write_paragraphs(docs / "a.md", 2)
    _write_paragraphs(elsewhere / "b.md", 2)
    _ingest_all(kb, collection)
    n_a = len(_chunks_for(collection, "docs/a.md")["ids"])
    n_b = len(_chunks_for(collection, "elsewhere/b.md")["ids"])

    # b.md is deleted, but this run only covers docs/, so it is out of scope
    (elsewhere / "b.md").unlink()
    live = _ingest_all(docs, collection)
    pruned = prune_orphaned_sources(collection, docs, [".md"], live)

    assert pruned == []
    assert collection.count() == n_a + n_b

    # A full knowledge-base run does catch it
    live = _ingest_all(kb, collection)
    pruned = prune_orphaned_sources(collection, kb, [".md"], live)
    assert pruned == ["elsewhere/b.md"]
    assert collection.count() == n_a


def test_prune_ignores_other_file_types(kb, collection):
    md = kb / "a.md"
    txt = kb / "b.txt"
    _write_paragraphs(md, 2)
    _write_paragraphs(txt, 2)
    ingest_file(md, collection, FakeEmbedder())
    ingest_file(txt, collection, FakeEmbedder())
    before = collection.count()

    txt.unlink()
    # Run scoped to .md only: the missing .txt is out of scope, so it stays
    pruned = prune_orphaned_sources(collection, kb, [".md"], {"a.md"})

    assert pruned == []
    assert collection.count() == before


def test_outside_files_get_absolute_source_names(kb, tmp_path_factory, collection):
    """Same file name inside and outside the KB must not collide."""
    external = tmp_path_factory.mktemp("external")
    _write_paragraphs(kb / "report.md", 2)
    _write_paragraphs(external / "report.md", 2)

    assert source_name_for(kb / "report.md") == "report.md"
    assert source_name_for(external / "report.md") == str(external / "report.md")

    n_in = ingest_file(kb / "report.md", collection, FakeEmbedder())
    n_out = ingest_file(external / "report.md", collection, FakeEmbedder())
    assert collection.count() == n_in + n_out


def test_prune_on_outside_directory_removes_its_own_orphans(kb, tmp_path_factory, collection):
    external = tmp_path_factory.mktemp("external")
    _write_paragraphs(external / "keep.md", 2)
    _write_paragraphs(external / "gone.md", 2)
    _ingest_all(external, collection)
    n_keep = len(_chunks_for(collection, str(external / "keep.md"))["ids"])

    (external / "gone.md").unlink()
    live = _ingest_all(external, collection)
    pruned = prune_orphaned_sources(collection, external, [".md"], live)

    assert pruned == [str(external / "gone.md")]
    assert collection.count() == n_keep


def test_kb_run_never_prunes_outside_sources(kb, tmp_path_factory, collection):
    external = tmp_path_factory.mktemp("external")
    _write_paragraphs(external / "ext.md", 2)
    ingest_file(external / "ext.md", collection, FakeEmbedder())
    before = collection.count()

    (external / "ext.md").unlink()
    # Even though the file is gone, it is outside the KB so a KB run leaves it alone
    pruned = prune_orphaned_sources(collection, kb, [".md"], set())

    assert pruned == []
    assert collection.count() == before


def test_kb_run_cleans_up_legacy_bare_name_outside_sources(kb, collection):
    """Chunks stored under a bare name with no matching KB file are stale leftovers from before absolute naming."""
    collection.upsert(
        ids=["legacy-0"],
        documents=["old text"],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source": "ext.md", "chunk_index": 0, "total_chunks": 1}],
    )

    pruned = prune_orphaned_sources(collection, kb, [".md"], set())

    assert pruned == ["ext.md"]
    assert collection.count() == 0


def test_prune_on_empty_collection_is_noop(kb, collection):
    assert prune_orphaned_sources(collection, kb, [".md"], set()) == []


def test_prune_handles_case_only_rename(kb, collection):
    """macOS reports Notes.md as existing when only notes.md does; prune must not be fooled."""
    old = kb / "Notes.md"
    _write_paragraphs(old, 3)
    _ingest_all(kb, collection)

    old.rename(kb / "notes.md")
    live = _ingest_all(kb, collection)
    pruned = prune_orphaned_sources(collection, kb, [".md"], live)

    assert pruned == ["Notes.md"]
    n_new = len(_chunks_for(collection, "notes.md")["ids"])
    assert collection.count() == n_new


def test_symlinked_subdirectory_keeps_kb_relative_name_and_survives_prune(kb, tmp_path_factory, collection):
    """A symlinked folder inside the KB is skipped by rglob; its chunks must not be treated as orphans."""
    real = tmp_path_factory.mktemp("real_notes")
    _write_paragraphs(real / "x.md", 2)
    (kb / "notes").symlink_to(real, target_is_directory=True)

    via_link = kb / "notes" / "x.md"
    assert source_name_for(via_link) == "notes/x.md"
    ingest_file(via_link, collection, FakeEmbedder())
    before = collection.count()

    # Default run: rglob doesn't descend into the symlink, so live_sources is empty
    pruned = prune_orphaned_sources(collection, kb, [".md"], set())

    assert pruned == []
    assert collection.count() == before


def test_prune_skips_unreadable_directory(kb, collection):
    private = kb / "private"
    private.mkdir()
    _write_paragraphs(private / "secret.md", 2)
    ingest_file(private / "secret.md", collection, FakeEmbedder())
    before = collection.count()

    private.chmod(0o000)
    try:
        pruned = prune_orphaned_sources(collection, kb, [".md"], set())
    finally:
        private.chmod(0o755)

    assert pruned == []
    assert collection.count() == before


# ──────────────────────────────────────────────────────────────
# Display names
# ──────────────────────────────────────────────────────────────

def test_display_source_keeps_kb_relative_names():
    assert display_source("notes/a.md") == "notes/a.md"


def test_display_source_shortens_home_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert display_source(str(tmp_path / "Downloads" / "a.md")) == "~/Downloads/a.md"


def test_display_source_leaves_other_absolute_paths_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert display_source("/opt/shared/a.md") == "/opt/shared/a.md"
