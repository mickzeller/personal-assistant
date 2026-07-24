"""
Document ingestion — load, chunk, embed, store in ChromaDB.

Each chunk gets stored with:
  1. The text itself (what the LLM will read)
  2. An embedding vector (what ChromaDB searches by)
  3. Metadata (source file, page number, date, etc.)

WHY CHUNKING MATTERS:
─────────────────────
If you embed an entire 50-page PDF as one vector, the embedding becomes
a vague "average" of everything in the document — too generic to match
specific questions. By chunking into ~512 character pieces, each vector
represents one focused idea, making retrieval much more precise.

The overlap (50 chars) prevents us from cutting a sentence in half at a
chunk boundary, which would confuse both the embedding model and the LLM.
"""

import hashlib
from pathlib import Path
from datetime import datetime

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from configs.config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_DIR,
    chunking_config,
    retrieval_config,
)
from src.embeddings import OllamaEmbedder

console = Console()


# ──────────────────────────────────────────────────────────────
# Document Loaders — one function per file type
# ──────────────────────────────────────────────────────────────

def load_text_file(file_path: Path) -> str:
    """Load plain text or markdown files."""
    return file_path.read_text(encoding="utf-8")


def load_json_file(file_path: Path) -> str:
    """
    Smart JSON loader that auto-detects the format and routes to the right parser.

    Handles:
      1. Claude export files (conversations.json, memories.json, projects.json)
      2. Simple chat logs: [{"role": "user", "content": "..."}]
      3. Wrapped chat logs: {"messages": [...]}
      4. JSONL (one JSON object per line)
      5. Fallback: pretty-prints as readable text

    Files named "users.json" are SKIPPED — they contain PII (email, phone).
    """
    import json

    # ── Safety: skip files that are likely PII ────────────────────
    if file_path.name.lower() == "users.json":
        console.print(f"  [yellow]⚠ Skipping {file_path.name} — likely contains personal info (email, phone)[/yellow]")
        return ""

    raw = file_path.read_text(encoding="utf-8")

    # Try JSONL first (one object per line)
    lines = raw.strip().split("\n")
    if len(lines) > 1:
        try:
            records = [json.loads(line) for line in lines if line.strip()]
            return _format_chat_records(records)
        except json.JSONDecodeError:
            pass  # not JSONL, try standard JSON

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    # Claude export formats, detected by their signature keys
    if isinstance(data, list) and data:
        first = data[0] if isinstance(data[0], dict) else {}

        if "chat_messages" in first:
            return _format_claude_conversations(data)

        if "conversations_memory" in first:
            return _format_claude_memories(data)

        if "docs" in first and "prompt_template" in first:
            return _format_claude_projects(data)

    # Generic chat log formats
    if isinstance(data, list):
        return _format_chat_records(data)

    if isinstance(data, dict):
        for key in ("messages", "conversation", "chat", "turns", "data"):
            if key in data and isinstance(data[key], list):
                return _format_chat_records(data[key])

    # Fallback: pretty-print
    return json.dumps(data, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
# Claude Export Formatters
# ──────────────────────────────────────────────────────────────

def _format_claude_conversations(conversations: list[dict]) -> str:
    """
    Format Claude's conversations.json export into readable text.

    Each convo becomes a titled block with its summary (dense and great
    for retrieval) plus the messages. Assistant "thinking out loud"
    preambles get filtered — that's process, not knowledge, and it
    would pollute the embeddings.
    """
    parts = []

    thinking_prefixes = (
        "the user ", "let me ", "i need to ", "i should ", "i'll ",
        "okay, ", "alright, ", "hmm", "so the user ", "this is ",
        "looking at ", "based on ",
    )

    for convo in conversations:
        if not isinstance(convo, dict):
            continue

        title = convo.get("name", "Untitled Conversation")
        summary = convo.get("summary", "")
        messages = convo.get("chat_messages", [])

        if not messages and not summary:
            continue

        convo_parts = [f"## {title}"]
        if summary:
            convo_parts.append(f"\nSummary: {summary}")

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            sender = msg.get("sender", "unknown")
            text = msg.get("text", "").strip()

            if not text:
                continue

            if sender == "human":
                label = "User"
            elif sender == "assistant":
                label = "Assistant"

                # Drop leading chain-of-thought lines ("The user wants to...")
                # that leaked into the export — the answer below is what we want
                text_lines = text.split("\n")
                filtered_lines = []
                skip_thinking = True

                for line in text_lines:
                    stripped = line.strip().lower()
                    if skip_thinking and stripped and any(stripped.startswith(p) for p in thinking_prefixes):
                        continue
                    else:
                        skip_thinking = False
                        filtered_lines.append(line)

                text = "\n".join(filtered_lines).strip()
                if not text:
                    continue
            else:
                label = sender.capitalize()

            convo_parts.append(f"\n{label}: {text}")

        if len(convo_parts) > 1:  # more than just the title
            parts.append("\n".join(convo_parts))

    return "\n\n---\n\n".join(parts)


def _format_claude_memories(memories: list[dict]) -> str:
    """
    Format Claude's memories.json export. The conversations_memory field
    is pure gold for a personal assistant — it's Claude's accumulated
    understanding of who you are and what you work on.
    """
    parts = []

    for mem_block in memories:
        if not isinstance(mem_block, dict):
            continue

        convo_memory = mem_block.get("conversations_memory", "").strip()
        if convo_memory:
            parts.append(f"## Personal Context (Claude's Memory)\n\n{convo_memory}")

        project_memories = mem_block.get("project_memories", [])
        for proj in project_memories:
            if not isinstance(proj, dict):
                continue

            proj_title = proj.get("project_title", "Unknown Project")
            proj_mems = proj.get("memories", [])

            if not proj_mems:
                continue

            mem_texts = []
            for m in proj_mems:
                if isinstance(m, dict):
                    mem_text = m.get("memory", "").strip()
                    if mem_text:
                        mem_texts.append(f"- {mem_text}")

            if mem_texts:
                parts.append(
                    f"## Project: {proj_title}\n\n" + "\n".join(mem_texts)
                )

    return "\n\n---\n\n".join(parts)


def _format_claude_projects(projects: list[dict]) -> str:
    """
    Format Claude's projects.json export — the docs are usually guides
    and reference material, great knowledge base content.
    """
    parts = []

    for proj in projects:
        if not isinstance(proj, dict):
            continue

        name = proj.get("name", "Untitled Project")
        description = proj.get("description", "").strip()
        prompt = proj.get("prompt_template", "").strip()
        docs = proj.get("docs", [])

        proj_parts = [f"## Project: {name}"]

        if description:
            proj_parts.append(f"\nDescription: {description}")

        if prompt:
            proj_parts.append(f"\nProject Instructions:\n{prompt}")

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            filename = doc.get("filename", "unknown")
            content = doc.get("content", "").strip()
            if content:
                proj_parts.append(f"\n### Document: {filename}\n\n{content}")

        if len(proj_parts) > 1:
            parts.append("\n".join(proj_parts))

    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Generic Chat Log Formatter
# ──────────────────────────────────────────────────────────────

def _format_chat_records(records: list[dict]) -> str:
    """
    Turn a list of chat message dicts into readable "Speaker: text" lines,
    handling the common field name variations.
    """
    role_keys = ("role", "sender", "from", "author", "speaker")
    content_keys = ("content", "text", "message", "body", "value")

    parts = []
    for record in records:
        if not isinstance(record, dict):
            continue

        role = "Unknown"
        for key in role_keys:
            if key in record:
                role = str(record[key]).strip().capitalize()
                break

        content = ""
        for key in content_keys:
            if key in record and isinstance(record[key], str):
                content = record[key].strip()
                break

        if content:
            parts.append(f"{role}: {content}")

    return "\n\n".join(parts)


def load_pdf_file(file_path: Path) -> str:
    """Load PDF text with PyMuPDF. Complex layouts (columns, tables) may get messy."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(file_path)) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text}")
    return "\n\n".join(pages)


def load_docx_file(file_path: Path) -> str:
    """Load Word doc body text (no tables, images, or headers — don't need them)."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


LOADERS = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".markdown": load_text_file,
    ".pdf": load_pdf_file,
    ".docx": load_docx_file,
    ".json": load_json_file,
    ".jsonl": load_json_file,
}


# ──────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks using LangChain's recursive splitter.

    The splitter tries to break on paragraph boundaries first (\n\n),
    then line breaks (\n), then sentences (". "), then words (" "),
    and finally characters as a last resort. This preserves the most
    natural reading units possible.

    Settings are pulled from configs/config.py — tweak chunk_size
    and chunk_overlap there to experiment.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
        separators=chunking_config.separators,
        length_function=len,
    )
    return splitter.split_text(text)


# ──────────────────────────────────────────────────────────────
# ChromaDB Storage
# ──────────────────────────────────────────────────────────────

def get_chroma_collection() -> chromadb.Collection:
    """
    Get (or create) our ChromaDB collection.

    ChromaDB stores data in a local SQLite database on disk. Each
    "collection" is like a table. We use one called "personal_knowledge"
    for all our documents. The data persists across restarts.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=retrieval_config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def generate_chunk_id(source: str, chunk_index: int) -> str:
    """
    Deterministic ID (hash of source + index) so re-ingesting the same
    file upserts instead of creating duplicates.
    """
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────
# Main Ingestion Pipeline
# ──────────────────────────────────────────────────────────────

def ingest_file(
    file_path: Path,
    collection: chromadb.Collection,
    embedder: OllamaEmbedder,
) -> int:
    """
    Ingest one file: load → chunk → embed → store.
    Returns the number of chunks created.
    """
    suffix = file_path.suffix.lower()

    if suffix not in LOADERS:
        console.print(f"  [yellow]Skipping unsupported file type: {file_path.name}[/yellow]")
        return 0

    # Step 1: Load
    loader = LOADERS[suffix]
    text = loader(file_path)

    if not text.strip():
        console.print(f"  [yellow]Skipping empty file: {file_path.name}[/yellow]")
        return 0

    # Step 2: Chunk
    chunks = chunk_text(text)

    if not chunks:
        return 0

    # Step 3 & 4: Embed and store in batches
    # ChromaDB has a max batch size (~5,461). Large files like a 331-conversation
    # Claude export can produce 7,000+ chunks, so we process in batches.
    try:
        source_name = str(file_path.resolve().relative_to(KNOWLEDGE_BASE_DIR.resolve()))
    except ValueError:
        # File lives outside the knowledge base dir
        source_name = file_path.name
    batch_size = 500  # well under ChromaDB's limit, keeps memory reasonable
    total = len(chunks)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_chunks = chunks[batch_start:batch_end]

        # Embed this batch
        batch_embeddings = embedder.embed_documents(batch_chunks)

        # Build IDs and metadata for this batch
        batch_ids = [
            generate_chunk_id(source_name, i)
            for i in range(batch_start, batch_end)
        ]
        batch_metadatas = [
            {
                "source": source_name,
                "file_type": suffix.lstrip("."),
                "chunk_index": i,
                "total_chunks": total,
                "ingested_at": datetime.now().isoformat(),
            }
            for i in range(batch_start, batch_end)
        ]

        collection.upsert(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

        # Progress feedback for large files
        if total > batch_size:
            console.print(
                f"    [dim]batch {batch_start // batch_size + 1}/"
                f"{(total + batch_size - 1) // batch_size} "
                f"({batch_end}/{total} chunks)[/dim]"
            )

    return total


def ingest_directory(
    directory: Path = None,
    file_types: list[str] = None,
) -> dict:
    """
    Ingest all supported files from a directory into ChromaDB.

    This is the main entry point for ingestion. Call this to process
    your entire knowledge base, or pass a specific directory/file type.

    Args:
        directory: Path to scan (defaults to knowledge_base/)
        file_types: List of extensions to include (e.g., [".md", ".pdf"])
                    Defaults to all supported types.

    Returns:
        Summary dict with counts of files processed and chunks created.
    """
    directory = directory or KNOWLEDGE_BASE_DIR
    file_types = file_types or list(LOADERS.keys())

    files = []
    for ext in file_types:
        files.extend(directory.rglob(f"*{ext}"))
    files.sort()

    if not files:
        console.print(f"[yellow]No supported files found in {directory}[/yellow]")
        return {"files_processed": 0, "chunks_created": 0, "errors": []}

    console.print(f"\n[bold blue]Found {len(files)} file(s) to ingest[/bold blue]")

    # Initialize our tools
    embedder = OllamaEmbedder()
    collection = get_chroma_collection()

    total_chunks = 0
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=len(files))

        for file_path in files:
            try:
                progress.update(task, description=f"Processing {file_path.name}...")
                n_chunks = ingest_file(file_path, collection, embedder)
                total_chunks += n_chunks
                console.print(
                    f"  [green]✓[/green] {file_path.name} → {n_chunks} chunks"
                )
            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})
                console.print(f"  [red]✗[/red] {file_path.name}: {e}")
            finally:
                progress.advance(task)

    console.print(f"\n[bold green]Ingestion complete![/bold green]")
    console.print(f"  Files processed: {len(files)}")
    console.print(f"  Chunks created:  {total_chunks}")
    console.print(f"  Collection size: {collection.count()} total chunks in DB")
    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")

    return {
        "files_processed": len(files),
        "chunks_created": total_chunks,
        "collection_total": collection.count(),
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def get_collection_stats() -> dict:
    """Quick summary of what's in the database."""
    collection = get_chroma_collection()
    count = collection.count()

    if count == 0:
        return {"total_chunks": 0, "sources": []}

    # Peek at some metadata to get source file list
    results = collection.get(limit=count, include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])

    return {
        "total_chunks": count,
        "sources": sorted(sources),
    }


def clear_collection():
    """Wipe the entire collection. Use carefully!"""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    client.delete_collection(retrieval_config.collection_name)
    console.print("[yellow]Collection cleared.[/yellow]")


# ──────────────────────────────────────────────────────────────
# CLI entry point — run this file directly to ingest
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Allow passing a custom directory as argument
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = KNOWLEDGE_BASE_DIR

    console.print(f"[bold]Ingesting documents from: {target_dir}[/bold]")
    results = ingest_directory(target_dir)
