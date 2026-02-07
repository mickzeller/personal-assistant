"""
Configuration for the Personal AI Assistant.

All the knobs and dials in one place. Tweak these as you experiment —
the defaults are solid starting points for your M4 Max setup.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"


class OllamaConfig(BaseSettings):
    """
    Ollama connection settings.

    Ollama runs a local HTTP API on port 11434 by default.
    If you ever change the port or run it on another machine,
    update the base_url here.
    """
    base_url: str = "http://localhost:11434"

    # The model used for chat / generation
    # Options on your machine: "gemma3:27b", "llama3.1:8b-instruct-q8_0"
    # For max quality: use gemma3:27b (your daily driver)
    # For testing/speed: use llama3.1:8b-instruct-q8_0
    chat_model: str = "llama3.1:8b-instruct-q8_0"

    # The model used to generate embeddings for RAG
    # nomic-embed-text is a great balance of quality and speed
    embedding_model: str = "nomic-embed-text"


class ChunkingConfig(BaseSettings):
    """
    Controls how documents get split into chunks for the vector database.

    WHY THIS MATTERS:
    ─────────────────
    Too big → chunks contain mixed topics, retrieval gets noisy
    Too small → chunks lose context, model can't understand them
    Sweet spot → each chunk captures one coherent idea with enough
                 context to be useful on its own

    512 tokens with 50 overlap is a good default. Experiment from here.
    """
    chunk_size: int = 512          # Target size in characters (~128 tokens)
    chunk_overlap: int = 50        # Overlap between chunks (prevents cutting sentences)

    # Separators to split on, tried in order (most → least desirable)
    # The splitter tries paragraph breaks first, then sentences, etc.
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]


class RetrievalConfig(BaseSettings):
    """
    Controls the RAG retrieval pipeline.

    The pipeline works in two stages:
    1. BROAD RETRIEVAL: Grab top_k candidates from ChromaDB (fast, approximate)
    2. RERANKING: Score those candidates with a cross-encoder (slow, precise)
       Then return the top rerank_top_k results

    WHY TWO STAGES:
    ────────────────
    Embedding similarity (stage 1) is fast but sometimes misses relevant
    chunks that use different wording. The cross-encoder (stage 2) reads
    the query AND each chunk together, catching relevance that embeddings
    miss. It's ~100ms slower but dramatically more accurate.
    """
    # Stage 1: How many chunks to pull from ChromaDB
    top_k: int = 10

    # Stage 2: How many chunks to keep after reranking
    rerank_top_k: int = 5

    # Cross-encoder model for reranking
    # ms-marco-MiniLM-L-6-v2 is small, fast, and excellent at relevance scoring
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Minimum similarity score to include a chunk (0.0 to 1.0)
    # Set to 0.0 to disable filtering — useful while you're learning
    min_relevance_score: float = 0.0

    # ChromaDB collection name
    collection_name: str = "personal_knowledge"


class PromptConfig(BaseSettings):
    """
    System prompt and context injection templates.

    This is where you define your assistant's "personality."
    Feel free to customize the system_prompt — it's your assistant!
    """
    system_prompt: str = (
        "You are a helpful, honest, and knowledgeable personal AI assistant. "
        "You answer questions based on the provided context and your general knowledge.\n\n"
        "RULES:\n"
        "- If the context contains relevant information, use it to inform your answer\n"
        "- If the context doesn't contain enough information, say so honestly — don't make things up\n"
        "- When citing specific information from the context, mention the source\n"
        "- Be direct and concise, but thorough when the question warrants it\n"
        "- If you're unsure about something, say so"
    )

    # Template for injecting retrieved context into the prompt
    # {context} and {query} are replaced at runtime
    context_template: str = (
        "Here is relevant context from your knowledge base:\n\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Using the context above and your general knowledge, "
        "please answer the following question:\n\n"
        "{query}"
    )

    # Template when NO context is retrieved (general conversation)
    no_context_template: str = "{query}"


# ──────────────────────────────────────────────────────────────
# Single config instances (import these in other modules)
# ──────────────────────────────────────────────────────────────

ollama_config = OllamaConfig()
chunking_config = ChunkingConfig()
retrieval_config = RetrievalConfig()
prompt_config = PromptConfig()
