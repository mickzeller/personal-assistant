"""All the knobs and dials in one place. Tweak away."""

from pathlib import Path
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"


class OllamaConfig(BaseSettings):
    """Ollama connection settings (local API on 11434 by default)."""
    base_url: str = "http://localhost:11434"

    # gemma3:27b for quality, llama3.1:8b-instruct-q8_0 for speed
    chat_model: str = "llama3.1:8b-instruct-q8_0"

    # RAG embeddings — nomic is a solid balance of quality and speed
    embedding_model: str = "nomic-embed-text"


class ChunkingConfig(BaseSettings):
    """
    How docs get split for the vector DB. Too big = noisy retrieval,
    too small = no context. 512/50 is a solid default.
    """
    chunk_size: int = 512          # characters (~128 tokens)
    chunk_overlap: int = 50        # keeps sentences from getting cut in half

    # Tried in order: paragraphs first, then lines, sentences, words
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]


class RetrievalConfig(BaseSettings):
    """
    RAG retrieval settings. Two stages: grab top_k candidates from
    ChromaDB (fast), then rerank with a cross-encoder that actually
    reads query + chunk together (~100ms slower, way more accurate).
    """
    top_k: int = 10          # stage 1: candidates from ChromaDB
    rerank_top_k: int = 5    # stage 2: keepers after reranking

    # Small, fast, and great at relevance scoring
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Minimum rerank score to keep a chunk (0.0 = no filtering)
    min_relevance_score: float = 0.0

    collection_name: str = "personal_knowledge"


class FineTuneConfig(BaseSettings):
    """
    LoRA fine-tuning config for MLX on Apple Silicon.

    Tiny dataset (~240 examples as of v3), so everything here is tuned to
    prevent overfitting — we want tone and phrasing, not memorized sentences.
    """

    # ── Model ──────────────────────────────────────────────────
    # Pre-quantized 4-bit MLX model (~4.5GB download on first run)
    base_model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

    # What we name the Ollama model after fine-tuning
    ollama_model_name: str = "llama3.1-assistant:latest"

    # Modelfile base for direct adapter import
    ollama_base_model: str = "llama3.1:8b-instruct-q8_0"

    # ── LoRA Hyperparameters ───────────────────────────────────
    # Rank 8 on 8 of 32 layers = light touch. Learns style without
    # memorizing. Could go rank 16 with 300+ examples.
    lora_rank: int = 8
    num_layers: int = 8

    # 10% dropout forces general patterns over specific sentences
    lora_dropout: float = 0.1

    # ── Inference ──────────────────────────────────────────────
    # 0.7 produced overly hyped output — dialed back for v3
    temperature: float = 0.5

    # ── Training ───────────────────────────────────────────────
    # Small batch = noisier gradients = less memorization
    batch_size: int = 2

    # ~10x lower than typical LoRA defaults — gentle style transfer
    learning_rate: float = 1e-5

    # ~5 epochs. Reduced from 400 — v2 was still too eager.
    # If val loss climbs while train loss drops, we've gone too far.
    iters: int = 300

    steps_per_eval: int = 50    # check val loss often enough to catch overfitting
    steps_per_save: int = 100   # checkpoint safety net

    # Only compute loss on assistant responses — without this the model
    # wastes capacity learning to generate the prompts themselves
    mask_prompt: bool = True

    # ── Paths ──────────────────────────────────────────────────
    data_dir: str = "data/training"
    prepared_dir: str = "data/training/prepared"
    adapter_path: str = "adapters/writing-style"
    fused_model_path: str = "models/fused/writing-style"   # GGUF export path only
    gguf_path: str = "models/gguf"                         # GGUF export path only

    # ── Test Prompts (used by model_tester.py) ─────────────────
    test_prompts: list[dict] = [
        {
            "context": "casual text to a close friend",
            "prompt": "Write a casual text to a friend about getting together this weekend",
        },
        {
            "context": "workplace communication",
            "prompt": "Write a Slack message to a coworker about a bug you found in their PR",
        },
        {
            "context": "community or volunteer context",
            "prompt": "Write a warm thank-you message to someone who helped with a community event",
        },
        {
            "context": "online review",
            "prompt": "Write a positive review for a local coffee shop you really like",
        },
        {
            "context": "technical writing or explanation",
            "prompt": "Explain what a REST API is to someone who's never heard of it",
        },
        {
            "context": "personal text message",
            "prompt": "Write a birthday message to your brother",
        },
        {
            "context": "workplace communication",
            "prompt": "Write a message to your team announcing you'll be out of office next week",
        },
    ]


class PromptConfig(BaseSettings):
    """System prompt and context templates — the assistant's personality."""
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

    # {context} and {query} get filled at runtime
    context_template: str = (
        "Here is relevant context from your knowledge base:\n\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Using the context above and your general knowledge, "
        "please answer the following question:\n\n"
        "{query}"
    )

    # Used when nothing relevant was retrieved
    no_context_template: str = "{query}"


# Import these singletons in other modules

ollama_config = OllamaConfig()
chunking_config = ChunkingConfig()
retrieval_config = RetrievalConfig()
prompt_config = PromptConfig()
finetune_config = FineTuneConfig()
