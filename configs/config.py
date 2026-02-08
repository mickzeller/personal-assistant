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


class FineTuneConfig(BaseSettings):
    """
    LoRA fine-tuning configuration for MLX on Apple Silicon.

    WHY THESE VALUES:
    ─────────────────
    With only 83 training examples, we're in "tiny dataset" territory.
    Every knob here is tuned to PREVENT OVERFITTING while still learning
    Mick's writing style. Think of it as teaching with a light touch —
    we want the model to pick up tone and phrasing patterns without
    memorizing specific sentences.

    The base model is a pre-quantized 4-bit Llama 3.1 8B from the
    mlx-community — no conversion needed, it's ready to train.
    """

    # ── Model ──────────────────────────────────────────────────
    # Pre-quantized MLX model from HuggingFace (downloads ~4.5GB on first run)
    # Using 4-bit keeps memory usage low and training fast on M4 Max
    base_model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

    # The Ollama model name we'll create after fine-tuning
    ollama_model_name: str = "llama3.1-mick:latest"

    # Ollama base model for Modelfile (used for direct adapter import)
    ollama_base_model: str = "llama3.1:8b-instruct-q8_0"

    # ── LoRA Hyperparameters ───────────────────────────────────
    # Rank 8 = fewer trainable parameters = less overfitting risk
    # For 142 examples, rank 8 is solid — could go to 16 with 300+ examples
    lora_rank: int = 8

    # Only adapt 8 of 32 layers — "light touch" fine-tuning
    # Full 32 layers would memorize; 8 layers learns style
    num_layers: int = 8

    # Aggressive dropout — randomly zeros 10% of LoRA weights each step
    # Acts like a regularizer, forces the model to learn general patterns
    lora_dropout: float = 0.1

    # ── Training ───────────────────────────────────────────────
    # Small batch = implicit regularization (noisier gradients = less memorization)
    batch_size: int = 2

    # Conservative learning rate — we want gentle style transfer, not overwriting
    # 1e-5 is ~10x lower than typical LoRA defaults (1e-4)
    learning_rate: float = 1e-5

    # 400 iterations ≈ 7 epochs over 113 training examples (80% of 142)
    # More data = fewer epochs needed. Watch val loss — if it climbs, we're overfitting
    iters: int = 400

    # Check validation loss every 50 steps (~1.5 epochs)
    # Frequent enough to catch overfitting early
    steps_per_eval: int = 50

    # Save adapter checkpoint every 100 steps (safety net)
    steps_per_save: int = 100

    # CRITICAL: Only compute loss on ASSISTANT responses, not prompts
    # Without this, the model wastes capacity "learning" to generate prompts
    mask_prompt: bool = True

    # ── Paths ──────────────────────────────────────────────────
    # Where the training JSONL lives
    data_dir: str = "data/training"

    # Where prepared train/valid/test splits go
    prepared_dir: str = "data/training/prepared"

    # Where the LoRA adapter weights get saved
    adapter_path: str = "adapters/writing-style"

    # Where fused models go (only used for GGUF export path)
    fused_model_path: str = "models/fused/writing-style"

    # Where GGUF files go (only used for GGUF export path)
    gguf_path: str = "models/gguf"

    # ── Test Prompts ───────────────────────────────────────────
    # Used by model_tester.py for A/B comparison
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
            "context": "church or ministry context",
            "prompt": "Write a warm thank-you message to someone who helped with a church event",
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
finetune_config = FineTuneConfig()
