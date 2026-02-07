"""Configuration module."""

from .config import (
    PROJECT_ROOT,
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_DIR,
    ollama_config,
    chunking_config,
    retrieval_config,
    prompt_config,
)

__all__ = [
    "PROJECT_ROOT",
    "KNOWLEDGE_BASE_DIR",
    "CHROMA_DB_DIR",
    "ollama_config",
    "chunking_config",
    "retrieval_config",
    "prompt_config",
]
