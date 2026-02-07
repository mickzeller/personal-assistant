"""
Embedding module — converts text into numerical vectors using Ollama.

HOW EMBEDDINGS WORK:
────────────────────
An embedding model reads text and produces a fixed-size list of numbers
(a "vector") that captures the *meaning* of that text. Texts with similar
meaning end up with similar vectors.

For example:
    "The cat sat on the mat"  →  [0.12, -0.45, 0.78, ...]
    "A kitten rested on a rug" → [0.11, -0.43, 0.76, ...]   (similar!)
    "Stock markets crashed"    → [-0.89, 0.22, -0.11, ...]   (very different)

We use Ollama to run nomic-embed-text locally — no API calls, no cost,
all your data stays on your machine.

WHY OLLAMA FOR EMBEDDINGS:
──────────────────────────
You could use sentence-transformers directly (and many tutorials do), but
since you already have Ollama running, we get:
- Consistent interface (same tool for chat + embeddings)
- Model management handled by Ollama (pulls, updates, quantization)
- One less thing to configure
"""

import ollama

from configs.config import ollama_config


class OllamaEmbedder:
    """
    Generates embeddings using Ollama's local embedding API.

    This wraps the `ollama.embed()` call with our config and adds
    batch support. Each call to Ollama is local — no network latency,
    no API key, no cost.
    """

    def __init__(self, model: str = None):
        self.model = model or ollama_config.embedding_model
        self._client = ollama.Client(host=ollama_config.base_url)

        # Verify the model is available
        self._verify_model()

    def _verify_model(self):
        """Check that the embedding model is pulled and ready."""
        try:
            # Quick test embed to make sure the model works
            self._client.embed(model=self.model, input="test")
        except ConnectionError:
            raise RuntimeError(
                "Can't connect to Ollama. Is it running?\n"
                "Start it with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(
                f"Embedding model '{self.model}' is not available in Ollama.\n"
                f"Pull it with: ollama pull {self.model}\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of document chunks.

        Uses the "search_document" prefix for nomic-embed-text, which
        tells the model these are documents being indexed (not queries).

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        # nomic-embed-text uses task prefixes for better quality
        # "search_document:" for docs being stored
        # "search_query:" for user questions
        prefixed = [f"search_document: {t}" for t in texts]

        response = self._client.embed(model=self.model, input=prefixed)
        return response["embeddings"]

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single search query.

        Uses the "search_query" prefix — this matters for nomic-embed-text!
        The model was trained to match queries to documents using these
        prefixes, so including them improves retrieval quality.

        Args:
            query: The user's question or search text

        Returns:
            A single embedding vector
        """
        prefixed = f"search_query: {query}"
        response = self._client.embed(model=self.model, input=prefixed)
        return response["embeddings"][0]
