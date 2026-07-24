"""
Embeddings via Ollama (nomic-embed-text) - all local, no API costs.

For example:
    "The cat sat on the mat"  →  [0.12, -0.45, 0.78, ...]
    "A kitten rested on a rug" → [0.11, -0.43, 0.76, ...]   (similar!)
    "Stock markets crashed"    → [-0.89, 0.22, -0.11, ...]   (very different)

We use Ollama to run nomic-embed-text locally - no API calls, no cost,
all of your data stays on your machine 😁

WHY OLLAMA FOR EMBEDDINGS:
──────────────────────────
We could use sentence-transformers directly, but
since Ollama is already running, we get:
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
    batch support. Each call to Ollama is local. No network latency,
    no API key, no cost. We got this!
    """

    def __init__(self, model: str = None):
        self.model = model or ollama_config.embedding_model
        self._client = ollama.Client(host=ollama_config.base_url)

        self._verify_model()

    def _verify_model(self):
        """Quick test embed to make sure the model is pulled and ready."""
        try:
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
        """Embed a list of document chunks."""
        if not texts:
            return []

        # nomic-embed-text was trained with task prefixes, "search_document:"
        # for stored docs, "search_query:" for questions. Using them matters.
        prefixed = [f"search_document: {t}" for t in texts]

        response = self._client.embed(model=self.model, input=prefixed)
        return response["embeddings"]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        prefixed = f"search_query: {query}"
        response = self._client.embed(model=self.model, input=prefixed)
        return response["embeddings"][0]
