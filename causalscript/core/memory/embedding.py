"""
Embedding Engine for CausalScript Memory.
"""

from abc import ABC, abstractmethod
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class Embedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings."""
        pass

class SemanticEmbedder(Embedder):
    """
    Uses sentence-transformers to generate dense vector embeddings.
    Default model: all-MiniLM-L6-v2
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not installed.")
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        embeddings = self.model.encode([text])
        return embeddings[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
