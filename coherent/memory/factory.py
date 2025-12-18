"""
Factory methods for Memory Module components.
"""
from .vector_store import VectorStoreBase, ChromaVectorStore
from .embedding import SemanticEmbedder

_STORE_INSTANCE = None
_EMBEDDER_INSTANCE = None

def get_vector_store(persist_path: str = "./vector_db") -> VectorStoreBase:
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = ChromaVectorStore(persist_path=persist_path)
    return _STORE_INSTANCE

def get_embedder() -> SemanticEmbedder:
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = SemanticEmbedder()
    return _EMBEDDER_INSTANCE
