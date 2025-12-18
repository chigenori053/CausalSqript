"""
Vector Store Adapter.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

class VectorStoreBase(ABC):
    @abstractmethod
    def add(self, collection_name: str, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Add vectors to the store."""
        pass

    @abstractmethod
    def query(self, collection_name: str, query_vec: List[float], filter: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the store.
        Returns a list of results (metadata + distance/score if available).
        """
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> None:
        pass

class ChromaVectorStore(VectorStoreBase):
    def __init__(self, persist_path: str = "./vector_db"):
        if chromadb is None:
            raise ImportError("chromadb is not installed.")
        
        # Ensure directory exists
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_path)

    def _get_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def add(self, collection_name: str, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        if not ids:
            return
            
        coll = self._get_collection(collection_name)
        coll.upsert(
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, collection_name: str, query_vec: List[float], filter: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        coll = self._get_collection(collection_name)
        
        # Chroma expects filter in a specific format if provided
        # If filter is None, pass None
        results = coll.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=filter if filter else None
        )
        
        # Unpack results
        # results['ids'], results['metadatas'], results['distances'] are lists of lists
        output = []
        if results['ids'] and len(results['ids']) > 0:
            count = len(results['ids'][0])
            for i in range(count):
                item = {
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0
                }
                output.append(item)
        
        return output

    def delete(self, collection_name: str, ids: List[str]) -> None:
        coll = self._get_collection(collection_name)
        coll.delete(ids=ids)
