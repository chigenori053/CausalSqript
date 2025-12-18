"""
retriever.py: Semantic Search Logic
"""

from typing import List, Dict, Any, Optional
from .factory import get_vector_store, get_embedder

class MemoryRetriever:
    """
    Facade for searching Knowledge and Experience collections.
    """
    def __init__(self):
        self.store = get_vector_store()
        self.embedder = get_embedder()

    def search_rules(
        self,
        query_text: str,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant rules (Knowledge).
        If category is provided, filters by domain/category.
        """
        query_vec = self.embedder.embed_text(query_text)
        
        filter_dict = None
        if category:
            # Simple mapping from category to domain/category in metadata
            # For now, let's assume 'category' field in metadata matches
            # or we filter by domain if mapped.
            # Design doc mentioned: filter={"domain": current_context_domain}
            # But the metadata we stored has 'domain' and 'category'.
            # Let's support an exact match on domain or category if possible.
            # ChromaDB simple filtering: {"metadata_field": "value"}
            # Complex filtering requires specific syntax ({"$or": ...})
            # Let's stick to simple domain filtering for MVP if category is a domain name.
            # Or pass the filter dict directly?
            # Let's map high level category to domain filter.
            
            # Map for common terms
            domain_map = {
                "calculus": "calculus",
                "algebra": "algebra",
                "geometry": "geometry"
            }
            target_domain = domain_map.get(category, category)
            filter_dict = {"domain": target_domain}

        results = self.store.query(
            collection_name="knowledge",
            query_vec=query_vec,
            filter=filter_dict,
            top_k=top_k
        )
        return results

    def search_experiences(
        self,
        query_text: str,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for past experiences.
        """
        query_vec = self.embedder.embed_text(query_text)
        
        filter_dict = None
        if category:
             filter_dict = {"category": category}

        results = self.store.query(
            collection_name="experience",
            query_vec=query_vec,
            filter=filter_dict,
            top_k=top_k
        )
        return results
