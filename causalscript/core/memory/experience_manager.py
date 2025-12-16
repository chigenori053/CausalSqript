
import logging
from typing import List, Optional, Dict, Any
import uuid
from .vector_store import VectorStoreBase
from .schema import ExperienceEntry

class ExperienceManager:
    """
    Manages the "Experience Network" (Edges between AST States).
    Handles saving and retrieving ExperienceEntry items from the VectorStore.
    """
    def __init__(self, vector_store: VectorStoreBase):
        self.vector_store = vector_store
        self.collection_name = "experience_network"
        self.logger = logging.getLogger(__name__)

    def save_edge(self, source_state_gen: str, target_state_gen: str, rule_id: str, source_vector: List[float]):
        """
        Saves a transition edge (Action) from Source to Target.
        
        Args:
            source_state_gen: Generalized string of source state.
            target_state_gen: Generalized string of target state.
            rule_id: The action taken.
            source_vector: Embedding of the source state.
        """
        edge_id = str(uuid.uuid4())
        
        entry = ExperienceEntry(
            id=edge_id,
            original_expr=source_state_gen, # Storing generalized form as the 'original' for lookup
            next_expr=target_state_gen,
            rule_id=rule_id,
            result_label="EXACT", # Default for now
            category="math",      # Default
            score=1.0,
            vector=source_vector
        )
        
        self.vector_store.add(
            collection_name=self.collection_name,
            vectors=[source_vector],
            metadatas=[entry.to_metadata()],
            ids=[edge_id]
        )
        self.logger.info(f"Saved Edge: {source_state_gen} -> [{rule_id}] -> {target_state_gen}")

    def find_similar_edges(self, query_vector: List[float], top_k: int = 5) -> List[ExperienceEntry]:
        """
        Recall: Finds edges starting from states similar to the query.
        """
        results = self.vector_store.query(
            collection_name=self.collection_name,
            query_vec=query_vector,
            top_k=top_k
        )
        
        entries = []
        for res in results:
            meta = res.get("metadata", {})
            try:
                entry = ExperienceEntry(
                    id=res["id"],
                    original_expr=meta.get("original_expr", ""),
                    next_expr=meta.get("next_expr", ""),
                    rule_id=meta.get("rule_id", ""),
                    result_label=meta.get("result_label", "EXACT"),
                    category=meta.get("category", "math"),
                    score=meta.get("score", 0.0),
                    vector=None, # Not returned by query usuall
                    metadata=meta
                )
                entries.append(entry)
            except Exception as e:
                self.logger.error(f"Failed to parse experience entry: {e}")
                
        return entries
