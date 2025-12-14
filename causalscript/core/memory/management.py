"""
Management utilities for Memory Module (Ingestion, Maintenance).
"""

from typing import List
from .factory import get_vector_store, get_embedder
from .schema import KnowledgeEntry
from ..knowledge_registry import KnowledgeRegistry

def ingest_all_rules(registry: KnowledgeRegistry, collection_name: str = "knowledge"):
    """
    Ingests all rules from the KnowledgeRegistry into the Vector Store.
    """
    store = get_vector_store()
    embedder = get_embedder()

    nodes = registry.nodes
    if not nodes:
        print("No rules found in registry.")
        return

    vectors = []
    ids = []
    metadatas = []

    print(f"Ingesting {len(nodes)} rules...")

    # Batch embedding could be optimized, but loop is fine for <1000 rules
    # Text representation for embedding:
    # "Expression: {pattern_before} -> {pattern_after}. Description: {description}"
    texts = []
    
    for node in nodes:
        # Construct a meaningful text representation for semantic search
        text = f"Transform {node.pattern_before} to {node.pattern_after}. {node.description}"
        texts.append(text)
        
        entry = KnowledgeEntry(
            id=node.id,
            description=node.description,
            pattern_before=node.pattern_before,
            pattern_after=node.pattern_after,
            domain=node.domain,
            priority=node.priority,
            metadata={"category": node.category}
        )
        
        ids.append(node.id)
        metadatas.append(entry.to_metadata())

    # Generate embeddings in batch
    vectors = embedder.embed_batch(texts)

    # Add to store
    store.add(
        collection_name=collection_name,
        vectors=vectors,
        metadatas=metadatas,
        ids=ids
    )
    print("Ingestion complete.")
