"""
Verification Script for Coherent Memory Module.
Compares Global Search vs Scoped Search.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coherent.memory.management import ingest_all_rules
from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.memory import get_vector_store

def setup():
    print(">>> Setting up Knowledge Registry...")
    engine = SymbolicEngine()
    # Assuming knowledge files are in coherent/engine/knowledge
    knowledge_path = Path("coherent/engine/knowledge")
    registry = KnowledgeRegistry(base_path=knowledge_path, engine=engine)
    
    print(f"Loaded {len(registry.nodes)} rules from {knowledge_path}")
    
    print(">>> Ingesting Rules into Vector Store...")
    # This might take a few seconds
    ingest_all_rules(registry)
    return registry

def run_experiment(registry):
    print("\n>>> Running Search Experiment: Global vs Scoped")
    
    test_cases = [
        {
            "query": "derivative of sine function",
            "category": "calculus",
            "description": "Calculus query about trigonometry"
        },
        {
            "query": "solve for x in linear equation",
            "category": "algebra",
            "description": "Algebraic query"
        },
        {
            "query": "matrix multiplication",
            "category": "linear_algebra",
            "description": "Linear Algebra query"
        },
        {
            "query": "area of a circle",
            "category": "geometry",
            "description": "Geometry query"
        }
    ]
    
    for case in test_cases:
        query = case["query"]
        category = case["category"]
        desc = case["description"]
        
        print(f"\n--- Case: {desc} ---")
        print(f"Query: '{query}'")
        print(f"Target Category: {category}")
        
        # Global Search
        print(f"[Global Search] (No filter)")
        results_global = registry.semantic_search(query, category=None, top_k=3)
        for r in results_global:
            meta = r['metadata']
            print(f"  - [{meta.get('domain')}] {meta.get('pattern_before')} -> {meta.get('pattern_after')} (Dist: {r.get('distance', 0):.4f})")
            
        # Scoped Search
        print(f"[Scoped Search] (Filter: {category})")
        results_scoped = registry.semantic_search(query, category=category, top_k=3)
        for r in results_scoped:
            meta = r['metadata']
            print(f"  - [{meta.get('domain')}] {meta.get('pattern_before')} -> {meta.get('pattern_after')} (Dist: {r.get('distance', 0):.4f})")

        # Basic validation
        # Check if scoped results obey the domain filter logic (if implemented in retriever)
        # Note: retriever implementation maps category to domain.
        
if __name__ == "__main__":
    try:
        registry = setup()
        run_experiment(registry)
    except ImportError as e:
        print(f"Dependency missing: {e}")
        print("Please ensure sentence-transformers and chromadb are installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
