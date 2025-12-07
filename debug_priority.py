import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry

def debug_priority():
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), engine)
    
    # Check Algebra Basic Map
    alg_map = next((m for m in registry.maps if m.id == "algebra_basic"), None)
    if not alg_map:
        print("Algebra Basic map not found!")
        return

    print(f"Map: {alg_map.id}")
    print("Rules in order:")
    for rid in alg_map.rules:
        node = registry.rules_by_id.get(rid)
        p = node.priority if node else "N/A"
        print(f"  {rid}: {p}")
        
    # Check specific rules
    op002 = registry.rules_by_id.get("ALG-OP-002")
    fac001 = registry.rules_by_id.get("ALG-FAC-001")
    
    print(f"\nALG-OP-002 Priority: {op002.priority}")
    print(f"ALG-FAC-001 Priority: {fac001.priority}")
    
    # Test Match
    expr = "x + x"
    target = "2*x"
    print(f"\nMatching '{expr}' -> '{target}'")
    match = registry.match(expr, target, context_domains=["algebra"])
    if match:
        print(f"Matched: {match.id} (Priority: {match.priority})")
    else:
        print("No match")

if __name__ == "__main__":
    debug_priority()
