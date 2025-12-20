import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.knowledge_registry import KnowledgeRegistry

def debug_algebra_2():
    print("--- Debugging 5a - 3b + 2a + b ---")
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(root_path=Path("coherent/engine/knowledge"), engine=engine)
    
    expr = "5*a - 3*b + 2*a + b"
    # Note: Human input "5a" becomes "5*a" via InputParser.
    
    print(f"Expr: {expr}")
    
    # Try ARITH-SUB-001 matching
    sub_rule = registry.rules_by_id.get("ARITH-SUB-001")
    if sub_rule:
        print(f"\nChecking ARITH-SUB-001: {sub_rule.pattern_before} -> {sub_rule.pattern_after}")
        bindings = engine.match_structure(expr, sub_rule.pattern_before)
        if bindings:
            print(f"  Bindings: {bindings}")
            result = engine.substitute(sub_rule.pattern_after, {k: str(v) for k, v in bindings.items()})
            print(f"  Result: {result}")
        else:
            print("  No match.")
            
    # Try ARITH-ADD-002 matching
    add_rule = registry.rules_by_id.get("ARITH-ADD-002")
    if add_rule:
        print(f"\nChecking ARITH-ADD-002: {add_rule.pattern_before} -> {add_rule.pattern_after}")
        bindings = engine.match_structure(expr, add_rule.pattern_before)
        if bindings:
             print(f"  Bindings: {bindings}")
             result = engine.substitute(add_rule.pattern_after, {k: str(v) for k, v in bindings.items()})
             print(f"  Result: {result}")
        else:
             print("  No match.")

if __name__ == "__main__":
    debug_algebra_2()
