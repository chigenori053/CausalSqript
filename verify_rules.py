from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from pathlib import Path
import os

# Setup
sym_engine = SymbolicEngine()
knowledge_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "core", "knowledge")))
registry = KnowledgeRegistry(knowledge_path, sym_engine)

def test_match(before, after, expected_id):
    print(f"Testing: {before} -> {after}")
    op = sym_engine.get_top_operator(before)
    print(f"  Top Op: {op}")
    
    # Debug specific rule
    if expected_id == "ALG-POW-FRAC-001":
        pattern = "(a / b)**2"
        print(f"  Debugging match against pattern: {pattern}")
        bindings = sym_engine.match_structure(before, pattern)
        print(f"  Bindings: {bindings}")

    match = registry.match(before, after)
    if match:
        print(f"  Matched: {match.id}")
        if match.id == expected_id:
            print("  PASS")
        else:
            print(f"  FAIL: Expected {expected_id}, got {match.id}")
    else:
        print("  FAIL: No match found")

# Test cases
test_match("(sqrt(3)/2)**2", "(sqrt(3)*sqrt(3))/(2*2)", "ALG-POW-FRAC-001")
test_match("(x/y)**3", "x**3 / y**3", "ALG-POW-FRAC-002")
test_match("(a/b)**2", "(a/b) * (a/b)", "ALG-POW-DEF-FRAC")
