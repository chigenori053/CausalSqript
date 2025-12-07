import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.classifier import ExpressionClassifier

def test_calculus_rules():
    print("--- Testing Calculus Rules ---")
    
    # Initialize
    sym_engine = SymbolicEngine()
    knowledge_path = Path("core/knowledge")
    registry = KnowledgeRegistry(knowledge_path, sym_engine)
    classifier = ExpressionClassifier(sym_engine)
    
    test_cases = [
        {
            "name": "CALC-INT-LINEAR",
            "before": "integrate(3 * x, x)",
            "after": "3 * integrate(x, x)",
            "expected_id": "CALC-INT-LINEAR"
        },
        {
            "name": "CALC-INT-POWER",
            "before": "integrate(x**2, x)",
            "after": "x**3 / 3",
            "expected_id": "CALC-INT-POWER"
        },
        {
            "name": "CALC-DEF-INT",
            "before": "integrate(x, (x, 0, 1))",
            "after": "Subs(integrate(x, x), x, 1) - Subs(integrate(x, x), x, 0)",
            "expected_id": "CALC-DEF-INT"
        },
        {
            "name": "CALC-BOUNDS-EVAL",
            "before": "Subs(x**2/2, x, 1) - Subs(x**2/2, x, 0)",
            "after": "(1)**2/2 - (0)**2/2",
            "expected_id": "CALC-BOUNDS-EVAL"
        },
        {
            "name": "CALC-DIFF-POW",
            "before": "Derivative(x**3, x)",
            "after": "3*x**2",
            "expected_id": "CALC-DIFF-POW"
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['name']}...")
        before = case["before"]
        after = case["after"]
        
        # Classify to get context
        # domains = classifier.classify(before)
        domains = ["calculus"]
        print(f"Domains: {domains}")
        
        match = registry.match(before, after, context_domains=domains)
        
        if match:
            print(f"MATCH: {match.id}")
            if match.id == case["expected_id"]:
                print("PASS")
            else:
                print(f"FAIL: Expected {case['expected_id']}, got {match.id}")
        else:
            print("FAIL: No match found.")
            # Debug match_structure
            print("DEBUG: Checking match_structure...")
            # Find the rule
            rule = registry.rules_by_id.get(case["expected_id"])
            if rule:
                print(f"Rule Pattern: {rule.pattern_before}")
                structure_match = sym_engine.match_structure(before, rule.pattern_before)
                print(f"Structure Match: {structure_match}")
                
                # Check internal reps
                before_internal = sym_engine.to_internal(before)
                pattern_internal = sym_engine.to_internal(rule.pattern_before) # This might fail if pattern has wildcards not defined?
                # Actually to_internal doesn't handle Wilds nicely unless we use parse_expr with locals
                
                from sympy import Wild
                from sympy.parsing.sympy_parser import parse_expr
                local_dict = {"e": sym_engine._symbolic_engine.E if hasattr(sym_engine, '_symbolic_engine') else None} # Hacky access
                # Better to use the logic inside match_structure
            else:
                print(f"Rule {case['expected_id']} not found in registry.")

if __name__ == "__main__":
    test_calculus_rules()
