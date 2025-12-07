import sys
import os
from pathlib import Path
import sympy

# Add project root to path
sys.path.append(os.getcwd())

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.input_parser import CausalScriptInputParser
from causalscript.core.latex_formatter import LaTeXFormatter

def test_rendering():
    print("--- Test Rendering ---")
    engine = SymbolicEngine()
    formatter = LaTeXFormatter(engine)
    
    expr = "2^3 - 0"
    normalized = CausalScriptInputParser.normalize(expr)
    print(f"Normalized: {normalized}")
    
    latex = formatter.format_expression(normalized)
    print(f"LaTeX: {latex}")
    
    # Check for the specific bad output mentioned by user
    if "(-1)" in latex and "0" in latex and "+" in latex:
        print("FAIL: Bad rendering detected.")
    else:
        print("PASS: Rendering looks okay (check visually).")

    # Inspect internal structure
    from sympy.parsing.sympy_parser import parse_expr
    from sympy import srepr
    local_dict = {"e": sympy.E, "pi": sympy.pi}
    internal = parse_expr(normalized, evaluate=False, local_dict=local_dict)
    print(f"Internal srepr: {srepr(internal)}")
    print(f"Internal str: {str(internal)}")
    
    # Try mul_symbol
    print(f"LaTeX with mul_symbol='dot': {sympy.latex(internal, mul_symbol='dot')}")
    print(f"LaTeX with mul_symbol='times': {sympy.latex(internal, mul_symbol='times')}")

def test_rule_matching():
    print("\n--- Test Rule Matching ---")
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), engine)
    
    expr = "2^3 - 0"
    normalized = CausalScriptInputParser.normalize(expr)
    
    # Check top operator
    op = engine.get_top_operator(normalized)
    print(f"Top Operator of '{normalized}': {op}")
    
    # Check ARITH-CALC-ADD pattern operator
    add_rule = next((r for r in registry.nodes if r.id == "ARITH-CALC-ADD"), None)
    if add_rule:
        add_op = engine.get_top_operator(add_rule.pattern_before)
        print(f"ARITH-CALC-ADD Pattern '{add_rule.pattern_before}' Operator: {add_op}")
        
        # Check if they match
        arithmetic_ops = {"Add", "Sub", "Mul", "Mult", "Div", "Pow"}
        if op in arithmetic_ops and add_op in arithmetic_ops:
            if op != add_op:
                print(f"Strict Check: {op} != {add_op} -> Should SKIP ARITH-CALC-ADD")
            else:
                print(f"Strict Check: {op} == {add_op} -> Can MATCH ARITH-CALC-ADD")
    
    # Perform match
    rule = registry.match(normalized, "8")
    if rule:
        print(f"Matched Rule: {rule.id}")
    else:
        print("No Rule Matched")

if __name__ == "__main__":
    test_rendering()
    test_rule_matching()
