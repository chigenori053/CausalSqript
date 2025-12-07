from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from pathlib import Path
import pytest

def debug_matching():
    engine = SymbolicEngine()
    knowledge_path = Path("core/knowledge")
    registry = KnowledgeRegistry(knowledge_path, engine)
    
    print("\n--- Debugging Constant Multiple Rule ---")
    expr = "Integral(3*x**2, x)"
    rule_id = "CALC-INT-CONST"
    rule = registry.rules_by_id.get(rule_id)
    
    if rule:
        print(f"Rule found: {rule.pattern_before}")
        
        # Inspect internals
        from sympy.parsing.sympy_parser import parse_expr
        from sympy import Wild
        local_dict = {"e": SymbolicEngine().to_internal("e"), "pi": SymbolicEngine().to_internal("pi")}
        
        expr_internal = parse_expr(expr, evaluate=False, local_dict=local_dict)
        print(f"Expr Internal: {expr_internal} (Type: {type(expr_internal)})")
        print(f"Expr Args: {expr_internal.args}")
        
        # Prepare pattern
        wild_names = ['c', 'f', 'x']
        pattern_locals = local_dict.copy()
        for name in wild_names:
            pattern_locals[name] = Wild(name, exclude=[])
            
        pattern_internal = parse_expr(rule.pattern_before, evaluate=False, local_dict=pattern_locals)
        print(f"Pattern Internal: {pattern_internal} (Type: {type(pattern_internal)})")
        print(f"Pattern Args: {pattern_internal.args}")
        
        matches = expr_internal.match(pattern_internal)
        print(f"Direct Match Result: {matches}")

    print("\n--- Debugging Definite Integral Rule ---")
    expr_def = "Integral(x, (x, 0, 2))"
    rule_def_id = "CALC-INT-DEF"
    rule_def = registry.rules_by_id.get(rule_def_id)
    
    if rule_def:
        print(f"Rule found: {rule_def.pattern_before}")
        
        expr_def_internal = parse_expr(expr_def, evaluate=False, local_dict=local_dict)
        print(f"Expr Def Internal: {expr_def_internal}")
        print(f"Expr Def Args: {expr_def_internal.args}")
        
        pattern_def_internal = parse_expr(rule_def.pattern_before, evaluate=False, local_dict=pattern_locals)
        print(f"Pattern Def Internal: {pattern_def_internal}")
        print(f"Pattern Def Args: {pattern_def_internal.args}")
        
        matches_def = expr_def_internal.match(pattern_def_internal)
        print(f"Direct Match Def Result: {matches_def}")

if __name__ == "__main__":
    debug_matching()
