import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.knowledge_registry import KnowledgeRegistry
from causalscript.core.input_parser import CausalScriptInputParser
from causalscript.core.evaluator import Evaluator, SymbolicEvaluationEngine
from causalscript.core.parser import Parser
from causalscript.core.learning_logger import LearningLogger
from causalscript.core.ast_nodes import ProgramNode

def test_strict_matching():
    print("--- Test Strict Matching ---")
    engine = SymbolicEngine()
    registry = KnowledgeRegistry(Path("core/knowledge"), engine)
    
    expr = "2^3 - 0"
    normalized = CausalScriptInputParser.normalize(expr)
    print(f"Normalized: {normalized}")
    
    op_expr = engine.get_top_operator(normalized)
    print(f"Top Operator (Expr): {op_expr}")
    
    rule = registry.match(normalized, "8")
    if rule:
        print(f"Matched Rule: {rule.id}")
    else:
        print("No Rule Matched")

    # Check ARITH-CALC-ADD specifically
    add_rule = next((r for r in registry.nodes if r.id == "ARITH-CALC-ADD"), None)
    if add_rule:
        op_pattern = engine.get_top_operator(add_rule.pattern_before)
        print(f"Top Operator (Pattern {add_rule.pattern_before}): {op_pattern}")
        
        # Manually check the logic
        arithmetic_ops = {"Add", "Sub", "Mul", "Mult", "Div", "Pow"}
        if op_expr in arithmetic_ops and op_pattern in arithmetic_ops:
            if op_expr != op_pattern:
                print(f"Should SKIP: {op_expr} != {op_pattern}")
            else:
                print(f"Should MATCH operator: {op_expr} == {op_pattern}")

def test_evaluator_logging():
    print("\n--- Test Evaluator Logging ---")
    script = """problem: 2^3 - 0
step: 2^3 - 0
step: 8
end: done"""
    
    parser = Parser(script)
    program = parser.parse()
    
    engine = SymbolicEvaluationEngine(SymbolicEngine())
    logger = LearningLogger()
    evaluator = Evaluator(program, engine, learning_logger=logger)
    
    evaluator.run()
    
    for record in logger.records:
        if record.phase == "step":
            print(f"Step Log: Expression='{record.expression}', Rendered='{record.rendered}'")

if __name__ == "__main__":
    test_strict_matching()
    test_evaluator_logging()
