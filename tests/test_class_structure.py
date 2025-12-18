import pytest
from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.engine.symbolic_engine import SymbolicEngine

def test_hybrid_usage():
    dsl = """
    meta:
        id: hybrid_demo
        topic: advanced_structure

    # 1. Class Definition Style (Main)
    problem: QuadraticSolver = a*x^2 + b*x + c

    prepare:
        - a = 1
        - b = 5
        - c = 6

    # 2. Variable Definition Style (Method: Discriminant)
    sub_problem: D = b^2 - 4*a*c
        step: 5^2 - 4*1*6
        step: 25 - 24
        end: 1

    # 3. Main Calculation using D
    # Verify D is available and used in calculation
    step: D*x^2 + 5*x + 6
    
    # 4. Legacy Style (Inline Replacement)
    sub_problem: 1*x^2
        step: x^2
        end: x^2

    step: x^2 + 5*x + 6
    end: done
    """
    
    parser = Parser(dsl)
    program = parser.parse()
    
    sym_engine = SymbolicEngine()
    engine = SymbolicEvaluationEngine(symbolic_engine=sym_engine)
    evaluator = Evaluator(program, engine)
    
    success = evaluator.run()
    assert success
    
    # Verify D was set (as string)
    assert engine._context.get("D") == "1"
    
    # Verify logs
    logs = evaluator.learning_logger.records
    
    # Check Problem log has scope
    prob_log = next(r for r in logs if r.phase == "problem")
    assert prob_log.meta["scope"] == "QuadraticSolver"
    
    # Check Sub-problem D
    sub_d_log = next(r for r in logs if r.phase == "sub_problem" and "D =" in r.expression)
    assert sub_d_log is not None
    
    # Check Sub-problem Legacy
    sub_legacy_log = next(r for r in logs if r.phase == "sub_problem" and "1*x^2" in r.expression)
    assert sub_legacy_log is not None

if __name__ == "__main__":
    test_hybrid_usage()
