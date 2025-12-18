import pytest
from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator, SymbolicEvaluationEngine
from coherent.engine.symbolic_engine import SymbolicEngine

def test_hierarchical_structure():
    source = """
    problem: x + 1
    prepare:
        x = 2
    step: 3
    end: done
    
    problem: y * 2
    prepare:
        y = 5
    step: 10
    end: 10
    """
    
    parser = Parser(source)
    program = parser.parse()
    
    assert len(program.body) == 2
    
    prob1 = program.body[0]
    assert prob1.expr == "x + 1"
    assert prob1.prepare.statements == ["x = 2"]
    assert len(prob1.steps) == 2
    assert prob1.steps[0].expr == "3"
    assert prob1.end_node.is_done
    
    prob2 = program.body[1]
    assert prob2.expr == "y*2"
    assert prob2.prepare.statements == ["y = 5"]
    assert len(prob2.steps) == 2
    assert prob2.steps[0].expr == "10"
    assert prob2.end_node.expr == "10"

def test_execution_multi_problem():
    source = """
    problem: x + 2
    prepare:
        x = 3
    step: 5
    end: done

    problem: x * 3
    prepare:
        x = 4  # New context for x
    step: 12
    end: done
    """
    
    parser = Parser(source)
    program = parser.parse()
    
    symbolic_engine = SymbolicEngine()
    # Mock knowledge matching if needed, or assume simple arithmetic works without rules
    engine = SymbolicEvaluationEngine(symbolic_engine)
    evaluator = Evaluator(program, engine)
    
    success = evaluator.run()
    assert success
    
    # Verify the context switching
    # Engine context ends with the last problem's context
    assert engine._context["x"] == 4

def test_step_block_parsing():
    source = """
    problem: a + b
    prepare:
        a = 1
        b = 2
    step:
        before: a + b
        after: 3
        note: Addition
    end: done
    """
    parser = Parser(source)
    program = parser.parse()
    prob = program.body[0]
    step = prob.steps[0]
    
    assert step.expr == "3"
    assert step.before_expr == "a + b"
    assert step.note == "Addition"

def test_missing_problem_error():
    source = """
    step: 1
    """
    parser = Parser(source)
    with pytest.raises(Exception): # SyntaxError
         program = parser.parse()
