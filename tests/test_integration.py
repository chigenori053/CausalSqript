import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.exercise_spec import ExerciseSpec
from coherent.engine.evaluator import Evaluator
from coherent.engine import ast_nodes as ast

@pytest.fixture
def runtime():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    validation = ValidationEngine(computation)
    hint = HintEngine(computation)
    
    return CoreRuntime(computation, validation, hint)

def test_evaluator_with_core_runtime_success(runtime):
    # Program:
    # Problem: x + x
    # Step: 2*x
    # End: done
    
    program = ast.ProgramNode(body=[
        ast.ProblemNode(expr="x + x"),
        ast.StepNode(expr="2*x"),
        ast.EndNode(expr="done", is_done=True)
    ])
    
    evaluator = Evaluator(program, runtime)
    success = evaluator.run()
    
    assert success is True
    assert runtime._current_expr == "2*x"

def test_evaluator_with_core_runtime_failure_and_hint(runtime):
    # Setup spec with hint
    spec = ExerciseSpec(
        id="test_int",
        target_expression="x**2",
        hint_rules={"2*x": "Differentiation detected"}
    )
    runtime.exercise_spec = spec
    
    # Program:
    # Problem: x**2
    # End: 2*x
    
    program = ast.ProgramNode(body=[
        ast.ProblemNode(expr="x**2"),
        ast.EndNode(expr="2*x", is_done=False)
    ])
    
    evaluator = Evaluator(program, runtime)
    success = evaluator.run()
    
    # Should complete successfully (program ran to end)
    assert success is True
    
    # Check logs for hint and mistake status
    logs = evaluator.learning_logger.records
    end_log = logs[-1]
    assert end_log["phase"] == "end"
    assert end_log["status"] == "mistake"
    assert "hint" in end_log["meta"]
    assert end_log["meta"]["hint"]["message"] == "Differentiation detected"
