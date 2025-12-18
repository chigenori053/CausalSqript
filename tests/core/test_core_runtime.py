import math
import pytest
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.exercise_spec import ExerciseSpec
from coherent.engine.errors import MissingProblemError

@pytest.fixture
def runtime():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    validation = ValidationEngine(computation)
    hint = HintEngine(computation)
    
    return CoreRuntime(computation, validation, hint)

def test_set_problem(runtime):
    runtime.set("x + 1")
    assert runtime._current_expr == "x + 1"

def test_check_step_valid(runtime):
    runtime.set("x + x")
    result = runtime.check_step("2*x")
    assert result["valid"] is True
    assert runtime._current_expr == "2*x"

def test_check_step_invalid(runtime):
    runtime.set("x + x")
    result = runtime.check_step("3*x")
    assert result["valid"] is False
    assert runtime._current_expr == "x + x"  # Should not update

def test_check_step_missing_problem(runtime):
    with pytest.raises(MissingProblemError):
        runtime.check_step("x")

def test_finalize_without_spec(runtime):
    runtime.set("x + 1")
    # Check against equivalent expression
    result = runtime.finalize("1 + x")
    assert result["valid"] is True
    
    # Check against non-equivalent
    result = runtime.finalize("x + 2")
    assert result["valid"] is False

def test_finalize_with_spec_correct(runtime):
    spec = ExerciseSpec(
        id="test1",
        target_expression="2*x",
        validation_mode="symbolic_equiv"
    )
    runtime.exercise_spec = spec
    runtime.set("x + x")
    
    result = runtime.finalize("2*x")
    assert result["valid"] is True
    assert "Correct!" in result["details"]["message"]

def test_finalize_with_spec_incorrect_with_hint(runtime):
    spec = ExerciseSpec(
        id="test2",
        target_expression="x**2",
        hint_rules={"2*x": "You differentiated instead of squaring."}
    )
    runtime.exercise_spec = spec
    runtime.set("x**2") # Initial state doesn't matter much for finalize check against spec
    
    # User answers 2*x
    result = runtime.finalize("2*x")
    assert result["valid"] is False
    assert "hint" in result["details"]
    assert result["details"]["hint"]["message"] == "You differentiated instead of squaring."

def test_variable_binding(runtime):
    runtime.set_variable("a", 10)
    assert runtime.evaluate("a + 5") == 15
    
    # Check if binding affects step checking (it should if variables are used)
    runtime.set("a + x")
    result = runtime.check_step("10 + x")
    assert result["valid"] is True


def test_runtime_function_analysis(runtime):
    analysis = runtime.analyze_function("x**2 + 1")
    assert analysis["domain"]["type"] == "all_real"
    assert "range" in analysis
    assert analysis["intercepts"]["y"] == pytest.approx(1.0)


def test_runtime_plot_data(runtime):
    plot_result = runtime.plot_function("x", start=-1, end=1, num_points=4)
    assert len(plot_result["x"]) == len(plot_result["y"]) == 4
    assert plot_result["x"][0] == -1


def test_runtime_describe_dataset(runtime):
    stats = runtime.describe_dataset([1, 2, 3, 4])
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["count"] == 4


def test_runtime_probability(runtime):
    result = runtime.compute_probability("normal", 0, params={"mean": 0, "std": 1})
    assert result["distribution"] == "normal"
    assert result["cdf"] == pytest.approx(0.5, rel=1e-3)


def test_runtime_visualize_dataset(runtime):
    viz = runtime.visualize_dataset([1, 2, 3, 4], bins=2)
    assert len(viz["bins"]) == 2
    assert sum(bin_info["count"] for bin_info in viz["bins"]) == 4


def test_runtime_trig_helpers(runtime):
    point = runtime.trig_unit_circle(90, unit="degrees")
    assert point["sin"] == 1.0
    assert runtime.trig_evaluate("sin", math.pi / 6) == pytest.approx(0.5, rel=1e-6)
    sum_identity = runtime.trig_apply_identity("sin_sum", math.pi / 6, other_angle=math.pi / 3)
    assert sum_identity == pytest.approx(1.0, rel=1e-6)


def test_runtime_calculus(runtime):
    derivative = runtime.calc_derivative("x**2", at=2)
    assert derivative["value"] == pytest.approx(4.0, rel=1e-3)
    integral = runtime.calc_integral("x", lower=0, upper=2)
    assert integral["value"] == pytest.approx(2.0, rel=1e-2)


def test_runtime_linear_algebra(runtime):
    vec = runtime.vector_add([1, 2], [3, 4])
    assert vec == [4, 6]
    dot = runtime.vector_dot([1, 2, 3], [4, 5, 6])
    assert dot == pytest.approx(32)
    mat = runtime.matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert mat == [[19, 22], [43, 50]]
    solution = runtime.solve_linear_system([[2, 1], [1, -1]], [4, 1])
    assert solution[0] == pytest.approx(5 / 3, rel=1e-6)
    eigenvals = runtime.matrix_eigenvalues([[2, 0], [0, 3]])
    assert set(round(val, 6) for val in eigenvals) == {2.0, 3.0}


def test_equation_step_support(runtime):
    runtime.set("2*x + 3 = 7")
    assert runtime._current_expr == "(2*x + 3) - (7)"
    step1 = runtime.check_step("2*x = 4")
    assert step1["valid"] is True
    step2 = runtime.check_step("x = 2")
    assert step2["valid"] is True

def test_review_hint_generation(runtime):
    """Test that 'review' status triggers encouraging hint."""
    from unittest.mock import MagicMock
    
    # Mock ValidationEngine to return REVIEW status
    # We must ensure CoreRuntime delegates to it and then generates hint
    mock_details = {"status": "review", "review_needed": True}
    runtime.validation_engine.validate_step = MagicMock(return_value={
        "valid": False, 
        "status": "review", 
        "details": mock_details
    })
    
    # Set problem first
    runtime.set("p")
    
    # Execute step (content doesn't matter due to mock)
    result = runtime.check_step("anything")
    
    assert result["valid"] is False
    assert result["details"]["status"] == "review"
    assert "hint" in result["details"]
    assert result["details"]["hint"]["type"] == "review_encouragement"
    assert "extremely close" in result["details"]["hint"]["message"]
