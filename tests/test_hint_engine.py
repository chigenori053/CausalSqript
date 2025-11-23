import pytest
from core.hint_engine import HintEngine, HintResult
from core.computation_engine import ComputationEngine
from core.symbolic_engine import SymbolicEngine
from core.exercise_spec import ExerciseSpec

@pytest.fixture
def engine():
    symbolic = SymbolicEngine()
    return ComputationEngine(symbolic)

@pytest.fixture
def hint_engine(engine):
    return HintEngine(engine)

def test_hint_result_structure():
    result = HintResult(message="Test hint", hint_type="test")
    assert result.message == "Test hint"
    assert result.hint_type == "test"
    assert result.details == {}

def test_pattern_matching_hint(hint_engine):
    spec = ExerciseSpec(
        id="test1",
        target_expression="x**2 + 2*x + 1",
        hint_rules={
            "x**2 + 1": "Did you forget the middle term?",
            "x**2 + 2*x": "Don't forget the constant term."
        }
    )
    
    # Test first pattern
    result = hint_engine.generate_hint_for_spec("x**2 + 1", spec)
    assert result.hint_type == "pattern_match"
    assert result.message == "Did you forget the middle term?"
    
    # Test second pattern
    result = hint_engine.generate_hint_for_spec("x**2 + 2*x", spec)
    assert result.hint_type == "pattern_match"
    assert result.message == "Don't forget the constant term."

def test_heuristic_sign_error(hint_engine):
    spec = ExerciseSpec(
        id="test2",
        target_expression="x - 5"
    )
    
    # User enters -(x - 5) which is -x + 5
    result = hint_engine.generate_hint_for_spec("-x + 5", spec)
    assert result.hint_type == "heuristic_sign_error"
    assert "sign error" in result.message.lower()

def test_heuristic_constant_offset(hint_engine):
    spec = ExerciseSpec(
        id="test3",
        target_expression="x + 10"
    )
    
    # User enters x + 12 (off by 2)
    result = hint_engine.generate_hint_for_spec("x + 12", spec)
    assert result.hint_type == "heuristic_constant_offset"
    assert "constant amount" in result.message
    assert float(result.details["offset"]) == 2.0

def test_fallback_hint(hint_engine):
    spec = ExerciseSpec(
        id="test4",
        target_expression="x**2"
    )
    
    # Random wrong answer
    result = hint_engine.generate_hint_for_spec("x + 5", spec)
    assert result.hint_type == "none"
    assert "checking your steps" in result.message.lower()

def test_syntax_error_hint(hint_engine):
    spec = ExerciseSpec(
        id="test5",
        target_expression="x"
    )
    
    result = hint_engine.generate_hint_for_spec("x +", spec)
    assert result.hint_type == "syntax_error"

def test_pattern_matching_with_equivalence(hint_engine):
    # Pattern matching should work even if user input is not identical string but equivalent
    spec = ExerciseSpec(
        id="test6",
        target_expression="x**2",
        hint_rules={
            "2*x": "You differentiated instead of squaring."
        }
    )
    
    # User enters x*2 which is equivalent to 2*x
    result = hint_engine.generate_hint_for_spec("x*2", spec)
    assert result.hint_type == "pattern_match"
    assert result.message == "You differentiated instead of squaring."
