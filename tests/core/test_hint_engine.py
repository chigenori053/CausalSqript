import pytest
sympy = pytest.importorskip("sympy")
from coherent.engine.hint_engine import HintEngine, HintResult, HintCandidate, HintPersona
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.exercise_spec import ExerciseSpec

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
    result = hint_engine.generate_hint_for_spec("x**2 + 1", spec)
    assert result.hint_type == "pattern_match"
    assert result.message == "Did you forget the middle term?"

def test_heuristic_sign_error(hint_engine):
    spec = ExerciseSpec(id="test2", target_expression="x - 5")
    result = hint_engine.generate_hint_for_spec("-x + 5", spec)
    assert result.hint_type == "heuristic_sign_error"
    assert "sign error" in result.message.lower()

def test_heuristic_constant_offset(hint_engine):
    spec = ExerciseSpec(id="test3", target_expression="x + 10")
    result = hint_engine.generate_hint_for_spec("x + 12", spec)
    assert result.hint_type == "heuristic_constant_offset"
    assert float(result.details["offset"]) == 2.0

def test_fallback_hint(hint_engine):
    spec = ExerciseSpec(id="test4", target_expression="x**2")
    result = hint_engine.generate_hint_for_spec("x + 5", spec)
    assert result.hint_type == "none"

# --- V2 Tests ---

def test_generate_candidates(hint_engine):
    candidates = hint_engine.generate_candidates("-12", "10")
    types = [c.type for c in candidates]
    assert "heuristic_sign_error" in types or "heuristic_constant_offset" in types
    assert "none" in types

def test_persona_selection_sparta(hint_engine):
    candidates = [
        HintCandidate(content="Specific", type="specific", probability=0.9, source="rule"),
        HintCandidate(content="Vague", type="vague", probability=0.5, source="heuristic")
    ]
    result = hint_engine.select_best_hint(candidates, persona=HintPersona.SPARTA)
    assert result.message == "Specific"

def test_persona_selection_support(hint_engine):
    candidates = [
        HintCandidate(content="Specific", type="specific", probability=0.9, source="rule"),
        HintCandidate(content="Vague", type="vague", probability=0.5, source="heuristic")
    ]
    result = hint_engine.select_best_hint(candidates, persona=HintPersona.SUPPORT)
    assert result.message == "Specific"

def test_hint_engine_integration(hint_engine):
    spec = ExerciseSpec(
        id="test1",
        target_expression="x**2",
        hint_rules={"x*2": "Square, not double"}
    )
    result = hint_engine.generate_hint_for_spec("x*2", spec, persona="balanced")
    assert result.hint_type == "pattern_match"
    assert result.message == "Square, not double"
