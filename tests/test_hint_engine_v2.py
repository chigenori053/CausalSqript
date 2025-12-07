
import pytest
sympy = pytest.importorskip("sympy")
from causalscript.core.hint_engine import HintEngine, HintResult, HintCandidate, HintPersona
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.exercise_spec import ExerciseSpec

@pytest.fixture
def engine():
    symbolic = SymbolicEngine()
    return ComputationEngine(symbolic)

@pytest.fixture
def hint_engine(engine):
    return HintEngine(engine)

def test_generate_candidates(hint_engine):
    # Test that multiple candidates are generated
    # Case: Sign error AND Constant offset (e.g. target=10, user=-12)
    # -12 vs -10 (sign error candidate)
    # -12 vs 10 (diff is -22, constant offset candidate)
    
    candidates = hint_engine.generate_candidates("-12", "10")
    
    types = [c.type for c in candidates]
    assert "heuristic_sign_error" in types or "heuristic_constant_offset" in types
    assert "none" in types # Fallback always there

def test_persona_selection_sparta(hint_engine):
    # Sparta prefers less help (or high self-correction utility).
    # Actually my implementation of Sparta:
    # u_self_bonus = 50.0 (High value for self-solve)
    # c_giveup = -20.0 (Low penalty for giveup)
    # Abstract hints (if I had them) would be preferred.
    # Currently I only have specific hints.
    # But let's check if it selects a hint at all.
    
    candidates = [
        HintCandidate(content="Specific", type="specific", probability=0.9, source="rule"),
        HintCandidate(content="Vague", type="vague", probability=0.5, source="heuristic")
    ]
    
    # In my current implementation:
    # Specific: U_self=0. P=0.9*0.9=0.81. EU = 0.81*100 + 0.19*(-20) = 81 - 3.8 = 77.2
    # Vague: U_self=50. P=0.5*0.5=0.25. EU = 0.25*(150) + 0.75*(-20) = 37.5 - 15 = 22.5
    # So it still prefers Specific because probability is so high.
    
    result = hint_engine.select_best_hint(candidates, persona=HintPersona.SPARTA)
    assert result.message == "Specific"

def test_persona_selection_support(hint_engine):
    # Support: c_giveup = -100. High penalty for failure.
    # Should definitely pick high probability hint.
    
    candidates = [
        HintCandidate(content="Specific", type="specific", probability=0.9, source="rule"),
        HintCandidate(content="Vague", type="vague", probability=0.5, source="heuristic")
    ]
    
    # Specific: EU = 0.81*100 + 0.19*(-100) = 81 - 19 = 62
    # Vague: U_self=10. EU = 0.25*110 + 0.75*(-100) = 27.5 - 75 = -47.5
    
    result = hint_engine.select_best_hint(candidates, persona=HintPersona.SUPPORT)
    assert result.message == "Specific"

def test_hint_engine_integration(hint_engine):
    # Test the full flow
    spec = ExerciseSpec(
        id="test1",
        target_expression="x**2",
        hint_rules={"x*2": "Square, not double"}
    )
    
    result = hint_engine.generate_hint_for_spec("x*2", spec, persona="balanced")
    assert result.hint_type == "pattern_match"
    assert result.message == "Square, not double"
