import pytest
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.computation_engine import ComputationEngine
from causalscript.core.symbolic_engine import SymbolicEngine
from causalscript.core.fuzzy.judge import FuzzyJudge
from causalscript.core.causal.causal_engine import CausalEngine
from causalscript.core.decision_theory import DecisionEngine, DecisionConfig, DecisionAction
from causalscript.core.fuzzy.encoder import ExpressionEncoder
from causalscript.core.fuzzy.metric import SimilarityMetric
from causalscript.core.fuzzy.types import FuzzyResult, FuzzyScore, FuzzyLabel
from causalscript.core.knowledge_registry import KnowledgeRegistry
from unittest.mock import MagicMock

@pytest.fixture
def engines():
    symbolic = SymbolicEngine()
    computation = ComputationEngine(symbolic)
    encoder = ExpressionEncoder()
    metric = SimilarityMetric()
    fuzzy = FuzzyJudge(encoder=encoder, metric=metric, symbolic_engine=symbolic)
    decision = DecisionEngine(DecisionConfig(strategy="balanced"))
    causal = CausalEngine(decision_config=DecisionConfig(strategy="balanced"))
    knowledge = MagicMock(spec=KnowledgeRegistry)
    # Default suggest_outcomes to empty list so it doesn't interfere with other tests
    knowledge.suggest_outcomes.return_value = []
    
    validation = ValidationEngine(
        computation_engine=computation,
        fuzzy_judge=fuzzy,
        causal_engine=causal,
        decision_engine=decision,
        knowledge_registry=knowledge
    )
    # Mock Optical Layer to return low ambiguity (0.0) so it doesn't interfere with symbolic/fuzzy tests
    validation.optical_layer = MagicMock()
    validation.optical_layer.predict.return_value = (None, 0.0)
    
    return validation

def test_validate_step_symbolic_correct(engines):
    """Test standard symbolic correctness."""
    result = engines.validate_step("x + x", "2*x")
    assert result["valid"] is True
    assert result["status"] == "correct"

def test_validate_step_symbolic_incorrect(engines):
    """Test standard symbolic incorrectness."""
    result = engines.validate_step("x + x", "3*x")
    # Should rely on fuzzy/decision now.
    # 3*x is not fuzzy equivalent to 2*x.
    assert result["valid"] is False
    assert result["status"] == "mistake"

def test_validate_step_fuzzy_decision_accept(engines):
    """Test that fuzzy match leads to ACCEPT via Decision Engine."""
    # We force Symbolic Fail but Fuzzy Pass by mocking is_equiv
    engines.symbolic_engine.is_equiv = MagicMock(return_value=False)
    
    # Mock FuzzyJudge to return high score, bypassing internal weight logic
    # We simulate a "perfect" fuzzy match
    # Use plain dicts to ensure runtime compatibility
    mock_score = {"combined_score": 0.95, "expr_similarity": 1.0, "rule_similarity": 1.0, "text_similarity": 1.0}
    mock_result = {
        "label": FuzzyLabel.EXACT,
        "score": mock_score,
        "reason": "Mocked",
        "debug": {}
    }
    engines.fuzzy_judge.judge_step = MagicMock(return_value=mock_result)

    # Use identical strings for high fuzzy score inputs (content doesn't matter due to mock)
    result = engines.validate_step("x", "x")
    
    assert result["valid"] is True
    assert result["status"] == "correct"
    assert "fuzzy_score" in result["details"]
    assert result["details"]["fuzzy_score"] == 0.95

def test_validate_step_review_mode(engines):
    """Test REVIEW status (Action.REVIEW)."""
    # This requires a score that falls into the REVIEW bucket of the Utility Matrix.
    # We mock decision engine to force REVIEW action
    engines.symbolic_engine.is_equiv = MagicMock(return_value=False)
    engines.decision_engine.decide = MagicMock(return_value=(DecisionAction.REVIEW, 10.0, {}))
    
    # Inputs don't matter as we mocked the decision
    result = engines.validate_step("a", "b")
    
    assert result["valid"] is False  # REVIEW maps to False (status="review") in current impl
    assert result["status"] == "review"

def test_validate_step_predictive_skip(engines):
    """Test that predictive skip bypasses symbolic check."""
    # Mock knowledge registry to predict "x+1" from "x"
    engines.knowledge_registry.suggest_outcomes.return_value = ["x + 1"]
    
    # Mock symbolic engine to fail equivalence (to prove we skipped it or it wasn't needed)
    # Actually, if we skip, is_equiv shouldn't be called.
    engines.symbolic_engine.is_equiv = MagicMock(side_effect=Exception("Should not be called"))
    
    # Test valid prediction
    result = engines.validate_step("x", "x + 1")
    
    assert result["valid"] is True
    assert result["status"] == "correct"
    assert "details" in result
    assert result["details"]["method"] == "predictive_skip"
    assert result["details"]["candidate"] == "x + 1"
