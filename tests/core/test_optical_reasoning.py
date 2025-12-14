
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from causalscript.core.optical.vectorizer import FeatureExtractor
from causalscript.core.optical.layer import OpticalScoringLayer
from causalscript.core.ast_nodes import Add, Sym, Int, Mul
from causalscript.core.reasoning.generator import HypothesisGenerator
from causalscript.core.validation_engine import ValidationEngine
from causalscript.core.decision_theory import DecisionEngine, DecisionConfig

class TestOpticalCore:
    def test_feature_extractor_vectorize(self):
        extractor = FeatureExtractor(vector_size=64)
        
        # Create a simple AST: a + b + 1
        # Add(terms=[Sym('a'), Sym('b'), Int(1)])
        ast = Add(terms=[Sym('a'), Sym('b'), Int(1)])
        
        vector = extractor.vectorize(ast)
        
        assert vector.shape == (64,)
        # Check specific features if possible (Type counts)
        # Add is type 0, Sym is type 9, Int is type 7 (based on list order in vectorizer)
        # Indices might vary if class definition order changes, but let's check non-zero
        assert np.sum(vector) > 0

    def test_optical_layer_predict(self):
        layer = OpticalScoringLayer(input_dim=64, output_dim=10)
        vec = np.random.rand(64).astype(np.float32)
        
        scores, ambiguity = layer.predict(vec)
        
        assert scores.shape == (10,)
        assert isinstance(ambiguity, float)
        assert 0.0 <= ambiguity <= 1.0

    def test_optical_ambiguity_calculation(self):
        layer = OpticalScoringLayer(input_dim=64, output_dim=10)
        
        # High entropy input (uniform)
        intensity_uniform = np.ones(10)
        ambiguity_uniform = layer._calculate_ambiguity(intensity_uniform)
        assert ambiguity_uniform > 0.9 # Should be close to 1
        
        # Low entropy input (delta)
        intensity_delta = np.zeros(10)
        intensity_delta[0] = 10.0
        ambiguity_delta = layer._calculate_ambiguity(intensity_delta)
        assert ambiguity_delta < 0.1 # Should be close to 0

class TestIntegration:
    @pytest.fixture
    def mock_components(self):
        registry = MagicMock()
        registry.nodes = [MagicMock(id="rule1"), MagicMock(id="rule2")]
        registry.rules_by_id = {"rule1": MagicMock(), "rule2": MagicMock()}
        engine = MagicMock()
        return registry, engine

    def test_hypothesis_generator_integration(self, mock_components):
        registry, engine = mock_components
        # Mock match_rules to return something so we can check hypothesis creation
        registry.match_rules.return_value = [(MagicMock(id="rule1", description="desc", category="cat", priority=50), "x+1")]
        
        generator = HypothesisGenerator(registry, engine)
        
        # Mock optical layer to return specific ambiguity
        generator.optical_layer.predict = MagicMock(return_value=(np.zeros(10), 0.85))
        
        hypotheses = generator.generate("x")
        
        assert len(hypotheses) > 0
        h = hypotheses[0]
        assert h.metadata["ambiguity"] == 0.85

    def test_validation_engine_decision_integration(self):
        # mock dep
        decision_config = DecisionConfig(ambiguity_threshold=0.5)
        decision_engine = DecisionEngine(decision_config)
        fuzzy_judge = MagicMock()
        fuzzy_judge.judge_step.return_value = {'score': {'combined_score': 0.9}, 'label': MagicMock(value="exact")}
        
        comp_engine = MagicMock()
        # Force symbolic mismatch to trigger standard fuzzy+decision pipeline
        comp_engine.symbolic_engine.is_equiv.return_value = False
        
        val_engine = ValidationEngine(
            computation_engine=comp_engine,
            fuzzy_judge=fuzzy_judge,
            decision_engine=decision_engine
        )
        
        # Mock optical layer to return LOW ambiguity
        val_engine.optical_layer.predict = MagicMock(return_value=(None, 0.1))
        
        # Result should be valid (high match, low ambiguity)
        res = val_engine.validate_step("a", "b") # exprs dont matter with mocks
        assert res["valid"] is True
        
        # Mock optical layer to return HIGH ambiguity
        val_engine.optical_layer.predict = MagicMock(return_value=(None, 0.9))
        
        # Result should be invalid/review (high match, HIGH ambiguity)
        # With threshold 0.5, 0.9 > 0.5 -> Force REVIEW or REJECT
        res = val_engine.validate_step("a", "b")
        assert res["valid"] is False
        assert res["status"] in ["review", "mistake"]
