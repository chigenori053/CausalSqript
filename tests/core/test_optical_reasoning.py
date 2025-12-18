
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from coherent.optical.vectorizer import FeatureExtractor
from coherent.optical.layer import OpticalInterferenceEngine
from coherent.engine.ast_nodes import Add, Sym, Int, Mul
from coherent.engine.reasoning.generator import HypothesisGenerator
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.decision_theory import DecisionEngine, DecisionConfig

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
        # input_dim=64, memory_capacity (formerly output_dim)=10
        layer = OpticalInterferenceEngine(input_dim=64, memory_capacity=10)
        import torch
        vec = torch.randn(64) # Use torch tensor
        
        # Call layer directly (forward)
        # Returns only intensity in new API
        scores = layer(vec)
        # Calculate ambiguity separately
        ambiguity = layer.get_ambiguity(scores)
        
        assert scores.shape == (1, 10) # [Batch, Out]
        assert isinstance(ambiguity, float)
        assert 0.0 <= ambiguity <= 1.0

    def test_optical_ambiguity_calculation(self):
        layer = OpticalInterferenceEngine(input_dim=64, memory_capacity=10)
        import torch
        
        # High entropy input (uniform) must be Tensor [Batch, Capacity]
        intensity_uniform = torch.ones(1, 10)
        ambiguity_uniform = layer.get_ambiguity(intensity_uniform)
        assert ambiguity_uniform > 0.9 # Should be close to 1
        
        # Low entropy input (delta)
        intensity_delta = torch.zeros(1, 10)
        intensity_delta[0, 0] = 10.0
        ambiguity_delta = layer.get_ambiguity(intensity_delta)
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
        import torch
        generator.optical_layer = MagicMock()
        # Mocking __call__ directly. 
        # It must return intensity now
        generator.optical_layer.return_value = torch.zeros(10)
        # Mock get_ambiguity separately
        generator.optical_layer.get_ambiguity.return_value = 0.85
        
        hypotheses = generator.generate("x")
        
        assert len(hypotheses) > 0
        h = hypotheses[0]
        assert h.metadata["ambiguity"] == 0.85

    def test_validation_engine_decision_integration(self):
        # mock dep
        # decision_config = DecisionConfig(ambiguity_threshold=0.5)
        # decision_engine = DecisionEngine(decision_config)
        from coherent.engine.decision_theory import DecisionAction
        decision_engine = MagicMock()
        
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
        import torch
        val_engine.optical_layer = MagicMock()
        val_engine.optical_layer.return_value = torch.zeros(10) # Dummy intensity
        val_engine.optical_layer.get_ambiguity.return_value = 0.1
        
        # Mock Decision Engine to ACCEPT
        decision_engine.decide.return_value = (DecisionAction.ACCEPT, 1.0, {})

        # Result should be valid (high match, low ambiguity)
        res = val_engine.validate_step("a", "b") # exprs dont matter with mocks
        assert res["valid"] is True
        
        # Mock optical layer to return HIGH ambiguity
        val_engine.optical_layer.get_ambiguity.return_value = 0.9
        
        # Mock Decision Engine to REJECT based on high ambiguity (simulated)
        decision_engine.decide.return_value = (DecisionAction.REJECT, 0.0, {})

        # Result should be invalid/review (high match, HIGH ambiguity)
        # With threshold 0.5, 0.9 > 0.5 -> Force REVIEW or REJECT
        res = val_engine.validate_step("a", "b")
        assert res["valid"] is False
        assert res["status"] in ["review", "mistake"]
