
import pytest
from unittest.mock import MagicMock, patch
from causalscript.core.reasoning.agent import ReasoningAgent

# Patch where imported
@patch("causalscript.core.reasoning.agent.ChromaVectorStore")
@patch("causalscript.core.multimodal.integrator.TransformerEncoder") 
def test_recall_first_behavior(MockEncoder, MockStore):
    # Setup Mocks
    mock_store_instance = MockStore.return_value
    mock_store_instance.query.return_value = [] # Default empty
    
    mock_encoder_instance = MockEncoder.return_value
    mock_encoder_instance.encode.return_value = [0.1]
    
    mock_runtime = MagicMock()
    mock_runtime.knowledge_registry.apply_rule.return_value = "calculated_result" # Allow rule application
    
    agent = ReasoningAgent(mock_runtime)
    agent.generator = MagicMock() # Mock generator to verify it's NOT called if recall succeeds
    
    # CASE 1: No Memory -> Should call generator
    agent.think("x + 1")
    agent.generator.generate.assert_called_once()
    
    # CASE 2: Memory Hit -> Should NOT call generator (or print Recall log)
    # Reset
    agent.generator.generate.reset_mock()
    
    # Mock Memory Hit
    mock_store_instance.query.return_value = [{
        "id": "mem1",
        "metadata": {
            "original_expr": "_v + 1",
            "next_expr": "_v",
            "rule_id": "subtract_rule"
        }
    }]
    
    # Run think
    hyp = agent.think("y + 1")
    
    # Assert
    # Generator should NOT be called because memory provided a hypothesis and it passed verification
    agent.generator.generate.assert_not_called()
    assert hyp.rule_id == "subtract_rule"
    assert hyp.metadata["source"] == "memory"
