
import pytest
from unittest.mock import MagicMock
from coherent.engine.multimodal.integrator import MultimodalIntegrator

def test_integrator_text_flow():
    integrator = MultimodalIntegrator()
    # Mock text encoder
    integrator.text_encoder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
    
    expr, vec = integrator.process_input("x^2 + 1", input_type="text")
    
    assert expr == "x**2 + 1"
    assert vec == [0.1, 0.2, 0.3]
    integrator.text_encoder.encode.assert_called_with("x**2 + 1")

def test_integrator_vision_flow():
    integrator = MultimodalIntegrator()
    # Mock vision encoder
    integrator.vision_encoder.encode = MagicMock(return_value=MagicMock(tensor='mocked'))
    # Mock text encoder
    integrator.text_encoder.encode = MagicMock(return_value=[0.5, 0.5])
    
    expr, vec = integrator.process_input("img.png", input_type="image")
    
    
    # Logic update: integrator with "image" input returns None expression unless vision encoder decodes text.
    assert expr is None
    # Check if vec is the mock object returned by vision encoder
    assert vec.tensor == 'mocked'
    integrator.vision_encoder.encode.assert_called_with("img.png")
    # Text encoder is not called for pure image input in new flow

