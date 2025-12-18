
import pytest
from unittest.mock import MagicMock
from coherent.engine.multimodal.integrator import MultimodalIntegrator

def test_integrator_text_flow():
    integrator = MultimodalIntegrator()
    # Mock text encoder
    integrator.text_encoder.encode = MagicMock(return_value=[0.1, 0.2, 0.3])
    
    expr, vec = integrator.process_input("x^2 + 1", input_type="text")
    
    assert expr == "x^2 + 1"
    assert vec == [0.1, 0.2, 0.3]
    integrator.text_encoder.encode.assert_called_with("x^2 + 1")

def test_integrator_vision_flow():
    integrator = MultimodalIntegrator()
    # Mock vision encoder
    integrator.vision_encoder.image_to_latex = MagicMock(return_value="x+y")
    # Mock text encoder
    integrator.text_encoder.encode = MagicMock(return_value=[0.5, 0.5])
    
    expr, vec = integrator.process_input("img.png", input_type="image")
    
    assert expr == "x+y"
    assert vec == [0.5, 0.5]
    integrator.vision_encoder.image_to_latex.assert_called_with("img.png")
    integrator.text_encoder.encode.assert_called_with("x+y")
