
import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

from coherent.core.optical.validator import OpticalValidator, OpticalResult
from coherent.core.validation_engine import ValidationEngine
from coherent.core.computation_engine import ComputationEngine

def test_optical_validator_structure_to_vector():
    """Verify that structure_to_vector is deterministic."""
    validator = OpticalValidator(input_dim=64)
    expr = "x + 2*y"
    
    vec1 = validator._structure_to_vector(expr)
    vec2 = validator._structure_to_vector(expr)
    
    assert vec1 == vec2
    assert len(vec1) == 64

def test_optical_validator_resonance():
    """Verify resonance scoring logic."""
    validator = OpticalValidator(input_dim=64)
    
    # Mocking _structure_to_vector to return controlled vectors
    # Case 1: Identical vectors -> High Resonance
    with patch.object(validator, '_structure_to_vector') as mock_vec:
        # Create a detailed complex vector simulation
        # Using real vectors, from_real_vector does FFT
        # If inputs are identical, FFTs are identical, Dot product is MAX.
        
        vec = [1.0] * 64
        mock_vec.return_value = vec
        
        # Identity check
        res = validator.validate("x", "x")
        assert res.is_resonant
        assert res.status == "accept"
        assert res.resonance_score > 0.9

def test_validation_engine_integration_parallel():
    """Verify ValidationEngine uses OpticalValidator and Parallel logic."""
    
    # Mock ComputationEngine
    comp_engine = MagicMock(spec=ComputationEngine)
    comp_engine.symbolic_engine = MagicMock()
    
    # Instantiate Engine
    engine = ValidationEngine(computation_engine=comp_engine)
    
    # Mock the newly verified OpticalValidator inside the engine
    # (It's lazily loaded, so we inject it or mock the import)
    
    mock_opt_validator = MagicMock()
    engine.optical_validator = mock_opt_validator
    
    # Case 1: Optical Accept, Symbolic Accept (Ideal)
    mock_opt_validator.parallel_validate.return_value = (
        OpticalResult(True, "accept", 0.95, 1.0),
        True # Sympy result
    )
    
    result = engine.validate_step("x+x", "2x")
    assert result["valid"] is True
    assert result["status"] == "correct"
    assert result["details"]["method"] == "optical_parallel"
    
    # Case 2: Optical Review, Symbolic Accept (Fallback works)
    mock_opt_validator.parallel_validate.return_value = (
        OpticalResult(True, "review", 0.6, 0.5),
        True
    )
    result = engine.validate_step("x+x", "2x")
    assert result["valid"] is True
    assert result["status"] == "correct"
    assert result["details"]["method"] == "symbolic_fallback"

    # Case 3: Optical Reject, Symbolic Accept (Rescue)
    mock_opt_validator.parallel_validate.return_value = (
        OpticalResult(False, "reject", 0.1, 1.0),
        True
    )
    result = engine.validate_step("mask assumption", "valid math")
    assert result["valid"] is True
    assert result["status"] == "correct"
    assert result["details"]["method"] == "symbolic_rescue"
