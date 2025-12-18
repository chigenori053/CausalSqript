import pytest
import torch
import numpy as np
from coherent.optical.layer import OpticalInterferenceEngine

def test_optical_layer_initialization():
    layer = OpticalInterferenceEngine(input_dim=10, memory_capacity=5)
    assert layer.input_dim == 10
    assert layer.memory_capacity == 5
    # Weights are now 'optical_memory'
    assert layer.optical_memory.shape == (5, 10)
    assert layer.optical_memory.dtype == torch.cfloat
    assert layer.optical_memory.requires_grad

def test_optical_layer_forward_shape():
    layer = OpticalInterferenceEngine(input_dim=10, memory_capacity=5)
    # Batch size 1
    input_tensor = torch.randn(10)
    intensity = layer(input_tensor)
    ambiguity = layer.get_ambiguity(intensity)
    
    assert intensity.shape == (1, 5)
    # Intensity should be real and non-negative
    assert intensity.dtype == torch.float32 or intensity.dtype == torch.float64
    assert torch.all(intensity >= 0)
    
    assert isinstance(ambiguity, float)
    assert 0.0 <= ambiguity <= 1.0

def test_optical_layer_forward_batch():
    layer = OpticalInterferenceEngine(input_dim=10, memory_capacity=5)
    # Batch size 3
    input_tensor = torch.randn(3, 10)
    intensity = layer(input_tensor)
    ambiguity = layer.get_ambiguity(intensity)
    
    assert intensity.shape == (3, 5)
    # Ambiguity is currently calc based on [0] for single inference compatibility, 
    # ensuring it doesn't crash on batch is the goal here.
    assert isinstance(ambiguity, float)

def test_optical_layer_backward():
    layer = OpticalInterferenceEngine(input_dim=10, memory_capacity=5)
    input_tensor = torch.randn(10, dtype=torch.cfloat)
    
    # Forward
    intensity = layer(input_tensor)
    
    # Loss (minimize sum)
    loss = intensity.sum()
    
    # Backward
    layer.zero_grad()
    loss.backward()
    
    assert layer.optical_memory.grad is not None
    # Gradient should be complex since weights are complex
    assert layer.optical_memory.grad.dtype == torch.cfloat
