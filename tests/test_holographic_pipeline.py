import pytest
import torch
import numpy as np
from coherent.engine.holographic.data_types import HolographicTensor, SpectrumConfig
from coherent.engine.multimodal.text_encoder import HolographicTextEncoder
from coherent.engine.multimodal.vision_encoder import HolographicVisionEncoder
from coherent.engine.multimodal.audio_encoder import HolographicAudioEncoder
from coherent.optical.layer import OpticalInterferenceEngine
from coherent.engine.holographic.memory import HolographicStorage
from coherent.engine.holographic.orchestrator import HolographicOrchestrator

# Mock existing dependencies if necessary
# Assuming sentence-transformers and Pillow might be missing in test env, 
# the encoders have fallback logic to return zero tensors. 
# We test that structure first.

def test_holographic_tensor_properties():
    """Verify HolographicTensor behaves like a complex tensor."""
    t = torch.randn(10, dtype=torch.complex64)
    ht = HolographicTensor(t)
    assert isinstance(ht, torch.Tensor)
    assert ht.is_complex()
    assert ht.dtype == torch.complex64

def test_text_encoder_output():
    """Test Text Encoder produces correct shape HolographicTensor."""
    encoder = HolographicTextEncoder(model_name="all-MiniLM-L6-v2")
    # Note: If sentence-transformers is missing, it logs warning and returns zeros.
    # This is acceptable for structural test.
    
    text = "Hello Holographic World"
    output = encoder.encode(text)
    
    assert isinstance(output, torch.Tensor)
    assert output.is_complex()
    assert output.shape[0] == SpectrumConfig.DIMENSION

def test_vision_encoder_output():
    """Test Vision Encoder logic."""
    encoder = HolographicVisionEncoder()
    # Mocking image input with a path that doesn't exist returns zeros
    output = encoder.encode("non_existent_image.jpg")
    
    assert isinstance(output, torch.Tensor)
    assert output.is_complex()
    assert output.shape[0] == SpectrumConfig.DIMENSION

def test_audio_encoder_output():
    """Test Audio Encoder logic."""
    encoder = HolographicAudioEncoder()
    # Mock waveform: 16000 samples (1 sec)
    waveform = torch.randn(16000)
    output = encoder.encode(waveform)
    
    assert isinstance(output, torch.Tensor)
    assert output.is_complex()
    assert output.shape[0] == SpectrumConfig.DIMENSION

def test_optical_interference_engine():
    """Test Optical Engine forward pass."""
    batch_size = 5
    dim = SpectrumConfig.DIMENSION
    memory_cap = 50
    
    engine = OpticalInterferenceEngine(memory_capacity=memory_cap, input_dim=dim)
    
    # Create random complex input batch
    input_batch = torch.randn(batch_size, dim, dtype=torch.complex64)
    
    resonance = engine(input_batch)
    
    assert resonance.shape == (batch_size, memory_cap)
    assert not resonance.is_complex() # Should be real energy
    assert (resonance >= 0).all() # Energy is positive

    ambiguity = engine.get_ambiguity(resonance)
    assert 0.0 <= ambiguity <= 1.0

def test_holographic_memory_storage_recall():
    """Test Memory superposition and cleanup."""
    dim = SpectrumConfig.DIMENSION
    memory = HolographicStorage(dimension=dim)
    
    # Create a signal
    signal = torch.zeros(dim, dtype=torch.complex64)
    signal[0] = 1 + 0j # Simple impulse
    
    # Store with phase 0
    memory.store_object(HolographicTensor(signal), phase_key=0.0)
    
    # Retrieve
    recovered = memory.extract_component(target_phase=0.0, bandwidth=0.1)
    
    # Check fidelity (should be close to original for single item)
    # Note: Phase filtering extraction multiplies by mask, so indices unmatched are zeroed.
    # Our signal is at index 0.
    assert torch.abs(recovered[0] - signal[0]) < 1e-5
    
    # Store another orthogonal item with phase pi
    signal2 = torch.zeros(dim, dtype=torch.complex64)
    signal2[1] = 1 + 0j
    memory.store_object(HolographicTensor(signal2), phase_key=3.14159)
    
    # Retrieve first item again
    recovered_1 = memory.extract_component(target_phase=0.0)
    
    # Ideally index 1 should be filtered out because it was stored at phase pi
    # But mask bandwidth matters.
    # Signal2 at index 1 was multiplied by e^(i*pi) = -1.
    # Superposition: [1, -1, 0...]
    # Extract phase 0: Mask checks angle of superposition. 
    # Angle at index 0 is 0. Angle at index 1 is pi.
    # Mask should keep index 0 and remove index 1.
    
    assert torch.abs(recovered_1[0]) > 0.9 # Kept
    assert torch.abs(recovered_1[1]) < 0.1 # Filtered out

def test_orchestrator_integration():
    """Test full pipeline via Orchestrator."""
    # We mock the inputs
    orchestrator = HolographicOrchestrator()
    
    # Input data with mocks
    # Audio encoder expects tensor/array
    waveform = np.random.randn(16000)
    
    inputs = {
        'text': "Test input",
        'audio': waveform
        # Vision skipped for simplicity of test env (no image file)
    }
    
    results = orchestrator.process_input(inputs)
    
    assert 'resonance_energy' in results
    assert 'ambiguity' in results
    
    # Check working memory has stored 2 items
    # (Text and Audio)
    # We can't easily check count in superposition without tracking, but we can check it's not zero
    assert orchestrator.working_memory.superposition_wave.abs().sum() > 0
