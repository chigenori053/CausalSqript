import torch
import torch.nn as nn
from typing import Optional, List
from causalscript.core.holographic.data_types import HolographicTensor, SpectrumConfig

class HolographicStorage:
    """
    Holographic Associative Memory.
    
    Principles:
    1. Superposition: Multiple items are stored in a single "Global Wave" by adding them together.
    2. Phase Encoding: Each item is shifted by a specific phase key before addition.
    3. Phase Filtering: Recovery is done by masking specific phase regions.
    """
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        # The global superposition wave (The "Hologram")
        # Registered as buffer if this were a Module, but here standard class is fine for V1.
        self.superposition_wave = torch.zeros(dimension, dtype=torch.complex64)
        
    def store_object(self, obj_spectrum: HolographicTensor, phase_key: float):
        """
        Encodes an object into the global wave with a specific phase key.
        Args:
            obj_spectrum: 1D HolographicTensor (complex)
            phase_key: Float angle in radians (e.g. 0 to 2pi) to shift the wave.
        """
        if obj_spectrum.shape[0] != self.dimension:
             raise ValueError(f"Dimension mismatch. Expected {self.dimension}, got {obj_spectrum.shape[0]}")
             
        # Phase Modulation (Shift)
        # Multiply by e^(i * theta)
        phase_shifter = torch.exp(torch.tensor(1j * phase_key, dtype=torch.complex64))
        modulated_wave = obj_spectrum * phase_shifter
        
        # Additive Storage (Interference/Superposition)
        self.superposition_wave += modulated_wave
        
    def extract_component(self, target_phase: float, bandwidth: float = 0.1) -> HolographicTensor:
        """
        Extracts a specific component using Phase Masking.
        Args:
            target_phase: The phase key used for storage.
            bandwidth: The angular width to filter around the target phase (tolerance).
        """
        # 1. Inspect current phases of the superposition wave
        current_phases = self.superposition_wave.angle() # Returns values in [-pi, pi]
        
        # Normalize target logic (optional, but assumes user passes values matching angle output)
        
        # Create Phase Mask
        # We handle wrapping logic simply here: checks absolute difference
        # In rigorous optics, this is a bandpass filter in phase domain
        
        # Simple logical mask: |current - target| < bandwidth
        # Handling circular wrapping: |a - b| > pi -> 2pi - |a - b|
        diff = torch.abs(current_phases - target_phase)
        wrapped_diff = torch.min(diff, 2 * torch.pi - diff)
        
        mask = (wrapped_diff <= bandwidth).type(torch.complex64)
        
        # 2. Apply Filter (Destructive Interference for non-matching phases)
        extracted_wave = self.superposition_wave * mask
        
        # 3. Demodulate (Shift back to original phase/zero phase align)
        # Multiply by e^(-i * theta)
        demodulator = torch.exp(torch.tensor(-1j * target_phase, dtype=torch.complex64))
        restored_wave = extracted_wave * demodulator
        
        return HolographicTensor(restored_wave)

    def clear(self):
        self.superposition_wave.zero_()
