import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import os

from causalscript.core.holographic.data_types import HolographicTensor

class OpticalInterferenceEngine(nn.Module):
    """
    Multimodal Optical Interference Engine.
    
    Core Principle:
    Simulates optical interference to calculate resonance between input "Probe Waves" (HolographicTensor)
    and stored "Memory Waves" (Weights).
    
    Operation:
    Resonance = | Probe x Memory^H |^2
    Where x is complex matrix multiplication and H is Hermitian transpose (conjugate transpose).
    """

    def __init__(self, memory_capacity: int = 100, input_dim: int = 1024, weights_path: Optional[str] = None):
        super().__init__()
        self.input_dim = input_dim
        self.memory_capacity = memory_capacity
        
        # Optical Memory Matrix (Knowledge Base)
        # Rows: Memory Content (Concept/Rule), Cols: Frequency Bins
        # Initialized with complex random numbers
        self.optical_memory = nn.Parameter(torch.randn(memory_capacity, input_dim, dtype=torch.cfloat))
        # Hook to fix "view_as_real doesn't work on unresolved conjugated tensors" error in Adam
        self.optical_memory.register_hook(lambda grad: grad.resolve_conj())
        
        if weights_path:
            self._load_memory(weights_path)

    def _load_memory(self, path: str):
        if os.path.exists(path):
            try:
                if path.endswith('.npy'):
                    np_weights = np.load(path)
                    with torch.no_grad():
                        self.optical_memory.copy_(torch.from_numpy(np_weights))
                else:
                    state_dict = torch.load(path)
                    self.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load optical memory from {path}. Using random initialization. Error: {e}")

    def forward(self, input_batch: Union[HolographicTensor, torch.Tensor]) -> torch.Tensor:
        """
        Massive Parallel Interference Calculation.
        
        Args:
            input_batch: [Batch, InputDim] Complex Tensor (Probe Wave)
            
        Returns:
            Resonance Energy: [Batch, MemoryCapacity] Real Tensor (Intensity)
        """
        # 1. Ensure Input is Complex Holographic Tensor
        if not input_batch.is_complex():
             input_batch = input_batch.type(torch.cfloat)
             
        # Add batch dim if single input
        if input_batch.dim() == 1:
            input_batch = input_batch.unsqueeze(0)
            
        # 2. Interference Calculation (Wave Superposition)
        # We compute the dot product between the input wave and the conjugate of the memory wave.
        # This is equivalent to physical optical correlation.
        # [B, D] x [M, D]^T -> [B, M] (using Hermitian transpose implicitly via conj in some frameworks, 
        # but torch.matmul with complex handles algebraic mult. We want correlation: sum(u * conj(v)))
        
        # To simulate "Interference" as similarity/resonance:
        # <u, v> = sum(u_i * conj(v_i))
        # We use .conj().t() for the memory
        
        interference_pattern = torch.matmul(input_batch, self.optical_memory.conj().resolve_conj().t())
        
        # 3. Intensity Detection (Square Law Detector)
        # Phase information is lost here, converted to energy (probability/relevance)
        resonance_energy = interference_pattern.abs() ** 2
        
        return resonance_energy

    def get_ambiguity(self, resonance_energy: torch.Tensor) -> float:
        """
        Calculates ambiguity (entropy) of the resonance distribution.
        High ambiguity = System is unsure (many memories resonating equally).
        """
        with torch.no_grad():
            # For simplicity, calculate on the mean of the batch or the first item
            energy_profile = resonance_energy.mean(dim=0)
            
            total_energy = energy_profile.sum()
            if total_energy == 0:
                return 1.0
                
            probs = energy_profile / total_energy
            epsilon = 1e-10
            entropy = -torch.sum(probs * torch.log(probs + epsilon))
            
            max_entropy = np.log(self.memory_capacity)
            if max_entropy == 0:
                return 0.0
                
            return float(entropy.item() / max_entropy)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

