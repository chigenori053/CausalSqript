from typing import NewType
import torch

# HolographicTensor: The fundamental unit of the MHCA.
# It represents a complex-valued waveform (spectrum).
# Shape: [Batch, FrequencyBins] or [Batch, Height, Width] (Complex)
HolographicTensor = NewType('HolographicTensor', torch.Tensor)

class SpectrumConfig:
    """Configuration for the frequency domain representations."""
    DIMENSION: int = 1024       # Standard vector/spectrum dimension
    PRECISION: torch.dtype = torch.complex64 # Standard precision for holographic operations
    MAX_BATCH: int = 256        # Maximum batch size for parallel interference

class ModalityType:
    """Constants for supported modalities."""
    TEXT = "text_cepstrum"
    VISION = "vision_hologram"
    AUDIO = "audio_spectrogram"
