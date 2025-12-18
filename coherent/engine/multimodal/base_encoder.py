from abc import ABC, abstractmethod
from typing import Any
from coherent.engine.holographic.data_types import HolographicTensor

class IHolographicEncoder(ABC):
    """
    Abstract base class for all Holographic Encoders.
    Ensures that all modalities output a compatible HolographicTensor (complex spectrum).
    """

    @abstractmethod
    def encode(self, raw_input: Any) -> HolographicTensor:
        """
        Converts raw input (text, image, audio) into a HolographicTensor (Complex Spectrum).
        """
        pass

    @abstractmethod
    def decode(self, spectrum: HolographicTensor) -> Any:
        """
        Reconstructs the raw input from its holographic representation (Inverse FFT).
        (Optional: Not all modalities may support perfect reconstruction)
        """
        pass
