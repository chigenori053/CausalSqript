import logging
import torch
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from causalscript.core.holographic.data_types import HolographicTensor, SpectrumConfig
from causalscript.core.multimodal.base_encoder import IHolographicEncoder

class HolographicTextEncoder(IHolographicEncoder):
    """
    Holographic Text Encoder (Cepstral Analysis).
    
    Logic: Text -> Token Embeddings -> FFT -> Log-Magnitude -> IFFT (Cepstrum)
    *Note: For the V1 implementation, we simplified this to FFT of the sentence embedding 
    to create a complex spectrum directly compatible with the optical engine.*
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = model_name
        self._target_dim = SpectrumConfig.DIMENSION
        
        if SentenceTransformer is None:
            self.logger.warning("sentence-transformers not installed. Semantic encoding will be disabled.")
        
    def _load_model(self):
        """Lazy load the model."""
        if self.model is None and SentenceTransformer:
            try:
                self.logger.info(f"Loading semantic model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                self.logger.error(f"Failed to load semantic model: {e}")
                self.model = None

    def encode(self, text: str) -> HolographicTensor:
        """
        Encodes text into a HolographicTensor (Complex Spectrum).
        """
        self._load_model()
        if self.model is None:
            # Return empty complex tensor if model fails
            return HolographicTensor(torch.zeros(self._target_dim, dtype=torch.complex64))
        
        try:
            # 1. Vectorize (Real-valued embedding)
            # Output shape: [EmbeddingDim] e.g. 384
            embedding_np = self.model.encode(text)
            embedding = torch.tensor(embedding_np, dtype=torch.float32)
            
            # Pad to target dimension if necessary
            curr_dim = embedding.shape[0]
            if curr_dim < self._target_dim:
                padding = torch.zeros(self._target_dim - curr_dim)
                embedding = torch.cat([embedding, padding])
            elif curr_dim > self._target_dim:
                embedding = embedding[:self._target_dim]
                
            # 2. Spectral Transform (FFT)
            # We treat the embedding as a signal in time/space domain and convert to frequency domain.
            spectrum = torch.fft.fft(embedding)
            
            # 3. Cast to standard complex64
            return HolographicTensor(spectrum.type(torch.complex64))
            
        except Exception as e:
            self.logger.error(f"Holographic encoding failed: {e}")
            return HolographicTensor(torch.zeros(self._target_dim, dtype=torch.complex64))

    def decode(self, spectrum: HolographicTensor) -> str:
        """
        Reconstruction from spectrum is not supported in this version.
        """
        raise NotImplementedError("Text reconstruction from holographic spectrum not available.")


class TransformerEncoder:
    """
    Legacy Semantic Encoder using SentenceTransformers.
    Retained for backward compatibility.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = model_name
        
        if SentenceTransformer is None:
            self.logger.warning("sentence-transformers not installed. Semantic encoding will be disabled.")
        
    def _load_model(self):
        if self.model is None and SentenceTransformer:
            try:
                self.logger.info(f"Loading semantic model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                self.logger.error(f"Failed to load semantic model: {e}")
                self.model = None

    def encode(self, text: str) -> list[float]:
        self._load_model()
        if self.model is None:
            return []
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            return []
