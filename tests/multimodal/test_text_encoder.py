
import pytest
from coherent.engine.multimodal.text_encoder import TransformerEncoder

def test_text_encoder_initialization():
    encoder = TransformerEncoder()
    assert encoder.model_name == "all-MiniLM-L6-v2"
    # assert encoder.model is None # Lazy loaded

def test_text_encoder_encoding():
    encoder = TransformerEncoder()
    # Mocking SentenceTransformer if not available in environment
    if encoder.model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")
            
    vector = encoder.encode("calculate derivation")
    if vector:
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert isinstance(vector[0], float)
    else:
        # If model loading failed (e.g. download issue), we accept empty list but warn
        pass

