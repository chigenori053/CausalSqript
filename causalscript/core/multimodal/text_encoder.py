import logging
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class TransformerEncoder:
    """
    Encodes text and mathematical expressions into semantic vectors using Transformer models.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = model_name
        
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

    def encode(self, text: str) -> list[float]:
        """
        Encodes the input text into a high-dimensional vector.
        
        Args:
            text: Input string (natural language or LaTeX).
            
        Returns:
            A list of floats representing the semantic vector.
            Returns empty list if model is unavailable.
        """
        self._load_model()
        if self.model is None:
            return []
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            return []
