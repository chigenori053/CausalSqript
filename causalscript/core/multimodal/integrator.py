
import logging
from typing import Tuple, Optional, Any
from .text_encoder import TransformerEncoder
from .vision_encoder import VisionEncoder

class MultimodalIntegrator:
    """
    Coordinator for Multimodal Inputs (Text, Vision, Symbolic).
    Standardizes inputs into a structure consumable by the Reasoning Agent.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_encoder = TransformerEncoder()
        self.vision_encoder = VisionEncoder()
        
    def process_input(self, input_data: Any, input_type: str = "text") -> Tuple[Optional[str], list[float]]:
        """
        Process raw input into (Symbolic Expression, Semantic Vector).
        
        Args:
            input_data: The raw input (string or path to image).
            input_type: 'text' or 'image'.
            
        Returns:
            Tuple(expression_str, semantic_vector)
        """
        expression = None
        semantic_vector = []
        
        try:
            # 1. Perception Phase (Raw -> Text/Expr)
            if input_type == "image":
                self.logger.info("Processing visual input...")
                expression = self.vision_encoder.image_to_latex(input_data)
                if not expression:
                    self.logger.warning("Vision encoder failed to extract expression.")
                    return None, []
            elif input_type == "text":
                expression = input_data
            else:
                self.logger.error(f"Unknown input type: {input_type}")
                return None, []

            # 2. Semantic Encoding Phase (Expr -> Vector)
            if expression:
                # We encode the textual representation (LaTeX/MathLang)
                semantic_vector = self.text_encoder.encode(expression)
                
            return expression, semantic_vector
            
        except Exception as e:
            self.logger.error(f"Multimodal integration error: {e}")
            return None, []
