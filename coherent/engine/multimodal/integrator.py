
import logging
from typing import Tuple, Optional, Any
import torch

from .text_encoder import TransformerEncoder, HolographicTextEncoder
from .vision_encoder import VisionEncoder, HolographicVisionEncoder
from .binding import HolographicBinding
from coherent.engine.holographic.data_types import HolographicTensor
from coherent.engine.input_parser import CoherentInputParser

class MultimodalIntegrator:
    """
    Coordinator for Multimodal Inputs (Text, Vision, Symbolic).
    Standardizes inputs into a structure consumable by the Reasoning Agent.
    Migrated to Output Holographic Tensors.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use Holographic Encoders
        self.text_encoder = HolographicTextEncoder()
        self.vision_encoder = HolographicVisionEncoder()
        
    def process_input(self, input_data: Any, input_type: str = "text") -> Tuple[Optional[str], HolographicTensor]:
        """
        Process raw input into (Symbolic Expression, Complex Holographic Vector).
        
        Args:
            input_data: The raw input (string or path to image).
            input_type: 'text', 'image', or 'multimodal'.
            
        Returns:
            Tuple(expression_str, holographic_tensor)
        """
        expression = None
        holographic_vector = None
        
        try:
            # 1. Perception & Encoding
            if input_type == "image":
                self.logger.info("Processing visual input...")
                # Vision Encoder (Spectral)
                holographic_vector = self.vision_encoder.encode(input_data)
                
                # Try to extract symbolic expression via legacy path if needed
                # (For now we rely on the complex vector for reasoning, 
                # but might need text for display)
                # expression = self.vision_encoder.decode(holographic_vector) # Not implemented fully
                # Use legacy logic just for expression string?
                # For now let's say expression is None unless we have OCR
                
            elif input_type == "text":
                expression = CoherentInputParser.normalize(input_data)
                holographic_vector = self.text_encoder.encode(expression)
                
            elif input_type == "multimodal":
                # Expect input_data to be dict {"image": ..., "text": ...}
                if isinstance(input_data, dict):
                    h_img = self.vision_encoder.encode(input_data.get("image"))
                    h_txt = self.text_encoder.encode(input_data.get("text") or "")
                    
                    # Core Logic: BIND image and text
                    # V_bound = V_img * V_txt
                    holographic_vector = HolographicBinding.bind(h_img, h_txt)
                    expression = input_data.get("text")
                else:
                     self.logger.error("Multimodal input requires dict")
                     return None, HolographicTensor(torch.tensor([]))

            else:
                self.logger.error(f"Unknown input type: {input_type}")
                return None, HolographicTensor(torch.tensor([]))

            if holographic_vector is None:
                 holographic_vector = HolographicTensor(torch.tensor([]))

            return expression, holographic_vector
            
        except Exception as e:
            self.logger.error(f"Multimodal integration error: {e}")
            return None, HolographicTensor(torch.tensor([]))
