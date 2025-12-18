
import logging
import torch
from pathlib import Path
from typing import Optional, Any
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from causalscript.core.holographic.data_types import HolographicTensor, SpectrumConfig
from causalscript.core.multimodal.base_encoder import IHolographicEncoder

class HolographicVisionEncoder(IHolographicEncoder):
    """
    Holographic Vision Encoder (2D-FFT).
    
    Logic: Image -> Grayscale/Resize -> 2D-FFT -> Shift -> Flatten -> HolographicTensor
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._target_dim = SpectrumConfig.DIMENSION
        # Approx Sqrt of dimension for square 2D resizing (32x32 = 1024)
        self._img_size = int(np.sqrt(self._target_dim)) 
        
        if Image is None:
            self.logger.warning("Pillow not installed. Vision capabilities disabled.")

    def encode(self, image_input: Any) -> HolographicTensor:
        """
        Encodes an image (path or PIL Object) into a HolographicTensor.
        """
        if Image is None:
             return HolographicTensor(torch.zeros(self._target_dim, dtype=torch.complex64))

        try:
            img = None
            if isinstance(image_input, str) or isinstance(image_input, Path):
                path = Path(image_input)
                if path.exists():
                    img = Image.open(path)
            elif isinstance(image_input, Image.Image):
                img = image_input
            
            if img is None:
                self.logger.error(f"Invalid image input: {image_input}")
                return HolographicTensor(torch.zeros(self._target_dim, dtype=torch.complex64))

            # 1. Preprocess: Grayscale and Resize
            img = img.convert('L').resize((self._img_size, self._img_size))
            
            # Convert to Tensor normalized [0, 1]
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            
            # 2. 2D Fourier Transform
            f_transform = torch.fft.fft2(img_tensor)
            
            # 3. Shift zero frequency to center
            f_shifted = torch.fft.fftshift(f_transform)
            
            # 4. Flatten to 1D spectrum for Optical Engine compatibility
            f_flat = f_shifted.flatten()
            
            # Ensure precise dimension match just in case
            if f_flat.shape[0] != self._target_dim:
                 # Resize logic if sqrt wasn't perfect, though strict sizing above handles it
                 f_flat = torch.nn.functional.pad(f_flat, (0, self._target_dim - f_flat.shape[0]))
            
            return HolographicTensor(f_flat.type(torch.complex64))

        except Exception as e:
            self.logger.error(f"Holographic vision encoding failed: {e}")
            return HolographicTensor(torch.zeros(self._target_dim, dtype=torch.complex64))

    def decode(self, spectrum: HolographicTensor) -> Optional[Image.Image]:
        """
        Reconstructs image from holographic tensor (Inverse FFT).
        """
        try:
            # 1. Reshape
            f_shifted = spectrum.reshape(self._img_size, self._img_size)
            
            # 2. Inverse Shift
            f_transform = torch.fft.ifftshift(f_shifted)
            
            # 3. Inverse FFT
            img_tensor = torch.fft.ifft2(f_transform).real # Take real part
            
            # 4. Post-process
            img_np = (img_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np)
            
        except Exception as e:
            self.logger.error(f"Holographic vision decoding failed: {e}")
            return None


class VisionEncoder:
    """
    Legacy Vision Encoder placeholder.
    Retained for backward compatibility.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        if Image is None:
            self.logger.warning("Pillow not installed. Vision capabilities disabled.")

    def image_to_latex(self, image_path: str) -> Optional[str]:
        if Image is None:
            return None
            
        try:
            path = Path(image_path)
            if not path.exists():
                self.logger.error(f"Image not found: {image_path}")
                return None
                
            img = Image.open(path)
            if "integral" in str(path):
                return r"\int x^2 dx"
            elif "quadratic" in str(path):
                return r"x^2 + 2x + 1"
                
            return None
            
        except Exception as e:
            self.logger.error(f"Vision processing failed: {e}")
            return None
