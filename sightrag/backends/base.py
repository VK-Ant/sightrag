"""Backend base interface. Every backend implements this."""

import numpy as np
from PIL import Image
from typing import List, Dict, Optional


class BackendBase:
    """Abstract base for all inference backends."""
    
    name = "base"
    embed_dim = 512
    
    def detect(self, image: Image.Image, confidence: float = 0.25) -> List[Dict]:
        """
        Detect objects in image.
        Returns: [{"bbox": [x1,y1,x2,y2], "label": str, 
                   "confidence": float, "crop": PIL.Image}]
        Always includes whole_image as last entry.
        """
        raise NotImplementedError
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed image → normalized vector of shape (embed_dim,)"""
        raise NotImplementedError
    
    def embed_text(self, text: str, domain_hint: str = None) -> np.ndarray:
        """Embed text → normalized vector of shape (embed_dim,)"""
        raise NotImplementedError
    
    def warmup(self):
        """Run warmup inference."""
        dummy = Image.new("RGB", (224, 224))
        self.detect(dummy)
        self.embed_image(dummy)
