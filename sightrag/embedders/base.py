"""Embedder base interface. User implements this for custom models."""

import numpy as np
from PIL import Image


class EmbedderBase:
    """
    Implement this to plug your own embedding model into SightRAG.
    
    Example:
        class MyEmbedder(EmbedderBase):
            embed_dim = 768
            
            def __init__(self):
                self.model = load_my_model()
            
            def embed_image(self, image):
                vec = self.model.encode_image(image)
                return vec / np.linalg.norm(vec)
            
            def embed_text(self, text, domain_hint=None):
                vec = self.model.encode_text(text)
                return vec / np.linalg.norm(vec)
        
        rag = SightRAG(embedder=MyEmbedder())
    """
    
    embed_dim = 512
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError("Implement embed_image()")
    
    def embed_text(self, text: str, domain_hint: str = None) -> np.ndarray:
        raise NotImplementedError("Implement embed_text()")
