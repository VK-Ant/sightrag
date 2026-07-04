# sightrag/embedder.py
# CLIP-based image and text embedder

import numpy as np
from PIL import Image


class Embedder:
    """
    CLIP embedder for images and text queries.
    Supports domain hints for better custom domain accuracy.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        self.model_name = model_name
        self._load()

    def _load(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            print(f"[SightRAG] Loading CLIP model...")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            print("[SightRAG] CLIP ready.")
        except Exception as e:
            raise RuntimeError(f"[SightRAG] CLIP load failed: {e}")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single image crop into a vector."""
        import torch
        try:
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            embedding = features[0].numpy()
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as e:
            print(f"[SightRAG] Image embedding error: {e}")
            return np.zeros(512, dtype=np.float32)

    def embed_text(self, text: str, domain_hint: str = None) -> np.ndarray:
        """
        Embed a text query into a vector.
        domain_hint improves accuracy for custom domains.
        """
        import torch
        try:
            # Enrich query with domain hint
            if domain_hint:
                enriched = f"{text} {domain_hint}"
            else:
                enriched = text

            inputs = self.processor(
                text=[enriched],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            embedding = features[0].numpy()
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as e:
            print(f"[SightRAG] Text embedding error: {e}")
            return np.zeros(512, dtype=np.float32)
