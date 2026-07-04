# sightrag/embedder.py
# CLIP embedder — works across all transformers versions

import os
import numpy as np
from PIL import Image

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Embedder:
    """CLIP embedder for images and text queries."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        self.embed_dim = None
        self._load(model_name)

    def _load(self, model_name):
        import warnings
        warnings.filterwarnings("ignore")

        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embed_dim = self.model.config.projection_dim

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed image → fixed-size normalized vector."""
        import torch
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get pixel values
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"]

            with torch.no_grad():
                # Use vision model + projection explicitly
                # This guarantees correct output dim across all versions
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                pooled = vision_out.pooler_output            # (1, hidden_dim)
                projected = self.model.visual_projection(pooled)  # (1, projection_dim)

            emb = projected[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        except Exception as e:
            print(f"[SightRAG] Image embed error: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)

    def embed_text(self, text: str, domain_hint: str = None) -> np.ndarray:
        """Embed text query → fixed-size normalized vector."""
        import torch
        try:
            query = f"{text} {domain_hint}" if domain_hint else text

            inputs = self.processor(
                text=[query], return_tensors="pt",
                padding=True, truncation=True
            )

            with torch.no_grad():
                # Use text model + projection explicitly
                text_out = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                pooled = text_out.pooler_output              # (1, hidden_dim)
                projected = self.model.text_projection(pooled)    # (1, projection_dim)

            emb = projected[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        except Exception as e:
            print(f"[SightRAG] Text embed error: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)
