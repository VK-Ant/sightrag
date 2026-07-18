"""
Person Re-ID embedder — track same person across cameras.

Usage:
    rag = SightRAG(embedder="reid")
    rag.index("./all_cameras/")
    results = rag.query(reference="./suspect.jpg")

Uses torchreid (OSNet) for body re-identification.
Install: pip install sightrag[reid]
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from .base import EmbedderBase

MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class ReIDEmbedder(EmbedderBase):
    """
    Person Re-Identification embedder.
    Produces body-specific embeddings for matching same person.
    Much better than CLIP for person tracking.
    """
    
    embed_dim = 512
    
    def __init__(self, model_name="osnet_x1_0"):
        self.model_name = model_name
        self._model = None
        self._transform = None
        self._clip_model = None
        self._clip_proc = None
        self._load()
    
    def _load(self):
        try:
            import torch
            import torchvision.transforms as T
            from torchreid.utils import FeatureExtractor
            
            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=os.path.join(MODEL_DIR, f"{self.model_name}.pth"),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            self._transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
            
        except ImportError:
            raise ImportError(
                "Person Re-ID requires: pip install torchreid\n"
                "Or: pip install sightrag[reid]"
            )
        
        # Also load CLIP for text queries (Re-ID is image-only)
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            import warnings
            warnings.filterwarnings("ignore")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model.eval()
        except Exception:
            pass
    
    def embed_image(self, image):
        """Embed person image using Re-ID model."""
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            features = self._extractor([image])
            emb = features[0].cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
            
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)
    
    def embed_text(self, text, domain_hint=None):
        """Text embedding via CLIP (Re-ID doesn't support text)."""
        import torch
        if self._clip_model is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        try:
            query = f"{text} {domain_hint}" if domain_hint else text
            inp = self._clip_proc(text=[query], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                tout = self._clip_model.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
                proj = self._clip_model.text_projection(tout.pooler_output)
            e = proj[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(e)
            return e / norm if norm > 0 else e
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)
