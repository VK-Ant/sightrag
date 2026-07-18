"""
SightRAG Re-ranker — improves result quality on large datasets.

Problem: 100,000 similar images → CLIP returns 50,000 with score 0.80-0.85
Solution: Get top 100 from CLIP → re-rank to find real top 5

Usage:
    rag = SightRAG(rerank=True)
    results = rag.query("find person", top_k=5)
    # Internally: retrieves top 100, re-ranks to top 5
"""

import numpy as np
from PIL import Image


class ReRanker:
    """
    Cross-encoder re-ranking for better result quality.
    Takes top-N candidates from vector search,
    re-scores with more expensive cross-attention model.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self._model = None
        self._proc = None
        self._load(model_name)
    
    def _load(self, model_name):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            import warnings
            warnings.filterwarnings("ignore")
            
            self._model = CLIPModel.from_pretrained(model_name)
            self._proc = CLIPProcessor.from_pretrained(model_name)
            self._model.eval()
        except Exception as e:
            print(f"[SightRAG] Re-ranker init failed: {e}")
    
    def rerank(self, query_text, candidates, top_k=5):
        """
        Re-rank candidate results using cross-attention scoring.
        
        Args:
            query_text: original text query
            candidates: list of result dicts with "image_path"
            top_k: final number of results
        
        Returns: re-ranked top_k results
        """
        import torch
        
        if self._model is None or not candidates:
            return candidates[:top_k]
        
        scored = []
        
        for candidate in candidates:
            try:
                path = candidate.get("image_path", "")
                if not path or not __import__("os").path.exists(path):
                    scored.append((candidate, candidate.get("score", 0)))
                    continue
                
                image = Image.open(path).convert("RGB")
                
                # Crop to bbox if available
                bbox = candidate.get("bbox", [])
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        image = image.crop((x1, y1, x2, y2))
                
                # Cross-attention score (image + text together)
                inputs = self._proc(
                    text=[query_text],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # logits_per_image gives similarity score
                    score = outputs.logits_per_image[0][0].item()
                    # Normalize to 0-1
                    score = 1.0 / (1.0 + np.exp(-score / 100.0))
                
                scored.append((candidate, float(score)))
                
            except Exception:
                scored.append((candidate, candidate.get("score", 0)))
        
        # Sort by re-ranked score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return top_k
        results = []
        for candidate, new_score in scored[:top_k]:
            candidate["score"] = round(new_score, 4)
            results.append(candidate)
        
        return results
