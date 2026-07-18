"""SightRAG retriever — text and reference image queries."""

import numpy as np
from PIL import Image


class Retriever:
    
    def __init__(self, embedder, detector, store, domain_hint=None):
        self.embedder = embedder
        self.detector = detector
        self.store = store
        self.domain_hint = domain_hint
    
    def query_text(self, text, top_k=5):
        count = self.store.count()
        if count == 0:
            raise RuntimeError("Nothing indexed yet. Run rag.index() first.")
        
        embedding = self.embedder.embed_text(text, self.domain_hint)
        
        # Fetch extra to compensate for filtering
        fetch_k = min(top_k * 3, count)
        results = self.store.search(embedding, fetch_k)
        return self._filter_and_format(results, top_k)
    
    def query_reference(self, image_path, top_k=5):
        count = self.store.count()
        if count == 0:
            raise RuntimeError("Nothing indexed yet. Run rag.index() first.")
        
        image = Image.open(image_path).convert("RGB")
        regions = self.detector.detect(image)
        
        if not regions:
            embedding = self.embedder.embed_image(image)
        else:
            # Use highest confidence detected region (not whole_image)
            real_regions = [r for r in regions if r["label"] != "whole_image"]
            if real_regions:
                best = max(real_regions, key=lambda r: r["confidence"])
            else:
                best = regions[0]
            embedding = self.embedder.embed_image(best["crop"])
        
        fetch_k = min(top_k * 3, count)
        results = self.store.search(embedding, fetch_k)
        return self._filter_and_format(results, top_k)
    
    def _filter_and_format(self, results, top_k):
        """Filter and prioritize results."""
        
        # Labels to deprioritize (too small/irrelevant for retrieval)
        SKIP_LABELS = {"tie", "watch", "handbag", "suitcase", "frisbee", 
                       "sports ball", "kite", "baseball bat", "baseball glove",
                       "remote", "cell phone", "mouse", "keyboard", 
                       "toothbrush", "hair drier", "scissors"}
        
        # Separate into priority groups
        real_detections = []
        small_objects = []
        whole_images = []
        
        for r in results:
            formatted = {
                "image_path":  r.get("image_path", ""),
                "score":       round(float(r.get("score", 0.0)), 4),
                "label":       r.get("label", ""),
                "confidence":  round(float(r.get("confidence", 0.0)), 4),
                "bbox":        r.get("bbox", []),
                "timestamp":   r.get("timestamp", ""),
                "source_type": r.get("source_type", "image")
            }
            
            label = r.get("label", "")
            if label == "whole_image":
                whole_images.append(formatted)
            elif label in SKIP_LABELS:
                small_objects.append(formatted)
            else:
                real_detections.append(formatted)
        
        # Prioritize real detections over whole_image
        # Deduplicate by image_path — keep highest score per image
        seen_paths = set()
        final = []
        
        for r in real_detections:
            path = r["image_path"]
            if path not in seen_paths:
                seen_paths.add(path)
                final.append(r)
        
        # If not enough, add small objects
        if len(final) < top_k:
            for r in small_objects:
                path = r["image_path"]
                if path not in seen_paths:
                    seen_paths.add(path)
                    final.append(r)
        
        # If still not enough, add whole_image
        if len(final) < top_k:
            for r in whole_images:
                path = r["image_path"]
                if path not in seen_paths:
                    seen_paths.add(path)
                    final.append(r)
        
        return final[:top_k]
