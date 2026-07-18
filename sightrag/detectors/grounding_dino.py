"""
Grounding DINO detector — open-vocabulary detection.
Detects ANY object by text description. No training needed.

Usage:
    rag = SightRAG(detector="grounding-dino")
    rag.index("./photos/")
    results = rag.query("find person")
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from .base import DetectorBase

MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class GroundingDINODetector(DetectorBase):
    """Open-vocabulary detector — any object by text."""
    
    def __init__(self, text_prompt=None, box_threshold=0.25, text_threshold=0.25):
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None
        self._load()
    
    def _load(self):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch
            
            model_id = "IDEA-Research/grounding-dino-tiny"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            
        except ImportError:
            raise ImportError(
                "Grounding DINO requires: pip install transformers torch"
            )
    
    def detect(self, image, confidence=0.25, text_prompt=None):
        import torch
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        prompt = text_prompt or self.text_prompt or "person. car. object. furniture. bottle."
        if not prompt.endswith("."):
            prompt = prompt + "."
        
        inputs = self._processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Handle different transformers versions
        w, h = image.size
        try:
            # Newer transformers API
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=confidence,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)]
            )[0]
        except TypeError:
            try:
                # Try with box_threshold
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=confidence,
                    text_threshold=self.text_threshold,
                    target_sizes=[(h, w)]
                )[0]
            except TypeError:
                # Oldest API — manual post-processing
                results = self._manual_post_process(outputs, image, confidence)
        
        regions = []
        
        boxes = results["boxes"].cpu().numpy() if hasattr(results["boxes"], 'cpu') else np.array(results["boxes"])
        scores = results["scores"].cpu().numpy() if hasattr(results["scores"], 'cpu') else np.array(results["scores"])
        labels = results.get("labels", results.get("text", [""]*len(scores)))
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            regions.append({
                "crop": image.crop((x1, y1, x2, y2)),
                "bbox": [x1, y1, x2, y2],
                "label": str(label).strip(),
                "confidence": float(score)
            })
        
        # Whole image fallback
        regions.append({
            "crop": image, "bbox": [0, 0, w, h],
            "label": "whole_image", "confidence": 0.1
        })
        
        return regions
    
    def _manual_post_process(self, outputs, image, threshold):
        """Fallback for older transformers versions."""
        import torch
        
        logits = outputs.logits.sigmoid()
        boxes = outputs.pred_boxes
        
        # Get best predictions
        mask = logits.max(dim=-1).values[0] > threshold
        filtered_boxes = boxes[0][mask]
        filtered_scores = logits.max(dim=-1).values[0][mask]
        
        w, h = image.size
        # Convert from cx, cy, w, h to x1, y1, x2, y2
        result_boxes = []
        for box in filtered_boxes:
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            result_boxes.append([x1, y1, x2, y2])
        
        return {
            "boxes": torch.tensor(result_boxes) if result_boxes else torch.zeros((0, 4)),
            "scores": filtered_scores,
            "labels": ["object"] * len(result_boxes)
        }
