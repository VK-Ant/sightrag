"""Detector base interface. User implements this for custom models."""

from PIL import Image
from typing import List, Dict


class DetectorBase:
    """
    Implement this to plug your own detection model into SightRAG.
    
    Example:
        class MyDetector(DetectorBase):
            def __init__(self):
                self.model = load_my_model()
            
            def detect(self, image, confidence=0.25):
                preds = self.model.predict(image)
                return [
                    {"bbox": [p.x1, p.y1, p.x2, p.y2],
                     "label": p.label,
                     "confidence": p.score,
                     "crop": image.crop((p.x1, p.y1, p.x2, p.y2))}
                    for p in preds if p.score >= confidence
                ]
        
        rag = SightRAG(detector=MyDetector())
    """
    
    def detect(self, image: Image.Image, confidence: float = 0.25) -> List[Dict]:
        raise NotImplementedError("Implement detect() in your detector")
