# sightrag/detector.py
# YOLO-based object detector
# Falls back to whole-image if no detections

import numpy as np
from PIL import Image


class Detector:
    """
    Tier 1 detection — YOLO on standard domains.
    Falls back gracefully if no detections found.
    """

    def __init__(self, model_size: str = "yolo11n.pt"):
        self.model = None
        self.model_size = model_size
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_size)
        except Exception as e:
            print(f"[SightRAG] YOLO load failed: {e}")
            print("[SightRAG] Falling back to whole-image mode.")
            self.model = None

    def detect(self, image: Image.Image, confidence: float = 0.25):
        """
        Returns list of detected regions.
        Each region: {"crop": PIL.Image, "bbox": [x1,y1,x2,y2],
                      "label": str, "confidence": float}
        Falls back to whole image if no detections.
        """
        if self.model is None:
            return self._whole_image_fallback(image)

        try:
            results = self.model(image, conf=confidence, verbose=False)
            regions = []

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]

                    # Crop region
                    crop = image.crop((x1, y1, x2, y2))
                    regions.append({
                        "crop":       crop,
                        "bbox":       [x1, y1, x2, y2],
                        "label":      label,
                        "confidence": conf
                    })

            if not regions:
                return self._whole_image_fallback(image)

            return regions

        except Exception as e:
            print(f"[SightRAG] Detection error: {e}")
            return self._whole_image_fallback(image)

    def _whole_image_fallback(self, image: Image.Image):
        """When YOLO finds nothing — use whole image."""
        w, h = image.size
        return [{
            "crop":       image,
            "bbox":       [0, 0, w, h],
            "label":      "whole_image",
            "confidence": 1.0
        }]
