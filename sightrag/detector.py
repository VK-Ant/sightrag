# sightrag/detector.py
# YOLO detection — models stored in ~/.sightrag/models/

import os
import numpy as np
from PIL import Image
from pathlib import Path


MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class Detector:
    """YOLO object detector with whole-image fallback."""

    def __init__(self, model_size: str = "yolo11n.pt", model_dir: str = None):
        self.model = None
        self.model_size = model_size
        self.model_dir = model_dir or MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            import logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)

            model_path = os.path.join(self.model_dir, self.model_size)

            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Download and move to our folder
                self.model = YOLO(self.model_size)
                # Move .pt file from current dir to model_dir
                cwd_model = os.path.join(os.getcwd(), self.model_size)
                if os.path.exists(cwd_model) and cwd_model != model_path:
                    import shutil
                    shutil.move(cwd_model, model_path)

        except Exception as e:
            print(f"[SightRAG] YOLO not available: {str(e)[:100]}")
            self.model = None

    def detect(self, image: Image.Image, confidence: float = 0.25):
        """Detect objects. Always returns at least whole image."""
        regions = []

        if self.model is not None:
            try:
                results = self.model(image, conf=confidence, verbose=False)
                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        w, h = image.size
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if (x2 - x1) < 10 or (y2 - y1) < 10:
                            continue
                        regions.append({
                            "crop":       image.crop((x1, y1, x2, y2)),
                            "bbox":       [x1, y1, x2, y2],
                            "label":      result.names[int(box.cls[0])],
                            "confidence": float(box.conf[0])
                        })
            except:
                pass

        # Always add whole image
        w, h = image.size
        regions.append({
            "crop":       image,
            "bbox":       [0, 0, w, h],
            "label":      "whole_image",
            "confidence": 1.0
        })

        return regions
