"""PyTorch backend — default. Works everywhere."""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from .base import BackendBase

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class PyTorchBackend(BackendBase):
    
    name = "pytorch"
    
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        import warnings
        warnings.filterwarnings("ignore")
        self._load_yolo()
        self._load_clip()
    
    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            import logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)
            
            model_path = os.path.join(MODEL_DIR, "yolo11n.pt")
            if os.path.exists(model_path):
                self._yolo = YOLO(model_path)
            else:
                self._yolo = YOLO("yolo11n.pt")
                cwd_model = os.path.join(os.getcwd(), "yolo11n.pt")
                if os.path.exists(cwd_model) and cwd_model != model_path:
                    import shutil
                    shutil.move(cwd_model, model_path)
        except Exception:
            self._yolo = None
    
    def _load_clip(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        
        self._clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip.eval()
        self.embed_dim = self._clip.config.projection_dim
    
    def detect(self, image, confidence=0.25):
        regions = []
        w, h = image.size
        img_area = w * h
        
        # Minimum detection size: 3% of image area
        # Filters out tiny objects like ties, watches, buttons
        min_area = img_area * 0.03
        
        if self._yolo is not None:
            try:
                results = self._yolo(image, conf=confidence, verbose=False)
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_area = box_w * box_h
                        
                        # Skip tiny detections
                        if box_w < 20 or box_h < 20:
                            continue
                        if box_area < min_area:
                            continue
                        
                        regions.append({
                            "crop": image.crop((x1, y1, x2, y2)),
                            "bbox": [x1, y1, x2, y2],
                            "label": r.names[int(box.cls[0])],
                            "confidence": float(box.conf[0])
                        })
            except Exception:
                pass
        
        # Sort by area — larger detections first (person > tie)
        regions.sort(key=lambda r: (r["bbox"][2]-r["bbox"][0]) * (r["bbox"][3]-r["bbox"][1]), reverse=True)
        
        # Always add whole image
        regions.append({
            "crop": image,
            "bbox": [0, 0, w, h],
            "label": "whole_image",
            "confidence": 0.1
        })
        return regions
    
    def embed_image(self, image):
        import torch
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            inp = self._clip_proc(images=image, return_tensors="pt")
            with torch.no_grad():
                vis = self._clip.vision_model(pixel_values=inp["pixel_values"])
                proj = self._clip.visual_projection(vis.pooler_output)
            e = proj[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(e)
            return e / norm if norm > 0 else e
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)
    
    def embed_text(self, text, domain_hint=None):
        import torch
        try:
            query = f"{text} {domain_hint}" if domain_hint else text
            inp = self._clip_proc(text=[query], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                tout = self._clip.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
                proj = self._clip.text_projection(tout.pooler_output)
            e = proj[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(e)
            return e / norm if norm > 0 else e
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)
