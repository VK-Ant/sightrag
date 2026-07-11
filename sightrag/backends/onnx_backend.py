"""ONNX backend — faster on CPU. Cross-platform."""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from .base import BackendBase

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class ONNXBackend(BackendBase):
    
    name = "onnx"
    
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        import warnings
        warnings.filterwarnings("ignore")
        self._load_yolo()
        self._load_clip()
    
    def _load_yolo(self):
        from ultralytics import YOLO
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        
        onnx_path = os.path.join(MODEL_DIR, "yolo11n.onnx")
        if not os.path.exists(onnx_path):
            pt_path = os.path.join(MODEL_DIR, "yolo11n.pt")
            if os.path.exists(pt_path):
                m = YOLO(pt_path)
            else:
                m = YOLO("yolo11n.pt")
            m.export(format="onnx")
            # Move exported file
            cwd_onnx = "yolo11n.onnx"
            if os.path.exists(cwd_onnx):
                import shutil
                shutil.move(cwd_onnx, onnx_path)
        
        self._yolo = YOLO(onnx_path, task="detect")
    
    def _load_clip(self):
        import onnxruntime as ort
        
        clip_onnx = os.path.join(MODEL_DIR, "clip_vision.onnx")
        
        if not os.path.exists(clip_onnx):
            self._export_clip_onnx(clip_onnx)
        
        self._clip_session = ort.InferenceSession(clip_onnx)
        
        from transformers import CLIPProcessor, CLIPModel
        self._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Text model stays PyTorch (ONNX text export is complex)
        import torch
        self._clip_text = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_text.eval()
        self.embed_dim = self._clip_text.config.projection_dim
    
    def _export_clip_onnx(self, output_path):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip.eval()
        
        class CLIPVisionExport(torch.nn.Module):
            def __init__(self, vision, proj):
                super().__init__()
                self.vision = vision
                self.proj = proj
            def forward(self, pixel_values):
                out = self.vision(pixel_values=pixel_values)
                return self.proj(out.pooler_output)
        
        wrapper = CLIPVisionExport(clip.vision_model, clip.visual_projection)
        wrapper.eval()
        
        dummy = proc(images=Image.new("RGB", (224, 224)), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, output_path,
                input_names=["pixel_values"],
                output_names=["embedding"],
                dynamic_axes={"pixel_values": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True
            )
        del clip
    
    def detect(self, image, confidence=0.25):
        regions = []
        try:
            results = self._yolo(image, conf=confidence, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w, h = image.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    regions.append({
                        "crop": image.crop((x1, y1, x2, y2)),
                        "bbox": [x1, y1, x2, y2],
                        "label": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0])
                    })
        except Exception:
            pass
        
        w, h = image.size
        regions.append({"crop": image, "bbox": [0, 0, w, h], "label": "whole_image", "confidence": 1.0})
        return regions
    
    def embed_image(self, image):
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            inp = self._clip_proc(images=image, return_tensors="np")["pixel_values"].astype(np.float32)
            result = self._clip_session.run(None, {"pixel_values": inp})
            e = result[0][0].astype(np.float32)
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
                tout = self._clip_text.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
                proj = self._clip_text.text_projection(tout.pooler_output)
            e = proj[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(e)
            return e / norm if norm > 0 else e
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)
