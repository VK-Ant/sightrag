"""TensorRT backend — fastest on NVIDIA GPU."""

import os
from pathlib import Path
from .pytorch_backend import PyTorchBackend

MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class TensorRTBackend(PyTorchBackend):
    
    name = "tensorrt"
    
    def _load_yolo(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires NVIDIA GPU")
        
        from ultralytics import YOLO
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        
        engine_path = os.path.join(MODEL_DIR, "yolo11n.engine")
        if not os.path.exists(engine_path):
            pt_path = os.path.join(MODEL_DIR, "yolo11n.pt")
            m = YOLO(pt_path) if os.path.exists(pt_path) else YOLO("yolo11n.pt")
            m.export(format="engine")
            cwd_engine = "yolo11n.engine"
            if os.path.exists(cwd_engine):
                import shutil
                shutil.move(cwd_engine, engine_path)
        
        self._yolo = YOLO(engine_path)
