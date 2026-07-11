"""OpenVINO backend — optimized for Intel CPU."""

import os
from pathlib import Path
from .pytorch_backend import PyTorchBackend

MODEL_DIR = os.path.join(Path.home(), ".sightrag", "models")


class OpenVINOBackend(PyTorchBackend):
    
    name = "openvino"
    
    def _load_yolo(self):
        from ultralytics import YOLO
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        
        ov_path = os.path.join(MODEL_DIR, "yolo11n_openvino_model")
        if not os.path.exists(ov_path):
            pt_path = os.path.join(MODEL_DIR, "yolo11n.pt")
            m = YOLO(pt_path) if os.path.exists(pt_path) else YOLO("yolo11n.pt")
            m.export(format="openvino")
            cwd_ov = "yolo11n_openvino_model"
            if os.path.exists(cwd_ov):
                import shutil
                shutil.move(cwd_ov, ov_path)
        
        self._yolo = YOLO(ov_path, task="detect")
