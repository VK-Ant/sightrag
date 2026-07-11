"""Backward compatibility — v0.1 detector.py now uses backends."""
# v0.2: detection moved to backends/
# This file kept so old imports don't break
from .backends.pytorch_backend import PyTorchBackend as _Backend
_b = None
class Detector:
    def __init__(self, *args, **kwargs):
        global _b
        if _b is None:
            _b = _Backend()
        self._backend = _b
    def detect(self, image, confidence=0.25):
        return self._backend.detect(image, confidence)
