"""Backward compatibility — v0.1 embedder.py now uses backends."""
from .backends.pytorch_backend import PyTorchBackend as _Backend
_b = None
class Embedder:
    def __init__(self, *args, **kwargs):
        global _b
        if _b is None:
            _b = _Backend()
        self._backend = _b
        self.embed_dim = self._backend.embed_dim
    def embed_image(self, image):
        return self._backend.embed_image(image)
    def embed_text(self, text, domain_hint=None):
        return self._backend.embed_text(text, domain_hint)
