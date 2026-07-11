from .base import BackendBase
from .pytorch_backend import PyTorchBackend

__all__ = ["BackendBase", "PyTorchBackend", "auto_select_backend"]


def auto_select_backend():
    """Auto-select fastest available backend. Silent."""
    
    # Try TensorRT (fastest, GPU only)
    try:
        import torch
        if torch.cuda.is_available():
            try:
                from .tensorrt_backend import TensorRTBackend
                return TensorRTBackend()
            except Exception:
                pass
    except ImportError:
        pass
    
    # Try ONNX (fast, cross-platform)
    try:
        import onnxruntime
        from .onnx_backend import ONNXBackend
        return ONNXBackend()
    except (ImportError, Exception):
        pass
    
    # Try OpenVINO (Intel optimized)
    try:
        from .openvino_backend import OpenVINOBackend
        return OpenVINOBackend()
    except (ImportError, Exception):
        pass
    
    # PyTorch always works
    return PyTorchBackend()
