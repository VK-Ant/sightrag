# SightRAG — Install Guide

## Requirements

- Python 3.9 - 3.12
- Windows / Linux / Mac
- GPU optional (NVIDIA CUDA for speed)

## Install

```bash
pip install sightrag
```

## GPU Support (NVIDIA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install sightrag
```

## Optional Features

```bash
pip install sightrag[onnx]             # faster inference
pip install sightrag[ocr]              # read text on images
pip install sightrag[grounding-dino]   # any domain detection
pip install sightrag[cli]              # terminal commands
pip install sightrag[qdrant]           # large scale store
pip install sightrag[api]              # REST API
pip install sightrag[all]              # everything
```

## Test

```bash
python test_sightrag.py
```

## Troubleshooting

**Python 3.13+ not working?**
Use Python 3.12. PyTorch doesn't support 3.13+ yet.

**CUDA not detected?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: reinstall torch with correct CUDA version
```

**torchaudio error on Windows?**
```bash
pip install torchaudio --force-reinstall
```
