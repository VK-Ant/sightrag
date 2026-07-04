# sightrag/utils/image.py
# Image loading and preprocessing utilities

from PIL import Image
from pathlib import Path

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: str) -> Image.Image:
    """Load and validate a single image."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {path.suffix}\n"
            f"Supported: {SUPPORTED_FORMATS}"
        )

    img = Image.open(path).convert("RGB")
    return img


def get_image_paths(folder: str):
    """Get all image paths from a folder."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    paths = []
    for fmt in SUPPORTED_FORMATS:
        paths.extend(folder.glob(f"*{fmt}"))
        paths.extend(folder.glob(f"*{fmt.upper()}"))

    if not paths:
        raise ValueError(
            f"No images found in {folder}\n"
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    return sorted(paths)
