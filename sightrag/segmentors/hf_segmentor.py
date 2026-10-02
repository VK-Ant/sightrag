"""
HuggingFace segmentor - use any segmentation model from HuggingFace Hub.

Supports models compatible with transformers image-segmentation pipeline:
    - Mask2Former
    - OneFormer
    - SegFormer
    - Any model with AutoModelForImageSegmentation

Usage:
    segmentor = HFSegmentor("facebook/mask2former-swin-base-coco-instance")
    segmentor = HFSegmentor("shi-labs/oneformer_coco_swin_large")
"""

import numpy as np
from PIL import Image
from .base import SegmentorBase


class HFSegmentor(SegmentorBase):
    """HuggingFace Hub segmentation model wrapper."""

    name = "huggingface"

    def __init__(self, model_name: str, device: str = None, task: str = None):
        """
        Args:
            model_name: HuggingFace model ID
            device: "cuda" or "cpu" (auto-detected if None)
            task: pipeline task type. Auto-detected if None.
                  Options: "image-segmentation", "panoptic-segmentation"
        """
        self.model_name = model_name
        self._device = device
        self._task = task
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load HuggingFace pipeline."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Install: pip install transformers torch"
            )

        if self._device is None:
            import torch
            self._device = 0 if torch.cuda.is_available() else -1

        task = self._task or "image-segmentation"

        print(f"[SightRAG] Loading HF segmentor: {self.model_name}")
        self._pipeline = pipeline(
            task=task,
            model=self.model_name,
            device=self._device
        )

    def segment(self, image, confidence: float = 0.25):
        self._load_pipeline()

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        img_w, img_h = pil_image.size

        try:
            outputs = self._pipeline(pil_image)
        except Exception as e:
            print(f"[SightRAG] HF segmentor error: {e}")
            return []

        detections = []

        for item in outputs:
            score = item.get("score", 1.0)
            if score < confidence:
                continue

            label = item.get("label", "object")

            # Get mask
            mask_img = item.get("mask")
            if mask_img is None:
                continue

            if isinstance(mask_img, Image.Image):
                mask = np.array(mask_img.convert("L")) > 127
            elif isinstance(mask_img, np.ndarray):
                mask = mask_img.astype(bool)
            else:
                continue

            # Resize mask if needed
            if mask.shape != (img_h, img_w):
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
                mask = np.array(mask_pil) > 127

            # Get bbox from mask
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            # Min area filter
            area = (x2 - x1) * (y2 - y1)
            if area < img_w * img_h * 0.01:
                continue

            crop = pil_image.crop((x1, y1, x2, y2))

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": round(score, 4),
                "crop": crop,
                "mask": mask,
            })

        # Sort by area descending
        detections.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) *
                          (d["bbox"][3] - d["bbox"][1]),
            reverse=True
        )

        return detections
