"""
Base interface for segmentation models.

Any segmentor must implement segment() which returns detections with masks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class SegmentorBase(ABC):
    """Base class for all segmentation models in SightRAG."""

    name = "base"

    @abstractmethod
    def segment(self, image, confidence: float = 0.25) -> List[Dict]:
        """
        Segment objects in an image.

        Args:
            image: PIL Image or numpy array
            confidence: minimum confidence threshold

        Returns:
            List of dicts, each with:
                - bbox: [x1, y1, x2, y2]
                - label: str
                - confidence: float
                - crop: PIL Image (cropped region)
                - mask: numpy array (H, W) binary mask for this instance
        """
        pass

    def segment_region(self, image, bbox) -> np.ndarray:
        """
        Segment a specific region given a bounding box.
        Returns binary mask (H, W) for the full image.
        Optional: subclasses can override for efficiency.
        """
        results = self.segment(image)
        if not results:
            return None

        # Find the detection closest to the given bbox
        best_match = None
        best_iou = 0
        for r in results:
            iou = self._bbox_iou(bbox, r["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = r

        if best_match and best_iou > 0.3:
            return best_match.get("mask")
        return None

    @staticmethod
    def _bbox_iou(box1, box2):
        """Calculate IoU between two bboxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0
