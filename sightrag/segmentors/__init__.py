"""
SightRAG Segmentors - pluggable instance segmentation.

Default: YOLO-Seg (fast, included with ultralytics)
Optional: SAM2 (segment anything), HuggingFace models
"""

from .base import SegmentorBase

__all__ = ["SegmentorBase"]
