"""
SightRAG Trackers - pluggable multi-object tracking.

Default: ByteTrack (pure Python, fast, no extra weights)
Optional: BoT-SORT (Re-ID aware), custom TrackerBase
"""

from .base import TrackerBase

__all__ = ["TrackerBase"]
