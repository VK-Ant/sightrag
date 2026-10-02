"""
Base interface for object trackers.

A tracker assigns persistent IDs to detections across video frames.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TrackState:
    """State of a single tracked object."""
    track_id: int
    bbox: List[int]          # [x1, y1, x2, y2]
    label: str
    confidence: float
    frame_idx: int
    timestamp: str = ""
    embedding: Optional[object] = None
    mask: Optional[object] = None


@dataclass
class Track:
    """Complete track of one object across frames."""
    track_id: int
    label: str
    first_frame: int = 0
    last_frame: int = 0
    first_timestamp: str = ""
    last_timestamp: str = ""
    frame_count: int = 0
    states: List[TrackState] = field(default_factory=list)
    representative_crop: Optional[object] = None
    representative_embedding: Optional[object] = None

    def add_state(self, state: TrackState):
        self.states.append(state)
        self.frame_count = len(self.states)
        if not self.states or state.frame_idx < self.first_frame:
            self.first_frame = state.frame_idx
            self.first_timestamp = state.timestamp
        if state.frame_idx > self.last_frame:
            self.last_frame = state.frame_idx
            self.last_timestamp = state.timestamp

    @property
    def duration_frames(self):
        return self.last_frame - self.first_frame + 1

    @property
    def bbox_trajectory(self):
        """Return list of (frame_idx, bbox) tuples."""
        return [(s.frame_idx, s.bbox) for s in self.states]

    @property
    def center_trajectory(self):
        """Return list of (frame_idx, (cx, cy)) tuples."""
        return [
            (s.frame_idx, (
                (s.bbox[0] + s.bbox[2]) // 2,
                (s.bbox[1] + s.bbox[3]) // 2
            ))
            for s in self.states
        ]

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "label": self.label,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "frame_count": self.frame_count,
            "bbox_trajectory": self.bbox_trajectory,
        }


class TrackerBase(ABC):
    """Base class for all trackers in SightRAG."""

    name = "base"

    @abstractmethod
    def update(self, detections: List[Dict], frame_idx: int,
               timestamp: str = "") -> List[TrackState]:
        """
        Update tracker with detections from current frame.

        Args:
            detections: list of detection dicts with bbox, label, confidence
            frame_idx: current frame index
            timestamp: current timestamp string

        Returns:
            List of TrackState with assigned track_ids
        """
        pass

    @abstractmethod
    def get_tracks(self) -> Dict[int, Track]:
        """Return all tracks (active and finished)."""
        pass

    def reset(self):
        """Reset tracker state for new video."""
        pass

    def get_active_tracks(self) -> Dict[int, Track]:
        """Return only currently active tracks."""
        all_tracks = self.get_tracks()
        return {tid: t for tid, t in all_tracks.items()
                if t.frame_count > 0}
