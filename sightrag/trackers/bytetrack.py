"""
ByteTrack - simple, fast multi-object tracker.

Pure Python. No extra model weights needed.
Associates detections across frames using IoU matching.
Key idea: uses BOTH high and low confidence detections for robust tracking.

Reference: Zhang et al. "ByteTrack: Multi-Object Tracking by
           Associating Every Detection Box" (ECCV 2022)
"""

import numpy as np
from typing import List, Dict, Optional
from .base import TrackerBase, Track, TrackState


class KalmanFilter:
    """Simple 2D Kalman filter for bounding box tracking."""

    def __init__(self):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.dim_x = 8
        self.dim_z = 4

        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 10
        self.Q = np.eye(self.dim_x) * 1
        self.R = np.eye(self.dim_z) * 1

        # State transition
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1

        # Measurement
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1

    def init(self, measurement):
        self.x[:4] = measurement
        self.x[4:] = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, measurement):
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        return self.x[:4].copy()


class STrack:
    """Single tracked object."""

    _next_id = 1

    def __init__(self, bbox, label, confidence):
        self.track_id = STrack._next_id
        STrack._next_id += 1

        self.kf = KalmanFilter()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.init(np.array([cx, cy, w, h], dtype=float))

        self.bbox = list(bbox)
        self.label = label
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.is_activated = True

    def predict(self):
        state = self.kf.predict()
        cx, cy, w, h = state
        self.bbox = [
            int(cx - w / 2), int(cy - h / 2),
            int(cx + w / 2), int(cy + h / 2)
        ]
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, label, confidence):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        state = self.kf.update(np.array([cx, cy, w, h], dtype=float))

        cx, cy, w, h = state
        self.bbox = [
            int(cx - w / 2), int(cy - h / 2),
            int(cx + w / 2), int(cy + h / 2)
        ]
        self.label = label
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.is_activated = True


class ByteTracker(TrackerBase):
    """
    ByteTrack multi-object tracker.

    Args:
        track_thresh: confidence threshold for first association (default 0.5)
        match_thresh: IoU threshold for matching (default 0.8)
        track_buffer: frames to keep lost tracks (default 30)
    """

    name = "bytetrack"

    def __init__(self, track_thresh: float = 0.5,
                 match_thresh: float = 0.8,
                 track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0

        STrack._next_id = 1

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.tracks = {}
        self.frame_count = 0
        STrack._next_id = 1

    def update(self, detections: List[Dict], frame_idx: int,
               timestamp: str = "") -> List[TrackState]:
        self.frame_count += 1
        results = []

        if not detections:
            # Predict existing tracks
            for strack in self.tracked_stracks:
                strack.predict()
            self._remove_lost()
            return results

        # Split detections by confidence
        high_dets = []
        low_dets = []
        for d in detections:
            if d["confidence"] >= self.track_thresh:
                high_dets.append(d)
            else:
                low_dets.append(d)

        # Predict existing tracks
        for strack in self.tracked_stracks:
            strack.predict()
        for strack in self.lost_stracks:
            strack.predict()

        # First association: high confidence dets vs tracked
        unmatched_tracks = list(range(len(self.tracked_stracks)))
        unmatched_dets = list(range(len(high_dets)))

        if self.tracked_stracks and high_dets:
            iou_matrix = self._iou_matrix(
                [s.bbox for s in self.tracked_stracks],
                [d["bbox"] for d in high_dets]
            )
            matched, unmatched_tracks, unmatched_dets = \
                self._linear_assignment(iou_matrix, self.match_thresh)

            for t_idx, d_idx in matched:
                strack = self.tracked_stracks[t_idx]
                det = high_dets[d_idx]
                strack.update(det["bbox"], det["label"], det["confidence"])

        # Second association: low confidence dets vs remaining tracked
        remaining_tracks = [self.tracked_stracks[i] for i in unmatched_tracks]
        if remaining_tracks and low_dets:
            iou_matrix = self._iou_matrix(
                [s.bbox for s in remaining_tracks],
                [d["bbox"] for d in low_dets]
            )
            matched2, still_unmatched, _ = \
                self._linear_assignment(iou_matrix, 0.5)

            for t_idx, d_idx in matched2:
                strack = remaining_tracks[t_idx]
                det = low_dets[d_idx]
                strack.update(det["bbox"], det["label"], det["confidence"])

            # Move unmatched tracks to lost
            for t_idx in still_unmatched:
                strack = remaining_tracks[t_idx]
                if strack.time_since_update > self.track_buffer:
                    self.removed_stracks.append(strack)
                else:
                    self.lost_stracks.append(strack)
        else:
            for t_idx in unmatched_tracks:
                strack = self.tracked_stracks[t_idx]
                if strack.time_since_update > self.track_buffer:
                    self.removed_stracks.append(strack)
                else:
                    self.lost_stracks.append(strack)

        # Try to recover lost tracks with unmatched high dets
        unmatched_high = [high_dets[i] for i in unmatched_dets]
        if self.lost_stracks and unmatched_high:
            iou_matrix = self._iou_matrix(
                [s.bbox for s in self.lost_stracks],
                [d["bbox"] for d in unmatched_high]
            )
            matched3, _, still_unmatched_dets = \
                self._linear_assignment(iou_matrix, self.match_thresh)

            recovered = set()
            for t_idx, d_idx in matched3:
                strack = self.lost_stracks[t_idx]
                det = unmatched_high[d_idx]
                strack.update(det["bbox"], det["label"], det["confidence"])
                self.tracked_stracks.append(strack)
                recovered.add(t_idx)

            self.lost_stracks = [
                s for i, s in enumerate(self.lost_stracks)
                if i not in recovered
            ]

            remaining_new = [unmatched_high[i] for i in still_unmatched_dets]
        else:
            remaining_new = unmatched_high

        # Create new tracks for unmatched high-confidence detections
        for det in remaining_new:
            strack = STrack(det["bbox"], det["label"], det["confidence"])
            self.tracked_stracks.append(strack)

        self._remove_lost()

        # Update track history and build results
        for strack in self.tracked_stracks:
            if not strack.is_activated:
                continue

            state = TrackState(
                track_id=strack.track_id,
                bbox=strack.bbox,
                label=strack.label,
                confidence=strack.confidence,
                frame_idx=frame_idx,
                timestamp=timestamp,
            )
            results.append(state)

            # Update Track history
            if strack.track_id not in self.tracks:
                self.tracks[strack.track_id] = Track(
                    track_id=strack.track_id,
                    label=strack.label
                )
            self.tracks[strack.track_id].add_state(state)

        return results

    def get_tracks(self) -> Dict[int, Track]:
        return self.tracks

    def _remove_lost(self):
        """Remove tracks that have been lost too long."""
        active = []
        for strack in self.tracked_stracks:
            if strack.time_since_update <= self.track_buffer:
                active.append(strack)
            else:
                self.lost_stracks.append(strack)
        self.tracked_stracks = active

        # Clean old lost tracks
        self.lost_stracks = [
            s for s in self.lost_stracks
            if s.time_since_update <= self.track_buffer * 2
        ]

    @staticmethod
    def _iou_matrix(bboxes1, bboxes2):
        """Compute IoU matrix between two sets of bboxes."""
        n = len(bboxes1)
        m = len(bboxes2)
        matrix = np.zeros((n, m))

        for i, b1 in enumerate(bboxes1):
            for j, b2 in enumerate(bboxes2):
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter

                matrix[i, j] = inter / union if union > 0 else 0

        return matrix

    @staticmethod
    def _linear_assignment(iou_matrix, threshold):
        """
        Simple greedy assignment (no scipy dependency).
        Returns: matched pairs, unmatched row indices, unmatched col indices.
        """
        n, m = iou_matrix.shape
        matched = []
        used_rows = set()
        used_cols = set()

        # Greedy: pick highest IoU pairs first
        while True:
            if len(used_rows) >= n or len(used_cols) >= m:
                break

            # Mask used entries
            masked = iou_matrix.copy()
            for r in used_rows:
                masked[r, :] = -1
            for c in used_cols:
                masked[:, c] = -1

            best = masked.max()
            if best < threshold:
                break

            r, c = np.unravel_index(masked.argmax(), masked.shape)
            matched.append((r, c))
            used_rows.add(r)
            used_cols.add(c)

        unmatched_rows = [i for i in range(n) if i not in used_rows]
        unmatched_cols = [i for i in range(m) if i not in used_cols]

        return matched, unmatched_rows, unmatched_cols
