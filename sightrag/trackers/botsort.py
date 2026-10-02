"""
BoT-SORT tracker - ByteTrack + Re-ID embedding matching.

Uses appearance embeddings alongside IoU for better long-term tracking.
Better at re-identifying objects after occlusion or camera cuts.
Requires an embedder (CLIP or Re-ID) for appearance features.

Reference: Aharon et al. "BoT-SORT: Robust Associations Multi-Pedestrian
           Tracking" (arXiv 2206.14651)
"""

import numpy as np
from typing import List, Dict, Optional
from .base import TrackerBase, Track, TrackState
from .bytetrack import ByteTracker, STrack, KalmanFilter


class STrackReID(STrack):
    """STrack with appearance embedding."""

    def __init__(self, bbox, label, confidence, embedding=None):
        super().__init__(bbox, label, confidence)
        self.embedding = embedding
        self.smooth_embedding = embedding

    def update(self, bbox, label, confidence, embedding=None):
        super().update(bbox, label, confidence)
        if embedding is not None:
            if self.smooth_embedding is not None:
                # Exponential moving average
                alpha = 0.9
                self.smooth_embedding = (
                    alpha * self.smooth_embedding +
                    (1 - alpha) * embedding
                )
                norm = np.linalg.norm(self.smooth_embedding)
                if norm > 0:
                    self.smooth_embedding /= norm
            else:
                self.smooth_embedding = embedding
            self.embedding = embedding


class BoTSORTTracker(TrackerBase):
    """
    BoT-SORT: ByteTrack + Re-ID appearance matching.

    Args:
        embedder: embedding model (must have embed_image method)
        track_thresh: confidence threshold (default 0.5)
        match_thresh: IoU threshold (default 0.8)
        appearance_thresh: cosine similarity threshold (default 0.25)
        track_buffer: frames to keep lost tracks (default 30)
        lambda_weight: balance between IoU and appearance (default 0.98)
    """

    name = "botsort"

    def __init__(self, embedder=None,
                 track_thresh: float = 0.5,
                 match_thresh: float = 0.8,
                 appearance_thresh: float = 0.25,
                 track_buffer: int = 30,
                 lambda_weight: float = 0.98):
        self.embedder = embedder
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.appearance_thresh = appearance_thresh
        self.track_buffer = track_buffer
        self.lambda_weight = lambda_weight

        self.tracked_stracks: List[STrackReID] = []
        self.lost_stracks: List[STrackReID] = []
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0

        STrack._next_id = 1

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.tracks = {}
        self.frame_count = 0
        STrack._next_id = 1

    def update(self, detections: List[Dict], frame_idx: int,
               timestamp: str = "") -> List[TrackState]:
        self.frame_count += 1
        results = []

        if not detections:
            for strack in self.tracked_stracks:
                strack.predict()
            self._remove_lost()
            return results

        # Get embeddings for detections
        embeddings = []
        for d in detections:
            emb = d.get("embedding")
            if emb is None and self.embedder and "crop" in d:
                try:
                    emb = self.embedder.embed_image(d["crop"])
                except Exception:
                    emb = None
            embeddings.append(emb)

        # Split by confidence
        high_dets = []
        high_embs = []
        low_dets = []
        for i, d in enumerate(detections):
            if d["confidence"] >= self.track_thresh:
                high_dets.append(d)
                high_embs.append(embeddings[i])
            else:
                low_dets.append(d)

        # Predict
        for strack in self.tracked_stracks:
            strack.predict()
        for strack in self.lost_stracks:
            strack.predict()

        # First association: combined IoU + appearance
        unmatched_tracks = list(range(len(self.tracked_stracks)))
        unmatched_dets = list(range(len(high_dets)))

        if self.tracked_stracks and high_dets:
            cost_matrix = self._combined_cost(
                self.tracked_stracks, high_dets, high_embs
            )
            matched, unmatched_tracks, unmatched_dets = \
                self._greedy_assignment(cost_matrix, self.match_thresh)

            for t_idx, d_idx in matched:
                strack = self.tracked_stracks[t_idx]
                det = high_dets[d_idx]
                strack.update(
                    det["bbox"], det["label"],
                    det["confidence"], high_embs[d_idx]
                )

        # Second association: remaining tracks vs low dets (IoU only)
        remaining = [self.tracked_stracks[i] for i in unmatched_tracks]
        if remaining and low_dets:
            iou_mat = ByteTracker._iou_matrix(
                [s.bbox for s in remaining],
                [d["bbox"] for d in low_dets]
            )
            matched2, still_unmatched, _ = \
                ByteTracker._linear_assignment(iou_mat, 0.5)

            for t_idx, d_idx in matched2:
                remaining[t_idx].update(
                    low_dets[d_idx]["bbox"],
                    low_dets[d_idx]["label"],
                    low_dets[d_idx]["confidence"]
                )

            for t_idx in still_unmatched:
                strack = remaining[t_idx]
                if strack.time_since_update > self.track_buffer:
                    pass
                else:
                    self.lost_stracks.append(strack)
        else:
            for t_idx in unmatched_tracks:
                strack = self.tracked_stracks[t_idx]
                self.lost_stracks.append(strack)

        # Recovery: lost tracks vs unmatched high dets (appearance-heavy)
        unmatched_high = [(high_dets[i], high_embs[i]) for i in unmatched_dets]
        if self.lost_stracks and unmatched_high:
            cost_matrix = self._combined_cost(
                self.lost_stracks,
                [u[0] for u in unmatched_high],
                [u[1] for u in unmatched_high],
                iou_weight=0.3  # favor appearance for recovery
            )
            matched3, _, still_unmatched_dets = \
                self._greedy_assignment(cost_matrix, 0.6)

            recovered = set()
            for t_idx, d_idx in matched3:
                strack = self.lost_stracks[t_idx]
                det, emb = unmatched_high[d_idx]
                strack.update(det["bbox"], det["label"],
                              det["confidence"], emb)
                self.tracked_stracks.append(strack)
                recovered.add(t_idx)

            self.lost_stracks = [
                s for i, s in enumerate(self.lost_stracks)
                if i not in recovered
            ]

            remaining_new = [unmatched_high[i] for i in still_unmatched_dets]
        else:
            remaining_new = unmatched_high

        # New tracks
        for det, emb in remaining_new:
            strack = STrackReID(
                det["bbox"], det["label"], det["confidence"], emb
            )
            self.tracked_stracks.append(strack)

        self._remove_lost()

        # Build results
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
                embedding=strack.embedding,
            )
            results.append(state)

            if strack.track_id not in self.tracks:
                self.tracks[strack.track_id] = Track(
                    track_id=strack.track_id,
                    label=strack.label
                )
            self.tracks[strack.track_id].add_state(state)

        return results

    def get_tracks(self) -> Dict[int, Track]:
        return self.tracks

    def _combined_cost(self, stracks, dets, embs, iou_weight=None):
        """
        Combined cost = lambda * IoU_cost + (1-lambda) * appearance_cost.
        """
        lam = iou_weight if iou_weight is not None else self.lambda_weight
        n = len(stracks)
        m = len(dets)

        # IoU distance
        iou_mat = ByteTracker._iou_matrix(
            [s.bbox for s in stracks],
            [d["bbox"] for d in dets]
        )
        iou_cost = 1.0 - iou_mat

        # Appearance distance (cosine)
        app_cost = np.ones((n, m))
        for i, strack in enumerate(stracks):
            if strack.smooth_embedding is None:
                continue
            for j, emb in enumerate(embs):
                if emb is None:
                    continue
                sim = float(np.dot(strack.smooth_embedding, emb))
                app_cost[i, j] = 1.0 - max(0, sim)

        # Combined
        cost = lam * iou_cost + (1 - lam) * app_cost
        return cost

    @staticmethod
    def _greedy_assignment(cost_matrix, threshold):
        """Greedy assignment on cost matrix (lower = better)."""
        n, m = cost_matrix.shape
        matched = []
        used_rows = set()
        used_cols = set()

        while True:
            if len(used_rows) >= n or len(used_cols) >= m:
                break

            masked = cost_matrix.copy()
            for r in used_rows:
                masked[r, :] = 999
            for c in used_cols:
                masked[:, c] = 999

            best = masked.min()
            if best > threshold:
                break

            r, c = np.unravel_index(masked.argmin(), masked.shape)
            matched.append((r, c))
            used_rows.add(r)
            used_cols.add(c)

        unmatched_rows = [i for i in range(n) if i not in used_rows]
        unmatched_cols = [i for i in range(m) if i not in used_cols]

        return matched, unmatched_rows, unmatched_cols

    def _remove_lost(self):
        active = []
        for strack in self.tracked_stracks:
            if strack.time_since_update <= self.track_buffer:
                active.append(strack)
            else:
                self.lost_stracks.append(strack)
        self.tracked_stracks = active

        self.lost_stracks = [
            s for s in self.lost_stracks
            if s.time_since_update <= self.track_buffer * 2
        ]
