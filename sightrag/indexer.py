"""SightRAG indexer — uses C++ core when available, Python fallback."""

import os
import numpy as np
from pathlib import Path
from .utils.image import load_image, SUPPORTED_FORMATS as IMAGE_FORMATS
from .utils.video import extract_frames, SUPPORTED_FORMATS as VIDEO_FORMATS

# Try C++ core — silent fallback
try:
    import sightrag_cpp
    FAST_MODE = True
except ImportError:
    FAST_MODE = False


class Indexer:

    def __init__(self, detector, embedder, store, ocr=None,
                 segmentor=None, tracker=None):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.ocr = ocr
        self.segmentor = segmentor
        self.tracker = tracker

    def _index_image(self, path_str, image, prefix,
                     frame_idx=None, timestamp="0.00"):
        count = 0
        regions = self.detector.detect(image)

        # Optional segmentation
        if self.segmentor:
            seg_results = self.segmentor.segment(image)
            regions = self._merge_detection_segmentation(regions, seg_results)

        frame_detections = []
        for j, region in enumerate(regions):
            embedding = self.embedder.embed_image(region["crop"])
            if not np.allclose(embedding, 0):
                metadata = {
                    "image_path": path_str,
                    "bbox": region["bbox"],
                    "label": region["label"],
                    "confidence": region["confidence"],
                    "source_type": "image",
                    "timestamp": timestamp,
                    "ocr_text": "",
                    "has_mask": "mask" in region,
                }
                # OCR at index time — reads text once, stored permanently
                if self.ocr:
                    ocr_result = self.ocr.read(region["crop"])
                    metadata["ocr_text"] = ocr_result["text"]

                self.store.add(f"{prefix}_{j}", embedding, metadata)
                count += 1

                # Collect for tracker
                if self.tracker and frame_idx is not None:
                    frame_detections.append({
                        "bbox": region["bbox"],
                        "label": region["label"],
                        "confidence": region["confidence"],
                        "embedding": embedding,
                        "crop": region.get("crop"),
                    })

        # Run tracker on this frame's detections
        if self.tracker and frame_idx is not None and frame_detections:
            track_states = self.tracker.update(
                frame_detections, frame_idx=frame_idx,
                timestamp=timestamp
            )
            for ts in track_states:
                self.store.add_track_detection(
                    track_id=ts.track_id,
                    video_path=path_str,
                    frame_idx=ts.frame_idx,
                    timestamp=ts.timestamp,
                    bbox=ts.bbox,
                    label=ts.label,
                    confidence=ts.confidence,
                )

        return count

    def _merge_detection_segmentation(self, detections, seg_results):
        """Merge detection bboxes with segmentation masks."""
        if not seg_results:
            return detections

        merged = []
        used_seg = set()

        for det in detections:
            best_iou = 0
            best_seg = None
            for si, seg in enumerate(seg_results):
                if si in used_seg:
                    continue
                iou = self._bbox_iou(det["bbox"], seg["bbox"])
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_seg = si

            if best_seg is not None:
                det["mask"] = seg_results[best_seg].get("mask")
                used_seg.add(best_seg)
            merged.append(det)

        for si, seg in enumerate(seg_results):
            if si not in used_seg:
                merged.append(seg)

        return merged

    @staticmethod
    def _bbox_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def index_folder(self, folder_path, fps=1):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Find images
        image_paths = []
        for fmt in IMAGE_FORMATS:
            image_paths.extend(folder.glob(f"*{fmt}"))
            image_paths.extend(folder.glob(f"*{fmt.upper()}"))
        image_paths = sorted(set(image_paths))

        # Find videos
        video_paths = []
        for fmt in VIDEO_FORMATS:
            video_paths.extend(folder.glob(f"*{fmt}"))
            video_paths.extend(folder.glob(f"*{fmt.upper()}"))
        video_paths = sorted(set(video_paths))

        if not image_paths and not video_paths:
            raise ValueError(f"No images or videos in {folder}")

        print(f"[SightRAG] Found {len(image_paths)} images, "
              f"{len(video_paths)} videos")

        # Reset tracker for folder sequence (images treated as frames)
        if self.tracker and image_paths:
            self.tracker.reset()

        # Index images — when tracker enabled, treat as ordered frame sequence
        if image_paths:
            total = len(image_paths)
            for i, path in enumerate(image_paths, 1):
                try:
                    image = load_image(str(path))
                    # Pass frame_idx so tracker runs on each image
                    timestamp = f"{(i-1)/max(fps,1):.2f}"
                    self._index_image(
                        str(path), image, path.stem,
                        frame_idx=i if self.tracker else None,
                        timestamp=timestamp,
                    )
                    pct = int((i / total) * 40)
                    bar = "█" * pct + "░" * (40 - pct)
                    print(f"\r  [{bar}] {i}/{total} images", end="", flush=True)
                except Exception as e:
                    print(f"\n  Skipping {path.name}: {e}")
            print()

            # Finalize tracks from image folder
            if self.tracker:
                tracks = self.tracker.get_tracks()
                for tid, track in tracks.items():
                    self.store.save_track(
                        track_id=tid,
                        video_path=str(folder_path),
                        label=track.label,
                        first_frame=track.first_frame,
                        last_frame=track.last_frame,
                        first_timestamp=track.first_timestamp,
                        last_timestamp=track.last_timestamp,
                        frame_count=track.frame_count,
                        trajectory=track.bbox_trajectory,
                    )
                if tracks:
                    print(f"[SightRAG] {len(tracks)} objects tracked across images.")

        # Index videos
        if video_paths:
            for v_idx, vpath in enumerate(video_paths, 1):
                try:
                    print(f"[SightRAG] Video {v_idx}/{len(video_paths)}: {vpath.name}")
                    self._index_video(str(vpath), fps)
                except Exception as e:
                    print(f"  Skipping {vpath.name}: {e}")

        print(f"[SightRAG] Done. {self.store.count()} regions indexed.")

    def index_video(self, video_path, fps=1):
        self._index_video(video_path, fps)
        print(f"[SightRAG] Done. {self.store.count()} regions indexed.")

        # Track summary
        if self.tracker:
            tracks = self.tracker.get_tracks()
            if tracks:
                print(f"[SightRAG] {len(tracks)} objects tracked.")

    def _index_video(self, video_path, fps=1):
        video_name = Path(video_path).stem

        if FAST_MODE:
            # C++ frame extraction
            raw_frames = sightrag_cpp.extract_frames(video_path, fps)
            from PIL import Image
            frames = [(Image.fromarray(f[:, :, ::-1]), f"{i/fps:.2f}")
                      for i, f in enumerate(raw_frames)]
        else:
            frames = extract_frames(video_path, fps=fps)

        total = len(frames)
        print(f"  {total} frames extracted...")

        # Reset tracker for new video
        if self.tracker:
            self.tracker.reset()

        for i, (image, timestamp) in enumerate(frames, 1):
            try:
                regions = self.detector.detect(image)

                # Optional segmentation
                if self.segmentor:
                    seg_results = self.segmentor.segment(image)
                    regions = self._merge_detection_segmentation(
                        regions, seg_results
                    )

                # Build detection dicts for tracker
                frame_detections = []
                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    if np.allclose(embedding, 0):
                        continue

                    det_entry = {
                        "bbox": region["bbox"],
                        "label": region["label"],
                        "confidence": region["confidence"],
                        "embedding": embedding,
                        "crop": region.get("crop"),
                    }
                    frame_detections.append(det_entry)

                    # Store in vector index (same as v0.4)
                    metadata = {
                        "image_path": video_path,
                        "bbox": region["bbox"],
                        "label": region["label"],
                        "confidence": region["confidence"],
                        "timestamp": timestamp,
                        "source_type": "video",
                        "has_mask": "mask" in region,
                    }
                    self.store.add(
                        f"{video_name}_f{i}_r{j}", embedding, metadata
                    )

                # Run tracker on this frame's detections
                if self.tracker and frame_detections:
                    track_states = self.tracker.update(
                        frame_detections, frame_idx=i, timestamp=timestamp
                    )

                    # Store track-detection association
                    for ts in track_states:
                        self.store.add_track_detection(
                            track_id=ts.track_id,
                            video_path=video_path,
                            frame_idx=ts.frame_idx,
                            timestamp=ts.timestamp,
                            bbox=ts.bbox,
                            label=ts.label,
                            confidence=ts.confidence,
                        )

                pct = int((i / total) * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {i}/{total} frames", end="", flush=True)
            except Exception:
                pass
        print()

        # Finalize tracks — store track summaries
        if self.tracker:
            tracks = self.tracker.get_tracks()
            for tid, track in tracks.items():
                self.store.save_track(
                    track_id=tid,
                    video_path=video_path,
                    label=track.label,
                    first_frame=track.first_frame,
                    last_frame=track.last_frame,
                    first_timestamp=track.first_timestamp,
                    last_timestamp=track.last_timestamp,
                    frame_count=track.frame_count,
                    trajectory=track.bbox_trajectory,
                )

    def index_camera(self, camera_id=0, fps=1, buffer_seconds=60):
        from .utils.camera import capture_frames
        print(f"[SightRAG] Camera {camera_id}. Press Ctrl+C to stop.")

        if self.tracker:
            self.tracker.reset()

        count = 0
        frame_idx = 0
        try:
            for image, timestamp in capture_frames(camera_id, fps, buffer_seconds):
                frame_idx += 1
                regions = self.detector.detect(image)

                frame_detections = []
                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    if not np.allclose(embedding, 0):
                        self.store.add(
                            f"cam{camera_id}_{timestamp}_{j}", embedding, {
                                "image_path": f"camera_{camera_id}",
                                "bbox": region["bbox"],
                                "label": region["label"],
                                "confidence": region["confidence"],
                                "timestamp": timestamp,
                                "source_type": "camera"
                            }
                        )
                        if self.tracker:
                            frame_detections.append({
                                "bbox": region["bbox"],
                                "label": region["label"],
                                "confidence": region["confidence"],
                                "embedding": embedding,
                                "crop": region.get("crop"),
                            })

                # Track
                if self.tracker and frame_detections:
                    self.tracker.update(
                        frame_detections, frame_idx=frame_idx,
                        timestamp=timestamp
                    )

                count += 1
                track_info = ""
                if self.tracker:
                    active = len(self.tracker.get_active_tracks())
                    track_info = f" | {active} tracked"
                print(
                    f"\r[SightRAG] {count} frames | {timestamp}{track_info}",
                    end="", flush=True
                )
        except KeyboardInterrupt:
            print(f"\n[SightRAG] Stopped. {self.store.count()} regions indexed.")
            if self.tracker:
                tracks = self.tracker.get_tracks()
                print(f"[SightRAG] {len(tracks)} objects tracked.")
