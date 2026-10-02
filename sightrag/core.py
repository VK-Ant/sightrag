"""
SightRAG v0.5 — See. Search. Retrieve. Track. Understand.

Basic:
    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")

With segmentation:
    rag = SightRAG(segment=True)
    results = rag.query("find person")  # results include masks

With tracking:
    rag = SightRAG(track=True)
    rag.index("./security_footage.mp4", fps=5)  # video
    rag.index("./photos/")                       # image folder (treated as frames)
    results = rag.query("find person in red shirt")
    timeline = rag.timeline(results[0])

Find + Track:
    rag = SightRAG(track=True)
    rag.index("./footage.mp4", fps=5)
    timeline = rag.find_and_track("person in red shirt")

With OCR:
    rag = SightRAG(ocr=True)
    results = rag.query("find Calgon")

With multimodal understanding:
    rag = SightRAG(ocr=True, multimodal="qwen2-vl")
    results = rag.query("find damaged product", understand=True)
"""

import os
import numpy as np
from pathlib import Path
from .backends import auto_select_backend
from .detectors.base import DetectorBase
from .embedders.base import EmbedderBase

SIGHTRAG_HOME = os.path.join(Path.home(), ".sightrag")


class SightRAG:

    def __init__(self,
                 detector=None,
                 embedder=None,
                 store="sqlite",
                 domain_hint=None,
                 index_path=None,
                 rerank=False,
                 ocr=False,
                 multimodal=None,
                 api_key=None,
                 # v0.5 — segmentation + tracking
                 segment=False,
                 segmentor=None,
                 track=False,
                 tracker=None):

        self.domain_hint = domain_hint
        self._store_type = store if isinstance(store, str) else "custom"
        self._index_path = index_path or os.path.join(SIGHTRAG_HOME, "index")
        self._rerank = rerank
        self._reranker = None
        self._ocr = None
        self._multimodal = None
        self._segmentor = None
        self._tracker = None
        self._segment = segment
        self._track = track

        os.makedirs(SIGHTRAG_HOME, exist_ok=True)

        # Capabilities list for status message
        _caps = []

        # Backend
        self._backend = auto_select_backend()

        # Detector
        if detector is None:
            self._detector = self._backend
        elif isinstance(detector, str):
            self._detector = self._load_detector(detector)
        elif isinstance(detector, DetectorBase):
            self._detector = detector
        else:
            raise TypeError("detector must be string or DetectorBase")

        # Embedder
        if embedder is None:
            self._embedder = self._backend
        elif isinstance(embedder, str):
            self._embedder = self._load_embedder(embedder)
        elif isinstance(embedder, EmbedderBase):
            self._embedder = embedder
        else:
            raise TypeError("embedder must be string or EmbedderBase")

        # Segmentor (v0.5)
        if segment or segmentor is not None:
            self._segmentor = self._load_segmentor(segmentor)
            self._segment = True

        # Tracker (v0.5)
        if track or tracker is not None:
            self._tracker = self._load_tracker(tracker)
            self._track = True

        # OCR
        if ocr:
            from .ocr import OCREngine
            self._ocr = OCREngine()

        # Multimodal
        if multimodal:
            from .multimodal import MultimodalEngine
            self._multimodal = MultimodalEngine(model=multimodal, api_key=api_key)

        # Re-ranker
        if rerank:
            from .reranker import ReRanker
            self._reranker = ReRanker()

        # Store
        self._store = self._init_store(store, self._index_path)

        # Indexer + Retriever
        from .indexer import Indexer
        from .retriever import Retriever

        self._indexer = Indexer(
            self._detector, self._embedder, self._store, self._ocr,
            segmentor=self._segmentor, tracker=self._tracker
        )
        self._retriever = Retriever(
            self._embedder, self._detector, self._store, domain_hint
        )

        # Build capabilities summary
        if self._segment:
            _caps.append("segment")
        if self._track:
            _caps.append("track")
        if self._ocr:
            _caps.append("ocr")
        if self._multimodal:
            _caps.append("multimodal")
        if self._reranker:
            _caps.append("rerank")

        cap_str = f" [{', '.join(_caps)}]" if _caps else ""
        print(f"[SightRAG] Ready.{cap_str}")

    def _load_detector(self, name):
        if name in ("grounding-dino", "grounding_dino", "gdino"):
            from .detectors.grounding_dino import GroundingDINODetector
            return GroundingDINODetector(text_prompt=self.domain_hint)
        elif name in ("yolo", "yolo11"):
            return self._backend
        else:
            raise ValueError(f"Unknown detector: {name}")

    def _load_embedder(self, name):
        if name in ("reid", "re-id", "person-reid"):
            from .embedders.reid_embedder import ReIDEmbedder
            return ReIDEmbedder()
        elif name in ("clip", "clip-vit"):
            return self._backend
        else:
            raise ValueError(f"Unknown embedder: {name}")

    def _load_segmentor(self, segmentor):
        """Load segmentor by name or instance."""
        from .segmentors.base import SegmentorBase

        if segmentor is None:
            # Default: YOLO-Seg
            from .segmentors.yolo_segmentor import YOLOSegmentor
            return YOLOSegmentor()
        elif isinstance(segmentor, str):
            if segmentor in ("yolo", "yolo-seg", "yolo11-seg"):
                from .segmentors.yolo_segmentor import YOLOSegmentor
                return YOLOSegmentor()
            elif segmentor in ("sam2", "sam-2", "segment-anything"):
                from .segmentors.sam2_segmentor import SAM2Segmentor
                return SAM2Segmentor(detector=self._detector)
            elif segmentor.startswith(("facebook/", "shi-labs/", "nvidia/")):
                # HuggingFace model ID
                from .segmentors.hf_segmentor import HFSegmentor
                return HFSegmentor(segmentor)
            else:
                raise ValueError(
                    f"Unknown segmentor: {segmentor}. "
                    "Use 'yolo', 'sam2', or a HuggingFace model ID."
                )
        elif isinstance(segmentor, SegmentorBase):
            return segmentor
        else:
            raise TypeError("segmentor must be string or SegmentorBase")

    def _load_tracker(self, tracker):
        """Load tracker by name or instance."""
        from .trackers.base import TrackerBase

        if tracker is None:
            # Default: ByteTrack (pure Python, no extra weights)
            from .trackers.bytetrack import ByteTracker
            return ByteTracker()
        elif isinstance(tracker, str):
            if tracker in ("bytetrack", "byte"):
                from .trackers.bytetrack import ByteTracker
                return ByteTracker()
            elif tracker in ("botsort", "bot-sort"):
                from .trackers.botsort import BoTSORTTracker
                return BoTSORTTracker(embedder=self._embedder)
            else:
                raise ValueError(
                    f"Unknown tracker: {tracker}. "
                    "Use 'bytetrack' or 'botsort'."
                )
        elif isinstance(tracker, TrackerBase):
            return tracker
        else:
            raise TypeError("tracker must be string or TrackerBase")

    def _init_store(self, store_type, path):
        if isinstance(store_type, str):
            if store_type == "sqlite":
                from .store.sqlite_store import SQLiteStore
                return SQLiteStore(path)
            elif store_type == "chroma":
                try:
                    from .store.chroma_store import ChromaStore
                    return ChromaStore(path)
                except ImportError:
                    from .store.sqlite_store import SQLiteStore
                    return SQLiteStore(path)
            elif store_type == "qdrant":
                from .store.qdrant_store import QdrantStore
                return QdrantStore()
            else:
                raise ValueError(f"Unknown store: {store_type}")
        return store_type

    def index(self, path=None, source=None, camera_id=0, fps=1):
        """
        Index images, video, or camera.

        All capabilities (detection, segmentation, tracking) work on
        all input types: single images, image folders, video files,
        video streams, and CCTV cameras.
        """
        if source == "camera":
            self._indexer.index_camera(camera_id=camera_id, fps=fps)
            return self
        if path is None:
            raise ValueError("Provide path or source='camera'")
        if os.path.isdir(path):
            self._indexer.index_folder(path, fps=fps)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv"}:
                self._indexer.index_video(path, fps=fps)
            else:
                self._index_single_image(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")
        return self

    def _index_single_image(self, path, frame_idx=1, timestamp="0.00"):
        from .utils.image import load_image
        image = load_image(path)
        regions = self._detector.detect(image)

        # Optional segmentation
        if self._segmentor:
            seg_results = self._segmentor.segment(image)
            regions = self._merge_detection_segmentation(regions, seg_results)

        frame_detections = []
        for j, region in enumerate(regions):
            embedding = self._embedder.embed_image(region["crop"])
            if not np.allclose(embedding, 0):
                metadata = {
                    "image_path": str(path),
                    "bbox": region["bbox"],
                    "label": region["label"],
                    "confidence": region["confidence"],
                    "source_type": "image",
                    "timestamp": timestamp,
                    "ocr_text": "",
                    "has_mask": "mask" in region,
                }
                # OCR at index time
                if self._ocr:
                    ocr_result = self._ocr.read(region["crop"])
                    metadata["ocr_text"] = ocr_result["text"]

                self._store.add(f"img_{j}", embedding, metadata)

                # Collect for tracker
                if self._tracker:
                    frame_detections.append({
                        "bbox": region["bbox"],
                        "label": region["label"],
                        "confidence": region["confidence"],
                        "embedding": embedding,
                        "crop": region.get("crop"),
                    })

        # Run tracker on this image's detections
        if self._tracker and frame_detections:
            track_states = self._tracker.update(
                frame_detections, frame_idx=frame_idx,
                timestamp=timestamp
            )
            for ts in track_states:
                self._store.add_track_detection(
                    track_id=ts.track_id,
                    video_path=str(path),
                    frame_idx=ts.frame_idx,
                    timestamp=ts.timestamp,
                    bbox=ts.bbox,
                    label=ts.label,
                    confidence=ts.confidence,
                )

        print(f"[SightRAG] 1 image indexed. Total: {self.count()} regions.")

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
                # Attach mask from segmentation to detection
                det["mask"] = seg_results[best_seg].get("mask")
                used_seg.add(best_seg)
            merged.append(det)

        # Add unmatched segmentation results as new detections
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

    def query(self, text=None, reference=None, top_k=5, understand=False,
              segment=False):
        """
        Search indexed content.

        rag.query("find person")                          # fast visual
        rag.query("find Calgon")                          # matches OCR text
        rag.query("find damaged product", understand=True) # LLM understanding
        rag.query("find person", segment=True)             # with masks
        """
        if text is None and reference is None:
            raise ValueError("Provide text or reference image.")

        # Fetch more if re-ranking or understanding needed
        fetch_k = top_k
        if self._reranker or (understand and self._multimodal):
            fetch_k = min(top_k * 20, self._store.count() or top_k)

        if text:
            results = self._retriever.query_text(text, fetch_k)

            # Check OCR text matches
            if self._ocr and text:
                results = self._boost_ocr_matches(results, text)
        else:
            results = self._retriever.query_reference(reference, fetch_k)

        # Re-rank with cross-encoder
        if self._reranker and text and len(results) > top_k:
            results = self._reranker.rerank(text, results, top_k)

        # Multimodal understanding (only when explicitly asked)
        if understand and self._multimodal and text:
            candidates = results[:min(10, len(results))]
            results = self._multimodal.rerank_with_understanding(
                text, candidates, top_k
            )

        # On-demand segmentation for query results
        if (segment or self._segment) and self._segmentor:
            results = self._add_masks_to_results(results)

        # Add track info if available
        if self._track:
            results = self._add_track_info(results)

        return results[:top_k]

    def _add_masks_to_results(self, results):
        """Run segmentation on query results to add masks."""
        from .utils.image import load_image

        for r in results:
            if r.get("mask") is not None:
                continue
            path = r.get("image_path", "")
            if not os.path.exists(path):
                continue
            try:
                image = load_image(path)
                bbox = r.get("bbox", [])
                if bbox and len(bbox) == 4:
                    mask = self._segmentor.segment_region(image, bbox)
                    r["mask"] = mask
            except Exception:
                pass
        return results

    def _add_track_info(self, results):
        """Add track_id and track summary to results from stored tracks."""
        for r in results:
            track_info = self._store.find_track_for_detection(
                r.get("image_path", ""),
                r.get("bbox", []),
                r.get("timestamp", "")
            )
            if track_info:
                r["track_id"] = track_info.get("track_id")
                r["track_label"] = track_info.get("label")
                r["track_first_seen"] = track_info.get("first_timestamp")
                r["track_last_seen"] = track_info.get("last_timestamp")
                r["track_frame_count"] = track_info.get("frame_count")
        return results

    def timeline(self, result_or_track_id):
        """
        Get full timeline for a tracked object.

        Usage:
            results = rag.query("find person in red")
            timeline = rag.timeline(results[0])
            # or
            timeline = rag.timeline(track_id=3)

        Returns:
            dict with track_id, label, first/last frame/timestamp,
            frame_count, and trajectory (list of frame states).
        """
        if not self._track:
            raise RuntimeError(
                "Tracking not enabled. Use SightRAG(track=True)."
            )

        if isinstance(result_or_track_id, dict):
            track_id = result_or_track_id.get("track_id")
            if track_id is None:
                raise ValueError(
                    "Result has no track_id. "
                    "Index video with track=True first."
                )
        else:
            track_id = int(result_or_track_id)

        track_data = self._store.get_track(track_id)
        if track_data is None:
            raise ValueError(f"Track {track_id} not found.")

        return track_data

    def find_and_track(self, query_text, video_path=None, top_k=1):
        """
        Find an object by description and return its full track timeline.

        Combines query + timeline in one call.

        Usage:
            timeline = rag.find_and_track("person in red shirt")

        Returns:
            list of track timelines for matched objects
        """
        if not self._track:
            raise RuntimeError(
                "Tracking not enabled. Use SightRAG(track=True)."
            )

        # If video_path given, index it first
        if video_path is not None:
            self.index(video_path)

        results = self.query(text=query_text, top_k=top_k * 3)

        # Collect unique tracks
        seen_tracks = set()
        timelines = []

        for r in results:
            tid = r.get("track_id")
            if tid is None or tid in seen_tracks:
                continue
            seen_tracks.add(tid)
            try:
                tl = self.timeline(tid)
                timelines.append(tl)
            except ValueError:
                pass
            if len(timelines) >= top_k:
                break

        if not timelines:
            print("[SightRAG] No tracked objects found for query.")

        return timelines

    def get_all_tracks(self):
        """
        Return all tracks from the store.

        Returns:
            list of track dicts with id, label, timestamps, frame_count
        """
        if not self._track:
            raise RuntimeError(
                "Tracking not enabled. Use SightRAG(track=True)."
            )
        return self._store.get_all_tracks()

    def _boost_ocr_matches(self, results, query):
        """Boost results where OCR text matches query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for r in results:
            ocr_text = r.get("ocr_text", "").lower()
            if not ocr_text:
                continue
            ocr_words = set(ocr_text.split())

            # Check word overlap
            matches = query_words & ocr_words
            if matches:
                boost = len(matches) / len(query_words) * 0.3
                r["score"] = min(1.0, r.get("score", 0) + boost)
                r["ocr_match"] = True

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results

    def show(self, results, save=None, max_show=5):
        """Visualize results with bounding boxes, masks, and tracks."""
        from .visualizer import show_results
        show_results(results, save=save, max_show=max_show)

    def count(self):
        return self._store.count()

    def clear(self):
        self._store.clear()
        if self._tracker:
            self._tracker.reset()
        print("[SightRAG] Index cleared.")
        return self

    def __repr__(self):
        features = [f"indexed={self.count()}"]
        if self._segmentor:
            features.append("segment=True")
        if self._tracker:
            features.append("track=True")
        if self._ocr:
            features.append("ocr=True")
        if self._multimodal:
            features.append("multimodal=True")
        return f"SightRAG({', '.join(features)})"
