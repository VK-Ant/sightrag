# sightrag/indexer.py
# Handles image folder, video, and camera indexing

import uuid
from pathlib import Path
from .detector import Detector
from .embedder import Embedder
from .utils.image import load_image, get_image_paths
from .utils.video import extract_frames


class Indexer:
    """
    Indexes images and video into the vector store.
    """

    def __init__(self, detector: Detector,
                 embedder: Embedder,
                 store,
                 domain_hint: str = None):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.domain_hint = domain_hint

    def index_folder(self, folder_path: str):
        """Index all images in a folder."""
        paths = get_image_paths(folder_path)
        total = len(paths)
        print(f"[SightRAG] Indexing {total} images...")

        for i, path in enumerate(paths, 1):
            try:
                image = load_image(str(path))
                regions = self.detector.detect(image)

                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    id = f"{path.stem}_{j}"
                    self.store.add(id, embedding, {
                        "image_path":  str(path),
                        "bbox":        region["bbox"],
                        "label":       region["label"],
                        "confidence":  region["confidence"],
                        "source_type": "image"
                    })

                # Progress
                pct = int((i / total) * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {i}/{total} images", end="")

            except Exception as e:
                print(f"\n[SightRAG] Skipping {path.name}: {e}")

        print(f"\n[SightRAG] Done. {self.store.count()} regions indexed.")

    def index_video(self, video_path: str, fps: int = 1):
        """Index video by extracting keyframes."""
        print(f"[SightRAG] Extracting frames at {fps} FPS...")
        frames = extract_frames(video_path, fps=fps)
        total = len(frames)
        print(f"[SightRAG] Indexing {total} frames...")

        for i, (image, timestamp) in enumerate(frames, 1):
            try:
                regions = self.detector.detect(image)

                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    id = f"frame_{i}_{j}"
                    self.store.add(id, embedding, {
                        "image_path":  video_path,
                        "bbox":        region["bbox"],
                        "label":       region["label"],
                        "confidence":  region["confidence"],
                        "timestamp":   timestamp,
                        "source_type": "video"
                    })

                pct = int((i / total) * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {i}/{total} frames", end="")

            except Exception as e:
                print(f"\n[SightRAG] Skipping frame {i}: {e}")

        print(f"\n[SightRAG] Done. {self.store.count()} regions indexed.")

    def index_camera(self, camera_id: int = 0,
                     fps: int = 1,
                     buffer_seconds: int = 60):
        """Index live camera frames into buffer."""
        from .utils.camera import capture_frames
        print(f"[SightRAG] Indexing camera {camera_id}...")
        print(f"[SightRAG] Press Ctrl+C to stop.")

        count = 0
        try:
            for image, timestamp in capture_frames(
                camera_id, fps, buffer_seconds
            ):
                regions = self.detector.detect(image)
                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    id = f"cam_{camera_id}_{timestamp}_{j}"
                    self.store.add(id, embedding, {
                        "image_path":  f"camera_{camera_id}",
                        "bbox":        region["bbox"],
                        "label":       region["label"],
                        "confidence":  region["confidence"],
                        "timestamp":   timestamp,
                        "source_type": "camera"
                    })
                count += 1
                print(f"\r[SightRAG] {count} frames indexed | "
                      f"Last: {timestamp}", end="")

        except KeyboardInterrupt:
            print(f"\n[SightRAG] Camera stopped. "
                  f"{self.store.count()} regions indexed.")
