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
    
    def __init__(self, detector, embedder, store):
        self.detector = detector
        self.embedder = embedder
        self.store = store
    
    def _index_image(self, path_str, image, prefix):
        count = 0
        regions = self.detector.detect(image)
        for j, region in enumerate(regions):
            embedding = self.embedder.embed_image(region["crop"])
            if not np.allclose(embedding, 0):
                self.store.add(f"{prefix}_{j}", embedding, {
                    "image_path": path_str,
                    "bbox": region["bbox"],
                    "label": region["label"],
                    "confidence": region["confidence"],
                    "source_type": "image"
                })
                count += 1
        return count
    
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
        
        mode = "C++" if FAST_MODE else "Python"
        print(f"[SightRAG] Found {len(image_paths)} images, {len(video_paths)} videos ({mode} mode)")
        
        # Index images
        if image_paths:
            total = len(image_paths)
            for i, path in enumerate(image_paths, 1):
                try:
                    image = load_image(str(path))
                    self._index_image(str(path), image, path.stem)
                    pct = int((i / total) * 40)
                    bar = "█" * pct + "░" * (40 - pct)
                    print(f"\r  [{bar}] {i}/{total} images", end="", flush=True)
                except Exception as e:
                    print(f"\n  Skipping {path.name}: {e}")
            print()
        
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
        
        for i, (image, timestamp) in enumerate(frames, 1):
            try:
                regions = self.detector.detect(image)
                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    if not np.allclose(embedding, 0):
                        self.store.add(f"{video_name}_f{i}_r{j}", embedding, {
                            "image_path": video_path,
                            "bbox": region["bbox"],
                            "label": region["label"],
                            "confidence": region["confidence"],
                            "timestamp": timestamp,
                            "source_type": "video"
                        })
                pct = int((i / total) * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {i}/{total} frames", end="", flush=True)
            except Exception:
                pass
        print()
    
    def index_camera(self, camera_id=0, fps=1, buffer_seconds=60):
        from .utils.camera import capture_frames
        print(f"[SightRAG] Camera {camera_id}. Press Ctrl+C to stop.")
        count = 0
        try:
            for image, timestamp in capture_frames(camera_id, fps, buffer_seconds):
                regions = self.detector.detect(image)
                for j, region in enumerate(regions):
                    embedding = self.embedder.embed_image(region["crop"])
                    if not np.allclose(embedding, 0):
                        self.store.add(f"cam{camera_id}_{timestamp}_{j}", embedding, {
                            "image_path": f"camera_{camera_id}",
                            "bbox": region["bbox"],
                            "label": region["label"],
                            "confidence": region["confidence"],
                            "timestamp": timestamp,
                            "source_type": "camera"
                        })
                count += 1
                print(f"\r[SightRAG] {count} frames | {timestamp}", end="", flush=True)
        except KeyboardInterrupt:
            print(f"\n[SightRAG] Stopped. {self.store.count()} regions indexed.")
