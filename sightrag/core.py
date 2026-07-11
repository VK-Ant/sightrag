"""
SightRAG v0.2 — See. Search. Retrieve.

Usage:
    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")
    rag.show(results)

Custom models:
    rag = SightRAG(detector=MyDetector(), embedder=MyEmbedder())

All data stored in ~/.sightrag/ — project folder stays clean.
Backend auto-selected: TensorRT > ONNX > OpenVINO > PyTorch.
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
                 index_path=None):
        
        self.domain_hint = domain_hint
        self._store_type = store
        self._index_path = index_path or os.path.join(SIGHTRAG_HOME, "index")
        
        os.makedirs(SIGHTRAG_HOME, exist_ok=True)
        
        print("[SightRAG] Initializing...")
        
        # Auto-select backend (fastest available)
        self._backend = auto_select_backend()
        print(f"[SightRAG] Backend: {self._backend.name}")
        
        # Detector: custom or backend default
        if detector is not None:
            if isinstance(detector, DetectorBase):
                self._detector = detector
                print(f"[SightRAG] Detector: custom ({type(detector).__name__})")
            else:
                raise TypeError("detector must be DetectorBase subclass")
        else:
            self._detector = self._backend
            print("[SightRAG] Detector: default (YOLO)")
        
        # Embedder: custom or backend default
        if embedder is not None:
            if isinstance(embedder, EmbedderBase):
                self._embedder = embedder
                print(f"[SightRAG] Embedder: custom ({type(embedder).__name__})")
            else:
                raise TypeError("embedder must be EmbedderBase subclass")
        else:
            self._embedder = self._backend
            print("[SightRAG] Embedder: default (CLIP)")
        
        # Store
        self._store = self._init_store(store, self._index_path)
        
        # Indexer + Retriever
        from .indexer import Indexer
        from .retriever import Retriever
        
        self._indexer = Indexer(self._detector, self._embedder, self._store)
        self._retriever = Retriever(self._embedder, self._detector, self._store, domain_hint)
        
        print("[SightRAG] Ready.")
    
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
                    print("[SightRAG] ChromaDB not found. Using SQLite.")
                    from .store.sqlite_store import SQLiteStore
                    return SQLiteStore(path)
            else:
                raise ValueError(f"Unknown store: {store_type}. Use 'sqlite' or 'chroma'.")
        else:
            # Custom store object
            return store_type
    
    def index(self, path=None, source=None, camera_id=0, fps=1):
        """
        Index images, video, or camera.
        
        rag.index("./photos/")          # folder
        rag.index("./video.mp4")        # video
        rag.index(source="camera")      # webcam
        """
        if source == "camera":
            self._indexer.index_camera(camera_id=camera_id, fps=fps)
            return self
        
        if path is None:
            raise ValueError("Provide a path or source='camera'")
        
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
    
    def _index_single_image(self, path):
        from .utils.image import load_image
        image = load_image(path)
        regions = self._detector.detect(image)
        for j, region in enumerate(regions):
            embedding = self._embedder.embed_image(region["crop"])
            if not np.allclose(embedding, 0):
                self._store.add(f"img_{j}", embedding, {
                    "image_path": str(path),
                    "bbox": region["bbox"],
                    "label": region["label"],
                    "confidence": region["confidence"],
                    "source_type": "image"
                })
        print(f"[SightRAG] 1 image indexed. Total: {self.count()} regions.")
    
    def query(self, text=None, reference=None, top_k=5):
        """
        Search indexed content.
        
        results = rag.query("find person")
        results = rag.query(reference="sample.jpg")
        """
        if text is None and reference is None:
            raise ValueError("Provide text or reference image.")
        if text:
            return self._retriever.query_text(text, top_k)
        else:
            return self._retriever.query_reference(reference, top_k)
    
    def show(self, results, save=None, max_show=5):
        """
        Visualize query results with bounding boxes.
        
        rag.show(results)                    # display on screen
        rag.show(results, save="./output/")  # save annotated images
        """
        from .visualizer import show_results
        show_results(results, save=save, max_show=max_show)
    
    def count(self):
        return self._store.count()
    
    def clear(self):
        self._store.clear()
        print("[SightRAG] Index cleared.")
        return self
    
    def __repr__(self):
        return (f"SightRAG(backend='{self._backend.name}', "
                f"indexed={self.count()} regions)")
