"""
SightRAG v0.3 — See. Search. Retrieve.

Usage:
    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")
    rag.show(results)

Custom models:
    rag = SightRAG(detector=MyDetector(), embedder=MyEmbedder())

Grounding DINO (any domain):
    rag = SightRAG(detector="grounding-dino")
    results = rag.query("find cracked solder joint")

Person Re-ID:
    rag = SightRAG(embedder="reid")
    results = rag.query(reference="./suspect.jpg")

Re-ranking:
    rag = SightRAG(rerank=True)
    results = rag.query("find person", top_k=5)
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
                 rerank=False):
        
        self.domain_hint = domain_hint
        self._store_type = store if isinstance(store, str) else "custom"
        self._index_path = index_path or os.path.join(SIGHTRAG_HOME, "index")
        self._rerank = rerank
        self._reranker = None
        
        os.makedirs(SIGHTRAG_HOME, exist_ok=True)
        
        print("[SightRAG] Initializing...")
        
        # Backend
        self._backend = auto_select_backend()
        print(f"[SightRAG] Backend: {self._backend.name}")
        
        # Detector
        if detector is None:
            self._detector = self._backend
            print("[SightRAG] Detector: default (YOLO)")
        elif isinstance(detector, str):
            self._detector = self._load_detector(detector)
            print(f"[SightRAG] Detector: {detector}")
        elif isinstance(detector, DetectorBase):
            self._detector = detector
            print(f"[SightRAG] Detector: custom ({type(detector).__name__})")
        else:
            raise TypeError("detector must be string or DetectorBase subclass")
        
        # Embedder
        if embedder is None:
            self._embedder = self._backend
            print("[SightRAG] Embedder: default (CLIP)")
        elif isinstance(embedder, str):
            self._embedder = self._load_embedder(embedder)
            print(f"[SightRAG] Embedder: {embedder}")
        elif isinstance(embedder, EmbedderBase):
            self._embedder = embedder
            print(f"[SightRAG] Embedder: custom ({type(embedder).__name__})")
        else:
            raise TypeError("embedder must be string or EmbedderBase subclass")
        
        # Re-ranker
        if rerank:
            from .reranker import ReRanker
            self._reranker = ReRanker()
            print("[SightRAG] Re-ranker: enabled")
        
        # Store
        self._store = self._init_store(store, self._index_path)
        
        # Indexer + Retriever
        from .indexer import Indexer
        from .retriever import Retriever
        
        self._indexer = Indexer(self._detector, self._embedder, self._store)
        self._retriever = Retriever(self._embedder, self._detector, self._store, domain_hint)
        
        print("[SightRAG] Ready.")
    
    def _load_detector(self, name):
        if name in ("grounding-dino", "grounding_dino", "gdino"):
            from .detectors.grounding_dino import GroundingDINODetector
            return GroundingDINODetector(text_prompt=self.domain_hint)
        elif name in ("yolo", "yolo11"):
            return self._backend
        else:
            raise ValueError(f"Unknown detector: {name}. Options: 'yolo', 'grounding-dino'")
    
    def _load_embedder(self, name):
        if name in ("reid", "re-id", "person-reid"):
            from .embedders.reid_embedder import ReIDEmbedder
            return ReIDEmbedder()
        elif name in ("clip", "clip-vit"):
            return self._backend
        else:
            raise ValueError(f"Unknown embedder: {name}. Options: 'clip', 'reid'")
    
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
            elif store_type == "qdrant":
                from .store.qdrant_store import QdrantStore
                return QdrantStore()
            else:
                raise ValueError(f"Unknown store: {store_type}. Options: 'sqlite', 'chroma', 'qdrant'")
        else:
            return store_type
    
    def index(self, path=None, source=None, camera_id=0, fps=1):
        """Index images, video, or camera."""
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
        """Search indexed content."""
        if text is None and reference is None:
            raise ValueError("Provide text or reference image.")
        
        # If re-ranking, fetch more candidates first
        fetch_k = top_k * 20 if self._reranker and text else top_k
        
        if text:
            results = self._retriever.query_text(text, fetch_k)
        else:
            results = self._retriever.query_reference(reference, fetch_k)
        
        # Re-rank if enabled and text query
        if self._reranker and text and len(results) > top_k:
            results = self._reranker.rerank(text, results, top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def show(self, results, save=None, max_show=5):
        """Visualize results with bounding boxes."""
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
