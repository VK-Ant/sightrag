# sightrag/core.py
# Main SightRAG class
# All data stored in ~/.sightrag/ — project folder stays clean

import os
from pathlib import Path
from .detector import Detector
from .embedder import Embedder
from .indexer import Indexer
from .retriever import Retriever

SIGHTRAG_HOME = os.path.join(Path.home(), ".sightrag")


class SightRAG:
    """
    SightRAG — Image and Video RAG.
    See. Search. Retrieve.

    Usage:
        rag = SightRAG()
        rag.index("./photos/")
        results = rag.query("find empty shelf")
    """

    def __init__(self,
                 store: str = "sqlite",
                 domain_hint: str = None,
                 index_path: str = None):

        self.domain_hint = domain_hint
        self._store_type = store

        if index_path is None:
            self._index_path = os.path.join(SIGHTRAG_HOME, "index")
        else:
            self._index_path = index_path

        os.makedirs(SIGHTRAG_HOME, exist_ok=True)

        print("[SightRAG] Initializing...")
        self._detector = Detector()
        self._embedder = Embedder()
        self._store = self._init_store(store, self._index_path)
        self._indexer = Indexer(
            self._detector, self._embedder,
            self._store, domain_hint
        )
        self._retriever = Retriever(
            self._embedder, self._detector,
            self._store, domain_hint
        )
        print("[SightRAG] Ready.")

    def _init_store(self, store_type: str, path: str):
        if store_type == "chroma":
            try:
                from .store.chroma_store import ChromaStore
                return ChromaStore(path)
            except ImportError:
                print("[SightRAG] ChromaDB not found. Using SQLite.")
                from .store.sqlite_store import SQLiteStore
                return SQLiteStore(path)
        elif store_type == "sqlite":
            from .store.sqlite_store import SQLiteStore
            return SQLiteStore(path)
        else:
            raise ValueError(f"Unknown store: {store_type}. Use 'chroma' or 'sqlite'.")

    def index(self, path: str = None, source: str = None,
              camera_id: int = 0, fps: int = 1):
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
                from .utils.image import load_image
                image = load_image(path)
                self._index_single_image(path, image)
                print(f"[SightRAG] 1 image indexed. Total: {self.count()} regions.")
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        return self

    def _index_single_image(self, path, image):
        """Index one image with detection + embedding."""
        import numpy as np
        regions = self._detector.detect(image)
        for j, region in enumerate(regions):
            embedding = self._embedder.embed_image(region["crop"])
            if not np.allclose(embedding, 0):
                self._store.add(f"img_{j}", embedding, {
                    "image_path":  str(path),
                    "bbox":        region["bbox"],
                    "label":       region["label"],
                    "confidence":  region["confidence"],
                    "source_type": "image"
                })

    def query(self, text: str = None, reference: str = None, top_k: int = 5):
        """Search indexed content with text or reference image."""
        if text is None and reference is None:
            raise ValueError("Provide text or reference image.")
        if text:
            return self._retriever.query_text(text, top_k)
        else:
            return self._retriever.query_reference(reference, top_k)

    def count(self) -> int:
        return self._store.count()

    def clear(self):
        self._store.clear()
        print("[SightRAG] Index cleared.")
        return self

    def __repr__(self):
        return f"SightRAG(store='{self._store_type}', indexed={self.count()} regions)"
