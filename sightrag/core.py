# sightrag/core.py
# Main SightRAG class — user only touches this

from .detector import Detector
from .embedder import Embedder
from .indexer import Indexer
from .retriever import Retriever


class SightRAG:
    """
    SightRAG — Image and Video RAG.
    See. Search. Retrieve.

    Basic usage:
        rag = SightRAG()
        rag.index("./photos/")
        results = rag.query("find empty shelf")

    With domain hint (custom domains):
        rag = SightRAG(domain_hint="pcb defect solder joint")
        rag.index("./circuit_boards/")

    With SQLite (lightweight fallback):
        rag = SightRAG(store="sqlite")
    """

    def __init__(self,
                 store: str = "chroma",
                 domain_hint: str = None,
                 index_path: str = "./sightrag_index"):

        self.domain_hint = domain_hint
        self._store_type = store
        self._index_path = index_path

        # Load models
        print("[SightRAG] Initializing...")
        self._detector = Detector()
        self._embedder = Embedder()
        self._store = self._init_store(store, index_path)
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
                print("[SightRAG] ChromaDB not found. Falling back to SQLite.")
                from .store.sqlite_store import SQLiteStore
                return SQLiteStore(path)
        elif store_type == "sqlite":
            from .store.sqlite_store import SQLiteStore
            return SQLiteStore(path)
        else:
            raise ValueError(
                f"Unknown store: {store_type}\n"
                f"Options: 'chroma' (default), 'sqlite'"
            )

    def index(self, path: str = None,
              source: str = None,
              camera_id: int = 0,
              fps: int = 1):
        """
        Index images, video, or camera.

        Image folder:
            rag.index("./photos/")

        Video file:
            rag.index("./video.mp4")

        Live camera:
            rag.index(source="camera")
            rag.index(source="camera", camera_id=1)
        """
        # Camera mode
        if source == "camera":
            self._indexer.index_camera(
                camera_id=camera_id, fps=fps
            )
            return self

        if path is None:
            raise ValueError(
                "Provide a path or source='camera'\n"
                "Example: rag.index('./photos/')\n"
                "Example: rag.index(source='camera')"
            )

        import os
        if os.path.isdir(path):
            self._indexer.index_folder(path)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv"}:
                self._indexer.index_video(path, fps=fps)
            else:
                # Single image
                from .utils.image import load_image
                image = load_image(path)
                regions = self._detector.detect(image)
                for j, region in enumerate(regions):
                    emb = self._embedder.embed_image(region["crop"])
                    self._store.add(f"img_{j}", emb, {
                        "image_path":  path,
                        "bbox":        region["bbox"],
                        "label":       region["label"],
                        "confidence":  region["confidence"],
                        "source_type": "image"
                    })
                print(f"[SightRAG] 1 image indexed.")
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        return self

    def query(self, text: str = None,
              reference: str = None,
              top_k: int = 5):
        """
        Search indexed content.

        Text query:
            results = rag.query("find empty shelf")

        Reference image:
            results = rag.query(reference="sample.jpg")

        Both (text takes priority):
            results = rag.query("empty shelf",
                                reference="sample.jpg")
        """
        if text is None and reference is None:
            raise ValueError(
                "Provide text or reference image.\n"
                "Example: rag.query('find empty shelf')\n"
                "Example: rag.query(reference='image.jpg')"
            )

        if text:
            return self._retriever.query_text(text, top_k)
        else:
            return self._retriever.query_reference(
                reference, top_k
            )

    def count(self) -> int:
        """Return number of indexed regions."""
        return self._store.count()

    def clear(self):
        """Clear all indexed data."""
        self._store.clear()
        print("[SightRAG] Index cleared.")
        return self

    def __repr__(self):
        return (
            f"SightRAG("
            f"store={self._store_type}, "
            f"indexed={self.count()} regions, "
            f"domain_hint={self.domain_hint})"
        )
