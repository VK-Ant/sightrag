"""
SightRAG v0.4 — See. Search. Retrieve. Understand.

Basic:
    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")

With OCR (reads text on images):
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
                 api_key=None):
        
        self.domain_hint = domain_hint
        self._store_type = store if isinstance(store, str) else "custom"
        self._index_path = index_path or os.path.join(SIGHTRAG_HOME, "index")
        self._rerank = rerank
        self._reranker = None
        self._ocr = None
        self._multimodal = None
        
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
            raise TypeError("detector must be string or DetectorBase")
        
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
            raise TypeError("embedder must be string or EmbedderBase")
        
        # OCR
        if ocr:
            from .ocr import OCREngine
            self._ocr = OCREngine()
            print(f"[SightRAG] OCR: {self._ocr.engine_name}")
        
        # Multimodal
        if multimodal:
            from .multimodal import MultimodalEngine
            self._multimodal = MultimodalEngine(model=multimodal, api_key=api_key)
            print(f"[SightRAG] Multimodal: {self._multimodal.model_name}")
        
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
        
        self._indexer = Indexer(
            self._detector, self._embedder, self._store, self._ocr
        )
        self._retriever = Retriever(
            self._embedder, self._detector, self._store, domain_hint
        )
        
        print("[SightRAG] Ready.")
    
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
        """Index images, video, or camera."""
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
    
    def _index_single_image(self, path):
        from .utils.image import load_image
        image = load_image(path)
        regions = self._detector.detect(image)
        for j, region in enumerate(regions):
            embedding = self._embedder.embed_image(region["crop"])
            if not np.allclose(embedding, 0):
                metadata = {
                    "image_path": str(path),
                    "bbox": region["bbox"],
                    "label": region["label"],
                    "confidence": region["confidence"],
                    "source_type": "image",
                    "ocr_text": ""
                }
                # OCR at index time
                if self._ocr:
                    ocr_result = self._ocr.read(region["crop"])
                    metadata["ocr_text"] = ocr_result["text"]
                
                self._store.add(f"img_{j}", embedding, metadata)
        print(f"[SightRAG] 1 image indexed. Total: {self.count()} regions.")
    
    def query(self, text=None, reference=None, top_k=5, understand=False):
        """
        Search indexed content.
        
        rag.query("find person")                          # fast visual
        rag.query("find Calgon")                          # matches OCR text
        rag.query("find damaged product", understand=True) # LLM understanding
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
        
        return results[:top_k]
    
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
                # Boost score based on OCR match
                boost = len(matches) / len(query_words) * 0.3
                r["score"] = min(1.0, r.get("score", 0) + boost)
                r["ocr_match"] = True
        
        # Re-sort by boosted score
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
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
        features = [f"backend='{self._backend.name}'"]
        features.append(f"indexed={self.count()}")
        if self._ocr:
            features.append("ocr=True")
        if self._multimodal:
            features.append(f"multimodal='{self._multimodal.model_name}'")
        return f"SightRAG({', '.join(features)})"
