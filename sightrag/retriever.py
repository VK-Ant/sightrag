# sightrag/retriever.py
# Handles text and reference image queries

from PIL import Image
from .embedder import Embedder
from .detector import Detector


class Retriever:
    """
    Retrieves matching results from vector store.
    Supports text queries and reference image queries.
    """

    def __init__(self, embedder: Embedder,
                 detector: Detector,
                 store,
                 domain_hint: str = None):
        self.embedder = embedder
        self.detector = detector
        self.store = store
        self.domain_hint = domain_hint

    def query_text(self, text: str, top_k: int = 5):
        """Search using plain English text."""
        if self.store.count() == 0:
            raise RuntimeError(
                "Nothing indexed yet.\n"
                "Run rag.index() first."
            )

        embedding = self.embedder.embed_text(
            text, self.domain_hint
        )
        results = self.store.search(embedding, top_k)
        return self._format(results)

    def query_reference(self, image_path: str, top_k: int = 5):
        """Search using a reference image."""
        if self.store.count() == 0:
            raise RuntimeError(
                "Nothing indexed yet.\n"
                "Run rag.index() first."
            )

        image = Image.open(image_path).convert("RGB")
        regions = self.detector.detect(image)

        if not regions:
            raise ValueError(
                "No regions detected in reference image.\n"
                "Try a clearer image with distinct objects."
            )

        # Use first/best detected region as query
        best = max(regions, key=lambda r: r["confidence"])
        embedding = self.embedder.embed_image(best["crop"])
        results = self.store.search(embedding, top_k)
        return self._format(results)

    def _format(self, results):
        """Format results for clean output."""
        formatted = []
        for r in results:
            formatted.append({
                "image_path":  r.get("image_path", ""),
                "score":       round(r.get("score", 0.0), 4),
                "label":       r.get("label", ""),
                "confidence":  round(r.get("confidence", 0.0), 4),
                "bbox":        r.get("bbox", []),
                "timestamp":   r.get("timestamp", ""),
                "source_type": r.get("source_type", "image")
            })
        return formatted
