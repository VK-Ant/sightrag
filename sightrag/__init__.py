"""
SightRAG — Image and Video RAG
See. Search. Retrieve.

pip install sightrag

Usage:
    from sightrag import SightRAG

    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find empty shelf")

REST API:
    from sightrag import serve
    serve(port=8000)
"""

from .core import SightRAG

__version__ = "0.1.0"
__author__ = "Ant (VK-Ant)"


def serve(host: str = "0.0.0.0", port: int = 8000):
    """Start SightRAG REST API server."""
    from .api import serve as _serve
    _serve(host=host, port=port)


__all__ = ["SightRAG", "serve"]
