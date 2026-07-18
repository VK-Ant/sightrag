"""
SightRAG — Image and Video RAG
See. Search. Retrieve.

Usage:
    from sightrag import SightRAG
    
    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")
    rag.show(results)

v0.3 new features:
    # Grounding DINO — any domain, no training
    rag = SightRAG(detector="grounding-dino")
    
    # Person Re-ID — track across cameras
    rag = SightRAG(embedder="reid")
    
    # Re-ranking — better results on large datasets
    rag = SightRAG(rerank=True)
    
    # CLI
    $ sightrag index ./photos/
    $ sightrag query "find person"
"""

from .core import SightRAG

__version__ = "0.3.0"
__author__ = "Ant (VK-Ant)"
__license__ = "Apache-2.0"

from .detectors.base import DetectorBase
from .embedders.base import EmbedderBase
from .store.base import VectorStoreBase

def serve(host="0.0.0.0", port=8000):
    from .api import app
    import uvicorn
    uvicorn.run(app, host=host, port=port)
