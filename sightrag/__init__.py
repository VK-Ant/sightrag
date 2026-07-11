"""
SightRAG — Image and Video RAG
See. Search. Retrieve.

Usage:
    from sightrag import SightRAG
    
    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")
    rag.show(results)

Custom models:
    from sightrag import SightRAG
    from sightrag.detectors.base import DetectorBase
    from sightrag.embedders.base import EmbedderBase
    
    rag = SightRAG(detector=MyDetector(), embedder=MyEmbedder())

Author: Ant (VK-Ant)
License: Apache 2.0
GitHub: https://github.com/VK-Ant/sightrag
"""

from .core import SightRAG

__version__ = "0.2.0"
__author__ = "Ant (VK-Ant)"
__license__ = "Apache-2.0"

# Public API
from .detectors.base import DetectorBase
from .embedders.base import EmbedderBase
from .store.base import VectorStoreBase

def serve(host="0.0.0.0", port=8000):
    """Start REST API server."""
    from .api import app
    import uvicorn
    uvicorn.run(app, host=host, port=port)
