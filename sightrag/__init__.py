"""
SightRAG — Image and Video RAG
See. Search. Retrieve. Track.

Usage:
    from sightrag import SightRAG

    rag = SightRAG()
    rag.index("./photos/")
    results = rag.query("find person")
    rag.show(results)

v0.5 new features:
    # Segmentation — pixel-level masks
    rag = SightRAG(segment=True)
    results = rag.query("find person")  # results include masks

    # Object Tracking — works on images, video, streams, cameras
    rag = SightRAG(track=True)
    rag.index("./footage.mp4", fps=5)      # video
    rag.index("./photos/")                  # image folder (frames)
    results = rag.query("find person in red shirt")
    timeline = rag.timeline(results[0])

    # Find + Track — one-step query and follow
    timeline = rag.find_and_track("person in red shirt")

    # Custom segmentor / tracker
    rag = SightRAG(segmentor="sam2")
    rag = SightRAG(tracker="botsort")

    # CLI
    $ sightrag index ./video.mp4 --fps 5 --track
    $ sightrag find-track "person in red"
    $ sightrag tracks
    $ sightrag timeline 3
"""

from .core import SightRAG

__version__ = "0.5.0"
__author__ = "Ant (VK-Ant)"
__license__ = "Apache-2.0"

from .detectors.base import DetectorBase
from .embedders.base import EmbedderBase
from .store.base import VectorStoreBase
from .segmentors.base import SegmentorBase
from .trackers.base import TrackerBase

def serve(host="0.0.0.0", port=8000):
    from .api import app
    import uvicorn
    uvicorn.run(app, host=host, port=port)
