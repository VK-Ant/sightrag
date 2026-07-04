"""
SightRAG Unit Tests
Run: python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSQLiteStore:
    """Test built-in SQLite vector store."""

    def setup_method(self):
        from sightrag.store.sqlite_store import SQLiteStore
        self.store = SQLiteStore(path="/tmp/test_sightrag_store")

    def teardown_method(self):
        import shutil
        shutil.rmtree("/tmp/test_sightrag_store", ignore_errors=True)

    def test_add_and_count(self):
        vec = np.random.randn(512).astype(np.float32)
        self.store.add("test_1", vec, {"image_path": "test.jpg", "label": "bottle"})
        assert self.store.count() == 1

    def test_search(self):
        for i in range(5):
            vec = np.random.randn(512).astype(np.float32)
            self.store.add(f"test_{i}", vec, {"image_path": f"img_{i}.jpg"})
        query = np.random.randn(512).astype(np.float32)
        results = self.store.search(query, top_k=3)
        assert len(results) == 3
        assert "score" in results[0]
        assert "image_path" in results[0]

    def test_delete(self):
        vec = np.random.randn(512).astype(np.float32)
        self.store.add("del_test", vec, {})
        assert self.store.count() == 1
        self.store.delete("del_test")
        assert self.store.count() == 0

    def test_clear(self):
        for i in range(10):
            vec = np.random.randn(512).astype(np.float32)
            self.store.add(f"clear_{i}", vec, {})
        assert self.store.count() == 10
        self.store.clear()
        assert self.store.count() == 0

    def test_empty_search(self):
        query = np.random.randn(512).astype(np.float32)
        results = self.store.search(query, top_k=5)
        assert len(results) == 0


class TestImageUtils:
    """Test image loading utilities."""

    def test_supported_formats(self):
        from sightrag.utils.image import SUPPORTED_FORMATS
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS

    def test_missing_folder(self):
        from sightrag.utils.image import get_image_paths
        with pytest.raises(FileNotFoundError):
            get_image_paths("/nonexistent/folder")

    def test_missing_image(self):
        from sightrag.utils.image import load_image
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")


class TestVideoUtils:
    """Test video utilities."""

    def test_supported_formats(self):
        from sightrag.utils.video import SUPPORTED_FORMATS
        assert ".mp4" in SUPPORTED_FORMATS
        assert ".avi" in SUPPORTED_FORMATS
        assert ".mov" in SUPPORTED_FORMATS

    def test_missing_video(self):
        from sightrag.utils.video import extract_frames
        with pytest.raises(FileNotFoundError):
            extract_frames("/nonexistent/video.mp4")


class TestImports:
    """Test import chain."""

    def test_main_import(self):
        from sightrag import SightRAG
        assert SightRAG is not None

    def test_version(self):
        from sightrag import __version__
        assert __version__ == "0.1.0"

    def test_author(self):
        from sightrag import __author__
        assert "Ant" in __author__

    def test_serve_import(self):
        from sightrag import serve
        assert callable(serve)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
