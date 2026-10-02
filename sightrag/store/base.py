# sightrag/store/base.py
# Abstract base - every store implements this

class VectorStoreBase:
    def add(self, id: str, embedding, metadata: dict):
        raise NotImplementedError

    def search(self, query_vector, top_k: int = 5):
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def delete(self, id: str):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    # v0.5 track methods — no-ops by default for stores that don't support tracks
    def save_track(self, **kwargs):
        pass

    def add_track_detection(self, **kwargs):
        pass

    def get_track(self, track_id):
        return None

    def get_all_tracks(self):
        return []

    def find_track_for_detection(self, image_path, bbox, timestamp):
        return None

    def track_count(self):
        return 0
