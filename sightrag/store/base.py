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
