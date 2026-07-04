# sightrag/store/chroma_store.py
# Optional ChromaDB store — for larger datasets
# pip install chromadb to use

from .base import VectorStoreBase


class ChromaStore(VectorStoreBase):
    """
    ChromaDB vector store.
    Good for 100k+ images.
    pip install chromadb
    """

    def __init__(self, path: str = "./sightrag_chroma"):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(
                name="sightrag",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError(
                "ChromaDB not installed.\n"
                "Run: pip install chromadb\n"
                "Or use default: SightRAG(store='sqlite')"
            )

    def add(self, id: str, embedding, metadata: dict = {}):
        self.collection.upsert(
            ids=[id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "image_path":  metadata.get("image_path", ""),
                "label":       metadata.get("label", ""),
                "timestamp":   metadata.get("timestamp", ""),
                "confidence":  str(metadata.get("confidence", 0.0)),
                "source_type": metadata.get("source_type", "image"),
            }]
        )

    def search(self, query_vector, top_k: int = 5):
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )

        output = []
        for i, id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            output.append({
                "score":       1 - results["distances"][0][i],
                "image_path":  meta.get("image_path", ""),
                "label":       meta.get("label", ""),
                "timestamp":   meta.get("timestamp", ""),
                "confidence":  float(meta.get("confidence", 0.0)),
                "source_type": meta.get("source_type", "image"),
                "bbox":        [],
                "metadata":    meta
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def delete(self, id: str):
        self.collection.delete(ids=[id])

    def clear(self):
        self.collection.delete(
            where={"source_type": {"$ne": ""}}
        )
