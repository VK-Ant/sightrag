# sightrag/store/chroma_store.py
# ChromaDB vector store — default for SightRAG

from .base import VectorStoreBase


class ChromaStore(VectorStoreBase):
    """
    ChromaDB vector store (default).
    """

    def __init__(self, path: str = "./sightrag_index"):
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
                "Or use: SightRAG(store='sqlite')"
            )

    def add(self, id: str, embedding, metadata: dict = {}):
        try:
            # Convert numpy to plain Python list of native floats
            emb_list = [float(x) for x in embedding.flatten()]

            # ChromaDB metadata ONLY accepts str, int, float, bool
            clean_meta = {
                "image_path":  str(metadata.get("image_path", "")),
                "label":       str(metadata.get("label", "")),
                "timestamp":   str(metadata.get("timestamp", "")),
                "confidence":  float(metadata.get("confidence", 0.0)),
                "source_type": str(metadata.get("source_type", "image")),
                "bbox":        str(metadata.get("bbox", "[]")),
            }

            self.collection.upsert(
                ids=[str(id)],
                embeddings=[emb_list],
                metadatas=[clean_meta]
            )
        except Exception as e:
            # Show SHORT error, not full embedding dump
            err_msg = str(e)[:200] if len(str(e)) > 200 else str(e)
            print(f"[SightRAG] Store error: {err_msg}")

    def search(self, query_vector, top_k: int = 5):
        try:
            count = self.collection.count()
            if count == 0:
                return []

            actual_k = min(top_k, count)
            emb_list = [float(x) for x in query_vector.flatten()]

            results = self.collection.query(
                query_embeddings=[emb_list],
                n_results=actual_k
            )

            output = []
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 1.0

                    bbox = meta.get("bbox", "[]")
                    if isinstance(bbox, str):
                        try:
                            import ast
                            bbox = ast.literal_eval(bbox)
                        except:
                            bbox = []

                    output.append({
                        "score":       round(1.0 - distance, 4),
                        "image_path":  meta.get("image_path", ""),
                        "label":       meta.get("label", ""),
                        "timestamp":   meta.get("timestamp", ""),
                        "confidence":  float(meta.get("confidence", 0.0)),
                        "source_type": meta.get("source_type", "image"),
                        "bbox":        bbox,
                        "metadata":    meta
                    })
            return output
        except Exception as e:
            print(f"[SightRAG] Search error: {str(e)[:200]}")
            return []

    def count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0

    def delete(self, id: str):
        try:
            self.collection.delete(ids=[str(id)])
        except:
            pass

    def clear(self):
        try:
            self.client.delete_collection("sightrag")
            self.collection = self.client.get_or_create_collection(
                name="sightrag",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            pass
