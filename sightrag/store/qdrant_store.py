"""
Qdrant vector store — large scale (1M+ images).
Install: pip install sightrag[qdrant]
"""

import numpy as np
from .base import VectorStoreBase


class QdrantStore(VectorStoreBase):
    """
    Qdrant vector database — production-grade, scalable.
    
    Usage:
        # Local Qdrant
        rag = SightRAG(store=QdrantStore())
        
        # Remote Qdrant
        rag = SightRAG(store=QdrantStore(url="http://qdrant-server:6333"))
    """
    
    def __init__(self, url=None, collection_name="sightrag", embed_dim=512):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise ImportError(
                "Qdrant requires: pip install qdrant-client\n"
                "Or: pip install sightrag[qdrant]"
            )
        
        self._PointStruct = PointStruct
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path="./.sightrag_qdrant")
        
        # Create collection if not exists
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embed_dim,
                    distance=Distance.COSINE
                )
            )
        
        self._id_counter = self.count()
    
    def add(self, id, embedding, metadata={}):
        emb = np.array(embedding, dtype=np.float32).flatten().tolist()
        
        # Ensure dimension matches
        if len(emb) != self.embed_dim:
            self.embed_dim = len(emb)
            # Recreate collection with correct dimension
            from qdrant_client.models import Distance, VectorParams
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE)
            )
        
        self._id_counter += 1
        
        # Clean metadata for Qdrant
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = str(v)
            else:
                clean_meta[k] = str(v)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[self._PointStruct(
                id=self._id_counter,
                vector=emb,
                payload=clean_meta
            )]
        )
    
    def search(self, query_vector, top_k=5):
        emb = np.array(query_vector, dtype=np.float32).flatten().tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=emb,
            limit=top_k
        )
        
        output = []
        for r in results:
            payload = r.payload or {}
            bbox = payload.get("bbox", "[]")
            if isinstance(bbox, str):
                try:
                    import ast
                    bbox = ast.literal_eval(bbox)
                except:
                    bbox = []
            
            output.append({
                "score": round(float(r.score), 4),
                "image_path": payload.get("image_path", ""),
                "bbox": bbox,
                "timestamp": payload.get("timestamp", ""),
                "confidence": float(payload.get("confidence", 0.0)),
                "label": payload.get("label", ""),
                "source_type": payload.get("source_type", "image"),
            })
        
        return output
    
    def count(self):
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except:
            return 0
    
    def delete(self, id):
        pass  # Qdrant uses integer IDs internally
    
    def clear(self):
        from qdrant_client.models import Distance, VectorParams
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE)
        )
        self._id_counter = 0
