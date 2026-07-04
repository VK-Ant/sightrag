# sightrag/store/sqlite_store.py
# Built-in vector store — zero extra dependencies

import sqlite3
import pickle
import json
import numpy as np
from pathlib import Path
from .base import VectorStoreBase


class SQLiteStore(VectorStoreBase):
    """Default vector store. No extra dependencies. Works everywhere."""

    def __init__(self, path: str = "./sightrag_index"):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.db_path = f"{path}/sightrag.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._setup()

    def _setup(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id          TEXT PRIMARY KEY,
                embedding   BLOB NOT NULL,
                image_path  TEXT,
                bbox        TEXT,
                timestamp   TEXT,
                confidence  REAL,
                label       TEXT,
                source_type TEXT,
                metadata    TEXT
            )
        """)
        self.conn.commit()

    def add(self, id: str, embedding, metadata: dict = {}):
        emb = np.array(embedding, dtype=np.float32).flatten()
        self.conn.execute("""
            INSERT OR REPLACE INTO vectors 
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            str(id),
            pickle.dumps(emb),
            str(metadata.get("image_path", "")),
            json.dumps(metadata.get("bbox", [])),
            str(metadata.get("timestamp", "")),
            float(metadata.get("confidence", 0.0)),
            str(metadata.get("label", "")),
            str(metadata.get("source_type", "image")),
            json.dumps(metadata)
        ))
        self.conn.commit()

    def search(self, query_vector, top_k: int = 5):
        rows = self.conn.execute("""
            SELECT id, embedding, image_path,
                   bbox, timestamp, confidence,
                   label, source_type, metadata
            FROM vectors
        """).fetchall()

        if not rows:
            return []

        q = np.array(query_vector, dtype=np.float32).flatten()
        q_dim = len(q)

        # Load and filter vectors
        valid = []
        for row in rows:
            try:
                vec = np.array(pickle.loads(row[1]), dtype=np.float32).flatten()
                if len(vec) == q_dim:
                    valid.append((vec, row))
            except:
                continue

        if not valid:
            # Dimension mismatch — find what dims exist
            dims = set()
            for row in rows:
                try:
                    vec = np.array(pickle.loads(row[1]), dtype=np.float32).flatten()
                    dims.add(len(vec))
                except:
                    pass
            print(f"[SightRAG] Dimension mismatch. Query: {q_dim}, Stored: {dims}")
            print("[SightRAG] Clear old index: rag.clear() and re-index.")
            return []

        vectors = np.array([v[0] for v in valid], dtype=np.float32)

        # Cosine similarity
        q_n = q / (np.linalg.norm(q) + 1e-8)
        v_n = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        scores = v_n @ q_n

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            row = valid[i][1]
            try:
                bbox = json.loads(row[3])
            except:
                bbox = []
            results.append({
                "score":       round(float(scores[i]), 4),
                "image_path":  row[2],
                "bbox":        bbox,
                "timestamp":   row[4],
                "confidence":  round(float(row[5]), 4),
                "label":       row[6],
                "source_type": row[7],
            })
        return results

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def delete(self, id: str):
        self.conn.execute("DELETE FROM vectors WHERE id=?", (str(id),))
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM vectors")
        self.conn.commit()
