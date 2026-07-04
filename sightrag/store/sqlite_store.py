# sightrag/store/sqlite_store.py
# Built-in vector store — zero extra dependencies
# Pure Python + SQLite + numpy

import sqlite3
import pickle
import json
import numpy as np
from pathlib import Path
from .base import VectorStoreBase


class SQLiteStore(VectorStoreBase):
    """
    Default vector store for SightRAG.
    No ChromaDB needed. No cost. Works everywhere.
    Good up to ~100k images.
    """

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
        self.conn.execute("""
            INSERT OR REPLACE INTO vectors 
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            id,
            pickle.dumps(embedding.astype(np.float32)),
            metadata.get("image_path", ""),
            json.dumps(metadata.get("bbox", [])),
            metadata.get("timestamp", ""),
            float(metadata.get("confidence", 0.0)),
            metadata.get("label", ""),
            metadata.get("source_type", "image"),
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

        # Batch cosine similarity — pure numpy
        stored = np.array([
            pickle.loads(row[1]) for row in rows
        ], dtype=np.float32)

        q = query_vector.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        s_norm = stored / (
            np.linalg.norm(stored, axis=1, keepdims=True) + 1e-8
        )
        scores = s_norm @ q_norm

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            results.append({
                "score":       float(scores[i]),
                "image_path":  rows[i][2],
                "bbox":        json.loads(rows[i][3]),
                "timestamp":   rows[i][4],
                "confidence":  rows[i][5],
                "label":       rows[i][6],
                "source_type": rows[i][7],
                "metadata":    json.loads(rows[i][8])
            })

        return results

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM vectors"
        ).fetchone()[0]

    def delete(self, id: str):
        self.conn.execute(
            "DELETE FROM vectors WHERE id=?", (id,)
        )
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM vectors")
        self.conn.commit()
