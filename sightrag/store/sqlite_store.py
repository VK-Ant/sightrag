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
                ocr_text    TEXT DEFAULT '',
                metadata    TEXT
            )
        """)

        # v0.5: tracks table — one row per tracked object
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id        INTEGER PRIMARY KEY,
                video_path      TEXT,
                label           TEXT,
                first_frame     INTEGER,
                last_frame      INTEGER,
                first_timestamp TEXT,
                last_timestamp  TEXT,
                frame_count     INTEGER,
                trajectory      TEXT
            )
        """)

        # v0.5: track_detections — links track_id to frame-level detections
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS track_detections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id        INTEGER,
                video_path      TEXT,
                frame_idx       INTEGER,
                timestamp       TEXT,
                bbox            TEXT,
                label           TEXT,
                confidence      REAL
            )
        """)

        # Auto-migrate: add columns if upgrading from old database
        try:
            cols = [row[1] for row in
                    self.conn.execute("PRAGMA table_info(vectors)").fetchall()]
            if "ocr_text" not in cols:
                self.conn.execute(
                    "ALTER TABLE vectors ADD COLUMN ocr_text TEXT DEFAULT ''"
                )
                self.conn.commit()
        except Exception:
            pass

        # Create index for fast track lookups
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_track_det_track
                ON track_detections(track_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_track_det_frame
                ON track_detections(video_path, frame_idx)
            """)
        except Exception:
            pass

        self.conn.commit()

    def add(self, id: str, embedding, metadata: dict = {}):
        emb = np.array(embedding, dtype=np.float32).flatten()
        self.conn.execute("""
            INSERT OR REPLACE INTO vectors
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            str(id),
            pickle.dumps(emb),
            str(metadata.get("image_path", "")),
            json.dumps(metadata.get("bbox", [])),
            str(metadata.get("timestamp", "")),
            float(metadata.get("confidence", 0.0)),
            str(metadata.get("label", "")),
            str(metadata.get("source_type", "image")),
            str(metadata.get("ocr_text", "")),
            json.dumps(metadata)
        ))
        self.conn.commit()

    def search(self, query_vector, top_k: int = 5):
        rows = self.conn.execute("""
            SELECT id, embedding, image_path,
                   bbox, timestamp, confidence,
                   label, source_type, ocr_text, metadata
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
                "ocr_text":    row[8] if len(row) > 8 else "",
            })
        return results

    # ── Track storage (v0.5) ──────────────────────────────────────────

    def save_track(self, track_id, video_path, label,
                   first_frame, last_frame,
                   first_timestamp, last_timestamp,
                   frame_count, trajectory=None):
        """Save or update a track summary."""
        traj_json = json.dumps(trajectory) if trajectory else "[]"
        self.conn.execute("""
            INSERT OR REPLACE INTO tracks
            (track_id, video_path, label, first_frame, last_frame,
             first_timestamp, last_timestamp, frame_count, trajectory)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            int(track_id), str(video_path), str(label),
            int(first_frame), int(last_frame),
            str(first_timestamp), str(last_timestamp),
            int(frame_count), traj_json
        ))
        self.conn.commit()

    def add_track_detection(self, track_id, video_path, frame_idx,
                            timestamp, bbox, label, confidence):
        """Store a frame-level track detection."""
        self.conn.execute("""
            INSERT INTO track_detections
            (track_id, video_path, frame_idx, timestamp, bbox, label, confidence)
            VALUES (?,?,?,?,?,?,?)
        """, (
            int(track_id), str(video_path), int(frame_idx),
            str(timestamp), json.dumps(bbox),
            str(label), float(confidence)
        ))
        # Batch commit — called frequently during indexing
        if frame_idx % 50 == 0:
            self.conn.commit()

    def get_track(self, track_id):
        """Get full track data including trajectory."""
        row = self.conn.execute("""
            SELECT track_id, video_path, label, first_frame, last_frame,
                   first_timestamp, last_timestamp, frame_count, trajectory
            FROM tracks WHERE track_id = ?
        """, (int(track_id),)).fetchone()

        if row is None:
            return None

        try:
            trajectory = json.loads(row[8])
        except:
            trajectory = []

        # Get all frame detections for this track
        det_rows = self.conn.execute("""
            SELECT frame_idx, timestamp, bbox, label, confidence
            FROM track_detections
            WHERE track_id = ?
            ORDER BY frame_idx
        """, (int(track_id),)).fetchall()

        states = []
        for dr in det_rows:
            try:
                bbox = json.loads(dr[2])
            except:
                bbox = []
            states.append({
                "frame_idx": dr[0],
                "timestamp": dr[1],
                "bbox": bbox,
                "label": dr[3],
                "confidence": round(float(dr[4]), 4),
            })

        return {
            "track_id": row[0],
            "video_path": row[1],
            "label": row[2],
            "first_frame": row[3],
            "last_frame": row[4],
            "first_timestamp": row[5],
            "last_timestamp": row[6],
            "frame_count": row[7],
            "trajectory": trajectory,
            "states": states,
        }

    def get_all_tracks(self):
        """Get summaries of all tracks."""
        rows = self.conn.execute("""
            SELECT track_id, video_path, label, first_frame, last_frame,
                   first_timestamp, last_timestamp, frame_count
            FROM tracks ORDER BY track_id
        """).fetchall()

        return [
            {
                "track_id": r[0],
                "video_path": r[1],
                "label": r[2],
                "first_frame": r[3],
                "last_frame": r[4],
                "first_timestamp": r[5],
                "last_timestamp": r[6],
                "frame_count": r[7],
            }
            for r in rows
        ]

    def find_track_for_detection(self, image_path, bbox, timestamp):
        """Find track that contains a detection at given frame/bbox."""
        if not bbox or not image_path:
            return None

        bbox_json = json.dumps(bbox)

        # Try exact match first
        row = self.conn.execute("""
            SELECT td.track_id, t.label, t.first_timestamp,
                   t.last_timestamp, t.frame_count
            FROM track_detections td
            JOIN tracks t ON td.track_id = t.track_id
            WHERE td.video_path = ? AND td.bbox = ?
            LIMIT 1
        """, (str(image_path), bbox_json)).fetchone()

        if row:
            return {
                "track_id": row[0],
                "label": row[1],
                "first_timestamp": row[2],
                "last_timestamp": row[3],
                "frame_count": row[4],
            }

        # Fallback: find by timestamp + closest bbox
        if timestamp:
            det_rows = self.conn.execute("""
                SELECT td.track_id, td.bbox, t.label,
                       t.first_timestamp, t.last_timestamp, t.frame_count
                FROM track_detections td
                JOIN tracks t ON td.track_id = t.track_id
                WHERE td.video_path = ? AND td.timestamp = ?
            """, (str(image_path), str(timestamp))).fetchall()

            if det_rows:
                best = None
                best_iou = 0
                for dr in det_rows:
                    try:
                        det_bbox = json.loads(dr[1])
                    except:
                        continue
                    iou = self._bbox_iou(bbox, det_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best = dr

                if best and best_iou > 0.3:
                    return {
                        "track_id": best[0],
                        "label": best[2],
                        "first_timestamp": best[3],
                        "last_timestamp": best[4],
                        "frame_count": best[5],
                    }

        return None

    @staticmethod
    def _bbox_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def track_count(self) -> int:
        """Count total tracked objects."""
        try:
            return self.conn.execute(
                "SELECT COUNT(*) FROM tracks"
            ).fetchone()[0]
        except:
            return 0

    def delete(self, id: str):
        self.conn.execute("DELETE FROM vectors WHERE id=?", (str(id),))
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM vectors")
        self.conn.execute("DELETE FROM tracks")
        self.conn.execute("DELETE FROM track_detections")
        self.conn.commit()
