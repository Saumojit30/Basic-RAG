"""A minimal vector store backed by SQLite.

Every chunk is stored with its embedding vector (as JSON). Searching loads
all embeddings and ranks them with cosine similarity. That is perfectly fine
for hundreds of chunks. For millions, swap this class for a real vector
database (Qdrant, pgvector, Upstash Vector, Pinecone...) behind the same
search() interface.
"""

import json
import math
import sqlite3
from pathlib import Path

from .config import settings


class VectorStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._path = Path(db_path or settings.db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name       TEXT NOT NULL,
                    chunk_index    INTEGER NOT NULL,
                    text           TEXT NOT NULL,
                    embedding      TEXT NOT NULL,
                    embedding_name TEXT NOT NULL DEFAULT ''
                )
                """
            )
            try:  # upgrade databases created before embeddings were tagged
                conn.execute("ALTER TABLE chunks ADD COLUMN embedding_name TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def add_chunks(
        self,
        doc_name: str,
        texts: list[str],
        embeddings: list[list[float]],
        embedding_name: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (doc_name, chunk_index, text, embedding, embedding_name) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (doc_name, i, text, json.dumps(vec), embedding_name)
                    for i, (text, vec) in enumerate(zip(texts, embeddings))
                ],
            )

    def delete_doc(self, doc_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))

    def list_docs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_name, COUNT(*) AS chunks, SUM(LENGTH(text)) AS chars, "
                "MAX(embedding_name) AS embedding_name "
                "FROM chunks GROUP BY doc_name ORDER BY doc_name"
            ).fetchall()
        return [
            {
                "name": r["doc_name"],
                "chunks": r["chunks"],
                "chars": r["chars"] or 0,
                "embedding_name": r["embedding_name"] or "",
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def search(
        self,
        query_vec: list[float],
        top_k: int,
        embedding_name: str = "",
    ) -> list[dict]:
        """Top-k chunks whose vectors are closest to the query vector.

        Only chunks embedded with the SAME embedding model are considered:
        vectors from different models live in different spaces and must never
        be compared.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_name, chunk_index, text, embedding FROM chunks "
                "WHERE embedding_name = ?",
                (embedding_name,),
            ).fetchall()

        scored = [(cosine(query_vec, json.loads(r["embedding"])), r) for r in rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {
                "score": round(score, 4),
                "doc_name": r["doc_name"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
            }
            for score, r in scored[:top_k]
        ]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: how aligned two vectors are, from -1 to 1."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
