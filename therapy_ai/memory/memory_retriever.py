"""
therapy_ai/memory/memory_retriever.py

Semantic memory retrieval using sqlite-vec — a pure SQLite extension.
Embeds session summaries / message excerpts and retrieves the most
emotionally relevant past context for the current user input.

Why this is better than keyword-based retrieval (ContextBuilder preamble alone):
  - Finds thematically similar past sessions even when different words are used.
  - Surfaces the most emotionally relevant history, not just the most recent.
  - All vectors stored in the same SQLite DB — zero extra infrastructure.

Requires:
  pip install sqlite-vec sentence-transformers

The embedding model runs 100% locally (no API calls).
Recommended: all-MiniLM-L6-v2 (22 MB, very fast, good semantic quality).
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2
_EMBED_DIM = 384
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _pack_vector(vec: list[float]) -> bytes:
    """Serialise a float list to bytes for sqlite-vec storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


class MemoryRetriever:
    """
    Manages a local vector index of session excerpts inside the existing SQLite DB.
    
    Two-table design:
      memory_chunks     — plaintext session excerpt metadata (session_id, summary)
      memory_vectors    — sqlite-vec virtual table holding the embeddings

    Usage:
        retriever = MemoryRetriever(db_path="data/sessions.db")
        retriever.index_session(session_id, text_summary)
        results = retriever.retrieve(query="I'm struggling with my father", top_k=3)
    """

    def __init__(
        self,
        db_path: str | Path = "data/sessions.db",
        embed_model_id: str = _EMBED_MODEL_ID,
    ):
        self.db_path = Path(db_path)
        self._embed_model_id = embed_model_id
        self._embedder = None   # lazy-loaded on first use
        self._conn = None
        self._setup_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def index_session(self, session_id: str, text: str) -> None:
        """
        Embed `text` and upsert into the vector index.
        Call this at session close with the session's theme summary.
        """
        vec = self._embed(text)
        conn = self._get_conn()
        # Upsert into metadata table
        conn.execute(
            "INSERT OR REPLACE INTO memory_chunks(session_id, summary) VALUES (?, ?)",
            (session_id, text),
        )
        # Upsert into vector table — sqlite-vec requires rowid alignment
        rowid = conn.execute(
            "SELECT rowid FROM memory_chunks WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
            (rowid, _pack_vector(vec)),
        )
        conn.commit()
        logger.debug("Indexed session %s into vector memory.", session_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Find the top_k most semantically similar past sessions to `query`.
        Returns list of {"session_id": ..., "summary": ..., "distance": ...}
        """
        query_vec = self._embed(query)
        conn = self._get_conn()

        rows = conn.execute(
            """
            SELECT
                c.session_id,
                c.summary,
                v.distance
            FROM memory_vectors AS v
            JOIN memory_chunks  AS c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance ASC
            """,
            (_pack_vector(query_vec), top_k + 1),  # +1 to allow filtering current session
        ).fetchall()

        results = [
            {"session_id": r[0], "summary": r[1], "distance": r[2]}
            for r in rows
            if r[0] != exclude_session_id
        ]
        return results[:top_k]

    def retrieve_as_context_string(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: Optional[str] = None,
    ) -> str:
        """
        Convenience wrapper — returns a formatted string ready to inject
        into the system prompt as a soft memory block.
        """
        hits = self.retrieve(query, top_k=top_k, exclude_session_id=exclude_session_id)
        if not hits:
            return ""
        lines = ["[Relevant past context retrieved from memory]"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"{i}. {hit['summary']}")
        lines.append("(Use these as background context — don't repeat them verbatim.)")
        return "\n".join(lines)

    def remove_session(self, session_id: str) -> None:
        """Remove a session's vector from the index (called on session delete)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT rowid FROM memory_chunks WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (row[0],))
            conn.execute("DELETE FROM memory_chunks WHERE session_id = ?", (session_id,))
            conn.commit()

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Encode text to a dense float vector using the local sentence transformer."""
        if self._embedder is None:
            self._load_embedder()
        result = self._embedder.encode(text, normalize_embeddings=True)
        return result.tolist()

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_id)
            logger.info("Loaded embedding model: %s", self._embed_model_id)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic memory.\n"
                "Install it: pip install sentence-transformers"
            )

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _setup_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            logger.warning(
                "sqlite-vec extension not loaded (%s). "
                "Semantic retrieval will be unavailable. "
                "Install it: pip install sqlite-vec", e
            )
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                session_id TEXT PRIMARY KEY,
                summary    TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors
            USING vec0(embedding float[{_EMBED_DIM}])
        """)
        conn.commit()

    def _get_conn(self):
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def __del__(self):
        if self._conn:
            self._conn.close()
