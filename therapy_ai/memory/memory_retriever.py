"""
therapy_ai/memory/memory_retriever.py

Semantic memory retrieval with PROTECTED derived storage.

Confidentiality model (post vault-retirement, R9/R15)
──────────────────────────────────────────────────────
* Source summaries are PERSISTED ENCRYPTED at rest (AES-256-GCM via the
  workspace Encryptor) in `memory_sources(session_id, summary_enc)`.
* The retrieval index (embeddings + plaintext summaries) is held IN MEMORY only,
  rebuilt from decrypted sources after unlock. Nothing sensitive is written to
  disk in plaintext.
* A pre-existing persistent PLAINTEXT index from the old design
  (`memory_chunks` / `memory_vectors`) is NOT silently reused. On first open we
  detect and quarantine it (rename), logging that a migration/cleanup is needed.

Policy (R10/R11)
─────────────────
* Only conversations whose stored `memory_opt_in` is True may be indexed
  (contribution consent). Export/training authorization does NOT govern
  indexing and is not consulted here.

The embedding model runs 100% locally (no API calls): all-MiniLM-L6-v2.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

_EMBED_DIM = 384
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Legacy plaintext artifacts from the pre-refactor design.
_LEGACY_PLAINTEXT_TABLES = ("memory_chunks", "memory_vectors")


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _cosine(a: list[float], b: list[float]) -> float:
    # Embeddings are normalized at encode time, so dot product == cosine sim.
    return sum(x * y for x, y in zip(a, b))


class MemoryRetriever:
    """
    Encrypted-source semantic memory with an in-memory retrieval index.

    Usage:
        retr = MemoryRetriever(db_path=..., encryptor=store._enc)
        retr.index_session(session_id, summary, opted_in=True)
        hits = retr.retrieve("I'm struggling with my father", top_k=3)

    The in-memory index is (re)built lazily from encrypted sources, so a fresh
    process after unlock reconstructs it without any plaintext on disk.
    """

    def __init__(
        self,
        db_path,
        encryptor=None,
        encryptor_provider=None,
        embed_model_id: str = _EMBED_MODEL_ID,
    ):
        """
        db_path:  path to the SQLite workspace DB.
        encryptor_provider:  PREFERRED. A zero-arg callable returning the live
            workspace Encryptor, or None when the workspace is locked / the
            lifecycle this retriever was bound to has been revoked. Use
            `store.encryptor_provider()` so crypto access is lifecycle-bound:
            after lock (or a later re-unlock under a new lifecycle) the provider
            returns None and this retriever can no longer decrypt/rebuild/index.
        encryptor:  DEPRECATED. A raw long-lived Encryptor. Retained for existing
            call sites/tests, but it is NOT lifecycle-bound (survives lock), so
            production code must pass `encryptor_provider` instead.
        """
        self.db_path = Path(db_path)
        if encryptor_provider is not None:
            self._enc_provider = encryptor_provider
        elif encryptor is not None:
            # Deprecated: wrap a raw encryptor so the internal accessor is
            # uniform. This path is NOT lifecycle-bound by design.
            self._enc_provider = lambda e=encryptor: e
        else:
            self._enc_provider = lambda: None
        self._embed_model_id = embed_model_id
        self._embedder = None
        self._conn = None
        # In-memory index: session_id -> (summary_plaintext, embedding)
        self._index: dict[str, tuple[str, list[float]]] = {}
        self._index_loaded = False
        self._setup_db()

    @property
    def _enc(self):
        """Resolve the live encryptor via the lifecycle-bound provider (or None)."""
        return self._enc_provider()

    # ── Public API ────────────────────────────────────────────────────────────

    def index_session(self, session_id: str, text: str, opted_in: bool) -> None:
        """
        Persist an ENCRYPTED source summary and add it to the in-memory index.

        `opted_in` is the authoritative contribution-consent decision from the
        store (SessionRecord.memory_opt_in). Indexing is refused when False —
        a non-consenting conversation never enters cross-session memory.
        """
        if not opted_in:
            logger.debug(
                "Session %s has not opted in to memory contribution; not indexed.",
                session_id,
            )
            return
        if self._enc is None:
            raise RuntimeError("MemoryRetriever requires an unlocked workspace encryptor.")

        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO memory_sources(session_id, summary_enc) VALUES (?, ?)",
            (session_id, self._enc.encrypt(text)),
        )
        conn.commit()

        self._ensure_index_loaded()
        self._index[session_id] = (text, self._embed(text))
        logger.debug("Indexed session %s into protected semantic memory.", session_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: str | None = None,
    ) -> list[dict]:
        # Lifecycle-bound: with no live key (locked / revoked lifecycle) we must
        # neither rebuild nor serve a previously cached in-memory index. Drop any
        # stale cache and return nothing.
        if self._enc is None:
            self._index.clear()
            self._index_loaded = False
            return []
        self._ensure_index_loaded()
        if not self._index:
            return []
        qv = self._embed(query)
        scored = [
            {"session_id": sid, "summary": summ, "distance": 1.0 - _cosine(qv, emb)}
            for sid, (summ, emb) in self._index.items()
            if sid != exclude_session_id
        ]
        scored.sort(key=lambda d: d["distance"])
        return scored[:top_k]

    def retrieve_as_context_string(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: str | None = None,
    ) -> str:
        hits = self.retrieve(query, top_k=top_k, exclude_session_id=exclude_session_id)
        if not hits:
            return ""
        lines = ["[Relevant past context retrieved from memory]"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"{i}. {hit['summary']}")
        lines.append("(Use these as background context — don't repeat them verbatim.)")
        return "\n".join(lines)

    def remove_session(self, session_id: str) -> None:
        """
        Invalidate a session's memory (on delete or opt-out).

        Lifecycle-bound: refuses to write when the workspace is locked / the
        bound lifecycle is revoked, consistent with the no-write-after-lock
        guarantee. (The on-disk source is encrypted regardless; this prevents a
        retained reference from mutating storage post-lock.)
        """
        if self._enc is None:
            raise RuntimeError(
                "MemoryRetriever is locked (no live workspace lifecycle); "
                "refusing to modify derived memory."
            )
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_sources WHERE session_id = ?", (session_id,))
        conn.commit()
        self._index.pop(session_id, None)

    def invalidate_index(self) -> None:
        """Drop the in-memory index (e.g. on workspace lock)."""
        self._index.clear()
        self._index_loaded = False

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            self._load_embedder()
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

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

    # ── Index (re)build from encrypted sources ──────────────────────────────────

    def _ensure_index_loaded(self) -> None:
        if self._index_loaded:
            return
        if self._enc is None:
            # Locked / no key: cannot decrypt sources. Leave index empty.
            self._index_loaded = True
            return
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT session_id, summary_enc FROM memory_sources"
        ).fetchall()
        for sid, enc_summary in rows:
            try:
                summary = self._enc.decrypt(enc_summary)
            except Exception:
                logger.warning("Could not decrypt memory source for %s; skipping.", sid)
                continue
            self._index[sid] = (summary, self._embed(summary))
        self._index_loaded = True

    # ── DB setup / legacy handling ──────────────────────────────────────────────

    def _setup_db(self) -> None:
        conn = self._get_conn()

        # Quarantine any pre-existing PLAINTEXT index from the old design rather
        # than silently reusing it.
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
        }
        for legacy in _LEGACY_PLAINTEXT_TABLES:
            if legacy in existing:
                quarantined = f"_legacy_plaintext_{legacy}"
                if quarantined not in existing:
                    try:
                        conn.execute(f"ALTER TABLE {legacy} RENAME TO {quarantined}")
                        logger.warning(
                            "Quarantined legacy plaintext memory table %s -> %s "
                            "(not reused; manual cleanup/migration advised).",
                            legacy, quarantined,
                        )
                    except Exception as e:
                        logger.warning("Could not quarantine legacy table %s: %s", legacy, e)

        conn.execute(
            """CREATE TABLE IF NOT EXISTS memory_sources (
                session_id TEXT PRIMARY KEY,
                summary_enc TEXT NOT NULL
            )"""
        )
        conn.commit()

    def _get_conn(self):
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def __del__(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
