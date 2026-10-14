"""
tests/component/test_lock_lifecycle.py

Lock-lifecycle access revocation (handoff section 5, tasks 1 & 3).

Corrected acceptance criterion (per review):
  invalidate_index() is CACHE invalidation, not access revocation. The real
  guarantee is: a retriever retained before workspace lock cannot retrieve,
  rebuild, decrypt, index, or write through supported interfaces after lock,
  and a subsequent unlock does NOT silently reactivate an object bound to the
  old unlocked lifecycle.

These tests drive the PRODUCTION revocation mechanism (`store.lock()`) and the
lifecycle-bound encryptor provider. They do NOT manually clear `retriever._enc`
and do NOT rely on Streamlit session_state clearing as the boundary.

Synthetic fixtures only; a fake embedder keeps the tests offline/deterministic.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from tests.conftest import TEST_PASSPHRASE
from therapy_ai.memory.memory_retriever import MemoryRetriever
from therapy_ai.memory.session_store import WorkspaceLockedError


class _FakeEmbedder:
    """Deterministic 3-dim embedder (mirrors test_memory_policy); no model."""
    KEYS = ("father", "work", "sleep")

    def encode(self, text, normalize_embeddings=True):
        t = text.lower()
        v = [float(t.count(k)) for k in self.KEYS]
        if normalize_embeddings:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / n for x in v]
        return _V(v)


class _V(list):
    def tolist(self):
        return list(self)


def _lifecycle_retriever(store) -> MemoryRetriever:
    """A retriever bound to the store's CURRENT lifecycle (production wiring)."""
    r = MemoryRetriever(
        db_path=store._engine.url.database,
        encryptor_provider=store.encryptor_provider(),
    )
    r._embedder = _FakeEmbedder()
    return r


def _opted_in_session(store, text: str) -> str:
    sid = store.create_session(mood_before=5)
    store.set_memory_opt_in(sid, True)
    store.add_message(sid, "user", text)
    return sid


# ── Store-level encryptor revocation ────────────────────────────────────────────

class TestStoreEncryptorRevocation:
    def test_active_encryptor_present_while_unlocked(self, store):
        assert store.is_unlocked is True
        assert store.active_encryptor is not None

    def test_lock_revokes_active_encryptor(self, store):
        store.lock()
        assert store.is_unlocked is False
        assert store.active_encryptor is None

    def test_protected_op_fails_closed_after_lock(self, store):
        sid = store.create_session(mood_before=5)
        store.lock()
        with pytest.raises(WorkspaceLockedError):
            store.load_messages(sid)


# ── Lifecycle-bound provider resolution ─────────────────────────────────────────

class TestLifecycleEncryptorProvider:
    def test_provider_resolves_while_unlocked(self, store):
        provider = store.encryptor_provider()
        assert provider() is not None

    def test_provider_resolves_none_after_lock(self, store):
        provider = store.encryptor_provider()
        store.lock()
        assert provider() is None

    def test_old_provider_stays_none_after_reunlock(self, store):
        """A provider bound to the OLD lifecycle must not reactivate on unlock."""
        old_provider = store.encryptor_provider()
        store.lock()
        store.unlock(TEST_PASSPHRASE)  # NEW lifecycle, new token
        assert old_provider() is None  # old binding stays revoked
        # A freshly obtained provider works again.
        assert store.encryptor_provider()() is not None


# ── Retained-retriever revocation through PRODUCTION mechanism ──────────────────

class TestRetainedRetrieverRevocation:
    def test_retained_retriever_cannot_retrieve_or_write_after_lock(self, store):
        """
        Core regression (item 1): retain a retriever, lock via the production
        mechanism, and confirm it can neither retrieve nor index/write — WITHOUT
        touching retriever internals.
        """
        sid = _opted_in_session(store, "my father and work")
        retr = _lifecycle_retriever(store)
        retr.index_session(sid, "my father and work", opted_in=True)
        assert retr.retrieve("father", top_k=3)  # works while unlocked

        # Production revocation — no manual _enc clearing.
        store.lock()

        # Retrieval: nothing (no rebuild, no serving a stale cache).
        assert retr.retrieve("father", top_k=3) == []
        # Indexing/writing through the retained reference: refused.
        with pytest.raises(RuntimeError):
            retr.index_session(sid, "my father and work", opted_in=True)
        with pytest.raises(RuntimeError):
            retr.remove_session(sid)

    def test_retained_retriever_stays_dead_after_reunlock(self, store):
        """A subsequent unlock must not silently reactivate the old retriever."""
        sid = _opted_in_session(store, "trouble at work and sleep")
        retr = _lifecycle_retriever(store)
        retr.index_session(sid, "trouble at work and sleep", opted_in=True)
        assert retr.retrieve("work", top_k=3)

        store.lock()
        store.unlock(TEST_PASSPHRASE)  # new lifecycle

        # The retained retriever is bound to the OLD lifecycle → still dead.
        assert retr.retrieve("work", top_k=3) == []
        with pytest.raises(RuntimeError):
            retr.index_session(sid, "trouble at work and sleep", opted_in=True)

    def test_fresh_retriever_after_reunlock_works(self, store):
        """A newly authorized retriever operates normally after re-unlock."""
        sid = _opted_in_session(store, "father and work and sleep")
        retr = _lifecycle_retriever(store)
        retr.index_session(sid, "father and work and sleep", opted_in=True)

        store.lock()
        store.unlock(TEST_PASSPHRASE)

        fresh = _lifecycle_retriever(store)  # bound to the new lifecycle
        assert fresh.retrieve("father", top_k=3)  # rebuilds from encrypted sources

    def test_cache_not_served_after_lock(self, store):
        """
        Even with the index already loaded in memory before lock, a locked
        retrieve() must not serve the cached plaintext summaries.
        """
        sid = _opted_in_session(store, "father father work")
        retr = _lifecycle_retriever(store)
        retr.index_session(sid, "father father work", opted_in=True)
        assert retr._index  # cache populated in memory

        store.lock()
        assert retr.retrieve("father", top_k=3) == []
        assert retr._index == {}  # stale cache dropped on locked access


# ── On-disk confidentiality is independent of lock state ────────────────────────

class TestSourceCiphertextAtRest:
    def test_encrypted_source_never_plaintext(self, store):
        secret = "father conflict specific detail"
        sid = _opted_in_session(store, secret)
        retr = _lifecycle_retriever(store)
        retr.index_session(sid, secret, opted_in=True)
        store.lock()

        raw = sqlite3.connect(store._engine.url.database)
        try:
            rows = raw.execute("SELECT summary_enc FROM memory_sources").fetchall()
        finally:
            raw.close()
        assert rows
        for (stored,) in rows:
            assert secret not in stored  # ciphertext only
