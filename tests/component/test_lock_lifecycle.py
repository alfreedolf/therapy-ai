"""
tests/component/test_lock_lifecycle.py

Focused verification of lock-lifecycle revocation at the component layer
(pending-verification tasks 1 & 3 from docs/development_handoff.md section 5):

  * On lock, the store's active encryptor is revoked (active_encryptor -> None)
    and protected operations fail closed.
  * invalidate_index() clears the retriever's in-memory index so no decrypted
    cross-session memory survives a lock (no post-lock display/leak).
  * A retriever that has lost its encryptor cannot repopulate its index from the
    encrypted sources on disk — retrieval yields nothing until re-unlock
    (retained-reference revocation).

These behaviors are implemented in the store (`lock`, `active_encryptor`,
`_require_unlocked`, lifecycle token) and the retriever (`invalidate_index`,
`_ensure_index_loaded` leaving the index empty when `_enc is None`); the UI
(`app._lock_and_clear`) composes them. The store-level lifecycle token is
already covered in test_workspace_auth; this module covers the retriever/
encryptor revocation path those UI behaviors depend on.

Synthetic fixtures only. A fake embedder keeps the tests offline/deterministic.
"""

from __future__ import annotations

import math

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


def _retriever_for(store) -> MemoryRetriever:
    r = MemoryRetriever(
        db_path=store._engine.url.database,
        encryptor=store.active_encryptor,
    )
    r._embedder = _FakeEmbedder()
    return r


def _opted_in_session(store, text: str) -> str:
    """Create a session that CONTRIBUTES to memory, with one message."""
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


# ── Retriever index invalidation on lock ────────────────────────────────────────

class TestRetrieverInvalidationOnLock:
    def test_invalidate_index_alone_is_insufficient_without_key_revocation(self, store):
        """
        VERIFICATION FINDING: invalidate_index() clears the in-memory index, but
        if the retriever still holds a live encryptor, the next retrieve() call
        repopulates the index from the encrypted sources on disk. Invalidation is
        therefore only safe when the encryptor reference is ALSO revoked.

        In the real UI this is correct: app._lock_and_clear() invalidates the
        index, locks the store, and then deletes all session_state (discarding
        the retriever object entirely), so no retained retriever survives with a
        live key. This test pins down the dependency so a future change that
        invalidates WITHOUT dropping the key doesn't silently reintroduce a leak.
        """
        sid = _opted_in_session(store, "my father and work")
        retr = _retriever_for(store)
        retr.index_session(sid, "my father and work", opted_in=True)
        assert retr.retrieve("father", top_k=3)

        # Invalidation alone: a retriever that still has its key rebuilds.
        retr.invalidate_index()
        assert retr.retrieve("father", top_k=3), (
            "invalidate_index() alone should not be relied on for lock safety; "
            "the encryptor reference must also be revoked"
        )

        # Invalidation + key revocation (what lock actually entails): empty.
        retr._enc = None
        retr.invalidate_index()
        assert retr.retrieve("father", top_k=3) == []

    def test_no_repopulation_without_encryptor(self, store):
        """
        After the index is invalidated AND the encryptor is gone, the retriever
        must not be able to rebuild the index from the encrypted sources on disk.
        This is the retained-reference revocation guarantee: a surviving
        retriever object yields nothing once the workspace is locked.
        """
        sid = _opted_in_session(store, "trouble at work and sleep")
        retr = _retriever_for(store)
        retr.index_session(sid, "trouble at work and sleep", opted_in=True)
        assert retr.retrieve("work", top_k=3)

        # Workspace lock: store drops its key; UI also drops the retriever's.
        store.lock()
        retr._enc = None            # mirror app._lock_and_clear dropping the ref
        retr.invalidate_index()

        # Encrypted sources still exist on disk, but with no key the index stays
        # empty — nothing decrypts, nothing is returned.
        assert retr.retrieve("work", top_k=3) == []
        assert retr._index == {}

    def test_encrypted_source_persists_but_is_not_plaintext(self, store, tmp_path):
        """Lock must not expose the source; the on-disk summary stays ciphertext."""
        import sqlite3

        secret = "father conflict detail"
        sid = _opted_in_session(store, secret)
        retr = _retriever_for(store)
        retr.index_session(sid, secret, opted_in=True)
        store.lock()

        raw = sqlite3.connect(store._engine.url.database)
        try:
            rows = raw.execute("SELECT summary_enc FROM memory_sources").fetchall()
        finally:
            raw.close()
        assert rows
        for (stored,) in rows:
            assert secret not in stored  # ciphertext only, even after lock


# ── Re-unlock restores retrieval ────────────────────────────────────────────────

class TestReUnlockRestores:
    def test_fresh_retriever_after_reunlock_retrieves_again(self, store, tmp_path):
        sid = _opted_in_session(store, "father and work and sleep")
        retr = _retriever_for(store)
        retr.index_session(sid, "father and work and sleep", opted_in=True)

        db = store._engine.url.database
        store.lock()
        store.unlock(TEST_PASSPHRASE)

        # A retriever rebuilt with the re-derived encryptor repopulates from the
        # encrypted sources and can retrieve again.
        retr2 = MemoryRetriever(db_path=db, encryptor=store.active_encryptor)
        retr2._embedder = _FakeEmbedder()
        assert retr2.retrieve("father", top_k=3)
