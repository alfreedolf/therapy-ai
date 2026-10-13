"""
tests/component/test_memory_policy.py

R9/R10/R11/R15: protected derived storage + authoritative memory policy.

  * Source summaries persisted ENCRYPTED at rest (no plaintext summary on disk).
  * memory_opt_in governs CONTRIBUTION only; default OFF for new conversations.
  * Receiving cross-session context is governed by global recall config, not by
    a per-session flag (no implicit isolated mode).
  * Export/training authorization does NOT govern indexing.
  * Opt-out / delete invalidates derived memory.
  * Pre-existing plaintext index artifacts are quarantined, not reused.

A fake embedder keeps these tests offline and deterministic (no model download).
Synthetic fixtures only.
"""

from __future__ import annotations

import sqlite3

import pytest

from therapy_ai.memory.memory_retriever import MemoryRetriever


class _FakeEmbedder:
    """Deterministic 3-dim embedder: counts occurrences of a few keywords."""
    KEYS = ("father", "work", "sleep")

    def encode(self, text, normalize_embeddings=True):
        import math
        t = text.lower()
        v = [float(t.count(k)) for k in self.KEYS]
        if normalize_embeddings:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / n for x in v]
        return _V(v)


class _V(list):
    def tolist(self):
        return list(self)


@pytest.fixture
def retriever(store, tmp_path):
    r = MemoryRetriever(
        db_path=store._engine.url.database,
        encryptor=store.active_encryptor,
    )
    r._embedder = _FakeEmbedder()  # inject offline embedder
    return r


class TestContributionConsent:
    def test_new_session_defaults_opt_out(self, store):
        sid = store.create_session(mood_before=5)
        assert store.memory_opt_in(sid) is False

    def test_opted_out_session_not_indexed(self, store, retriever):
        retriever.index_session("s1", "talking about work", opted_in=False)
        assert retriever.retrieve("work", top_k=5) == []

    def test_opted_in_session_indexed_and_retrievable(self, store, retriever):
        retriever.index_session("s1", "my father and work", opted_in=True)
        hits = retriever.retrieve("father", top_k=5)
        assert any(h["session_id"] == "s1" for h in hits)

    def test_opt_in_toggle_persists(self, store):
        sid = store.create_session(mood_before=5)
        store.set_memory_opt_in(sid, True)
        assert store.memory_opt_in(sid) is True


class TestProtectedDerivedStorage:
    def test_summary_encrypted_at_rest(self, store, retriever, tmp_path):
        sentinel = "SUMMARY_SENTINEL_SHOULD_NOT_APPEAR_42"
        retriever.index_session("s1", sentinel, opted_in=True)

        raw = sqlite3.connect(store._engine.url.database)
        try:
            rows = raw.execute("SELECT summary_enc FROM memory_sources").fetchall()
        finally:
            raw.close()
        assert rows
        for (stored,) in rows:
            assert sentinel not in stored  # ciphertext only

    def test_index_rebuilds_from_encrypted_sources(self, store, tmp_path):
        # First retriever indexes and persists encrypted source.
        r1 = MemoryRetriever(store._engine.url.database, encryptor=store.active_encryptor)
        r1._embedder = _FakeEmbedder()
        r1.index_session("s1", "father", opted_in=True)

        # A fresh retriever (new process simulation) rebuilds from encrypted src.
        r2 = MemoryRetriever(store._engine.url.database, encryptor=store.active_encryptor)
        r2._embedder = _FakeEmbedder()
        hits = r2.retrieve("father", top_k=5)
        assert any(h["session_id"] == "s1" for h in hits)

    def test_remove_session_invalidates(self, store, retriever):
        retriever.index_session("s1", "work", opted_in=True)
        retriever.remove_session("s1")
        assert retriever.retrieve("work", top_k=5) == []

    def test_legacy_plaintext_index_quarantined(self, store, tmp_path):
        # Simulate a pre-refactor plaintext index.
        db = store._engine.url.database
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE memory_chunks (session_id TEXT PRIMARY KEY, summary TEXT)")
        c.execute("INSERT INTO memory_chunks VALUES ('old','leaky plaintext')")
        c.commit(); c.close()

        MemoryRetriever(db, encryptor=store.active_encryptor)  # setup quarantines

        tables = {
            r[0]
            for r in sqlite3.connect(db).execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "memory_chunks" not in tables
        assert "_legacy_plaintext_memory_chunks" in tables


class TestReceivingIsSeparateFromContribution:
    def test_own_history_always_present_regardless_of_opt_in(self, store):
        from therapy_ai.memory.context_builder import ContextBuilder, ContextConfig
        sid = store.create_session(mood_before=5)  # opt-out by default
        store.add_message(sid, "user", "my own words")
        builder = ContextBuilder(store, config=ContextConfig(use_semantic_memory=False))
        ctx = builder.build(sid, latest_user_text="hi")
        assert any(m.content == "my own words" for m in ctx)

    def test_non_contributing_past_session_absent_from_preamble(self, store):
        from therapy_ai.memory.context_builder import ContextBuilder, ContextConfig
        prev = store.create_session(mood_before=4)
        store.add_message(prev, "user", "about my mother")
        store.close_session(prev, mood_after=6, rating=5, themes=["family"])
        # prev did NOT opt in → must not feed the preamble.

        cur = store.create_session(mood_before=5)
        store.add_message(cur, "user", "hello")
        builder = ContextBuilder(store, config=ContextConfig(use_semantic_memory=False))
        joined = " ".join(m.content for m in builder.build(cur, latest_user_text="hi"))
        assert "family" not in joined

    def test_contributing_past_session_feeds_preamble(self, store):
        from therapy_ai.memory.context_builder import ContextBuilder, ContextConfig
        prev = store.create_session(mood_before=4, memory_opt_in=True)
        store.add_message(prev, "user", "about my mother")
        store.close_session(prev, mood_after=6, rating=5, themes=["family"])

        cur = store.create_session(mood_before=5)
        store.add_message(cur, "user", "hello")
        builder = ContextBuilder(store, config=ContextConfig(use_semantic_memory=False))
        joined = " ".join(m.content for m in builder.build(cur, latest_user_text="hi"))
        assert "family" in joined


class TestExportSeparation:
    def test_export_does_not_toggle_memory(self, store):
        """Exporting must not change any conversation's contribution consent."""
        from therapy_ai.memory.exporter import FinetuneExporter

        s1 = store.create_session(mood_before=5)  # opt-out
        store.add_message(s1, "user", "u1")
        store.add_message(s1, "assistant", "a1")
        store.add_message(s1, "user", "u2")
        store.add_message(s1, "assistant", "a2")
        store.close_session(s1, mood_after=6, rating=5)

        before = store.memory_opt_in(s1)
        FinetuneExporter(store).export(output_dir=store._engine.url.database + "_ft")
        after = store.memory_opt_in(s1)
        assert before == after == False
