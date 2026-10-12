"""
tests/component/test_vault_context.py

Component tests for vault-awareness in ContextBuilder (Issue A).

Guarantees under test:
  1. A vault session builds context from its OWN messages (no crash) when the
     vault passphrase is supplied.
  2. Building a vault session's context WITHOUT the passphrase raises (never
     returns ciphertext).
  3. A vault session's context does NOT include the cross-session memory block
     (preamble / semantic) — vault sessions are isolated.
  4. Ordinary (non-vault) sessions are unchanged.
  5. The semantic index refuses to index vault sessions (defense-in-depth).
"""

from __future__ import annotations

import pytest

from therapy_ai.memory.context_builder import ContextBuilder, ContextConfig

VAULT_PASS = "vault-secret-123"


@pytest.fixture
def builder(store):
    # Semantic memory off for determinism; the preamble (Layer 1) still runs.
    return ContextBuilder(store, config=ContextConfig(use_semantic_memory=False))


class TestVaultContextBuild:
    def test_vault_session_builds_with_passphrase(self, store, builder):
        sid = store.create_session(mood_before=5, vault_passphrase=VAULT_PASS)
        store.add_message(sid, "user", "A locked thought.", vault_passphrase=VAULT_PASS)

        context = builder.build(
            sid, latest_user_text="more", vault_passphrase=VAULT_PASS
        )
        assert any(m.content == "A locked thought." for m in context)

    def test_vault_build_without_passphrase_raises(self, store, builder):
        sid = store.create_session(mood_before=5, vault_passphrase=VAULT_PASS)
        store.add_message(sid, "user", "A locked thought.", vault_passphrase=VAULT_PASS)

        with pytest.raises(PermissionError):
            builder.build(sid, latest_user_text="more")  # no passphrase

    def test_vault_context_excludes_cross_session_memory(self, store, builder):
        # A closed non-vault session with themes would normally feed the preamble.
        prev = store.create_session(mood_before=4)
        store.add_message(prev, "user", "I struggle with my mother.")
        store.close_session(prev, mood_after=6, rating=5, themes=["family", "guilt"])

        # New vault session: its context must NOT carry the cross-session preamble.
        vid = store.create_session(mood_before=5, vault_passphrase=VAULT_PASS)
        store.add_message(vid, "user", "Something private.", vault_passphrase=VAULT_PASS)

        context = builder.build(
            vid, latest_user_text="x", vault_passphrase=VAULT_PASS
        )
        joined = " ".join(m.content for m in context)
        assert "Memory from previous sessions" not in joined
        assert "family" not in joined  # no theme bleed-through
        # ...but the vault's own message is present.
        assert "Something private." in joined


class TestOrdinarySessionUnchanged:
    def test_non_vault_still_gets_preamble(self, store, builder):
        prev = store.create_session(mood_before=4)
        store.add_message(prev, "user", "About my mother.")
        store.close_session(prev, mood_after=6, rating=5, themes=["family"])

        cur = store.create_session(mood_before=5)
        store.add_message(cur, "user", "Hello.")
        context = builder.build(cur, latest_user_text="hi")

        joined = " ".join(m.content for m in context)
        # Ordinary sessions still receive the cross-session memory preamble.
        assert "Memory from previous sessions" in joined


class TestIndexRefusesVault:
    def test_index_session_skips_vault(self, store):
        from therapy_ai.memory.memory_retriever import MemoryRetriever

        try:
            retr = MemoryRetriever(db_path=store._engine.url.database)
        except Exception:
            pytest.skip("sqlite-vec not available")

        # Indexing a vault session must be a no-op (no plaintext leak into index).
        retr.index_session("vault-sid", "secret vault summary", is_vault=True)
        hits = retr.retrieve("secret vault summary", top_k=5)
        assert all(h["session_id"] != "vault-sid" for h in hits)
