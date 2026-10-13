"""
tests/component/test_transition_guard.py

R8: Unretired legacy vault records must never be treated as ordinary records.
Ordinary flows either EXCLUDE them (list/mood/history) or raise
RetirementRequiredError (direct get/load/close/opt-in). Their message content
is NEVER decrypted with the workspace key.

Also covers the VAULT_ONLY_LEGACY classification (R5 precondition): a workspace
whose only content is unretired vault rows has no app-key anchor, so ordinary
unlock is refused.

Synthetic fixtures only.
"""

from __future__ import annotations

import base64
import os
import sqlite3

import pytest

from therapy_ai.memory.session_store import (
    Encryptor,
    RetirementRequiredError,
    SessionStore,
    WorkspaceState,
    WorkspaceStateError,
)

PW = "workspace-pass"
VAULT_PW = "legacy-vault-pass"
VERIFIER_PLAINTEXT = "therapy-ai-vault-ok"  # the OLD per-vault verifier token


def _insert_legacy_vault_row(db_path, salt_path, *, with_app_anchor: bool):
    """
    Create a DB containing a legacy vault session (own salt + verifier, content
    sealed with a vault key). Optionally also add one ordinary app-key message
    so the workspace is bootstrappable vs. vault-only.
    """
    # Workspace salt (for the optional app anchor + later init).
    salt = os.urandom(16)
    with open(salt_path, "wb") as f:
        f.write(base64.b64encode(salt))

    # Build schema matching current ORM (so SessionStore opens it cleanly).
    c = sqlite3.connect(db_path)
    c.execute("""CREATE TABLE sessions (
        id TEXT PRIMARY KEY, created_at DATETIME, ended_at DATETIME,
        mood_before INTEGER, mood_after INTEGER, rating INTEGER,
        themes_enc TEXT, note_enc TEXT,
        memory_opt_in BOOLEAN DEFAULT 0,
        is_vault BOOLEAN DEFAULT 0, vault_salt TEXT, vault_verifier TEXT
    )""")
    c.execute("""CREATE TABLE messages (
        id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
        content_enc TEXT, timestamp DATETIME, tokens INTEGER, is_flagged BOOLEAN
    )""")

    # Vault row: per-row salt + verifier + vault-key-sealed message.
    vsalt = os.urandom(16)
    venc = Encryptor(VAULT_PW, salt=vsalt)
    c.execute(
        "INSERT INTO sessions (id, created_at, is_vault, vault_salt, vault_verifier) "
        "VALUES (?,?,?,?,?)",
        ("vault-1", "2026-01-01 00:00:00", 1,
         base64.b64encode(vsalt).decode(), venc.encrypt(VERIFIER_PLAINTEXT)),
    )
    c.execute(
        "INSERT INTO messages (session_id, role, content_enc, timestamp) VALUES (?,?,?,?)",
        ("vault-1", "user", venc.encrypt("vault-only secret"), "2026-01-01 00:00:01"),
    )

    if with_app_anchor:
        aenc = Encryptor(PW, salt=salt)
        c.execute(
            "INSERT INTO sessions (id, created_at, is_vault) VALUES (?,?,0)",
            ("app-1", "2026-01-02 00:00:00"),
        )
        c.execute(
            "INSERT INTO messages (session_id, role, content_enc, timestamp) VALUES (?,?,?,?)",
            ("app-1", "user", aenc.encrypt("ordinary"), "2026-01-02 00:00:01"),
        )
    c.commit(); c.close()
    return salt


class TestVaultExcludedFromOrdinaryFlows:
    @pytest.fixture
    def store_with_vault(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _insert_legacy_vault_row(db, isolated_salt, with_app_anchor=True)
        s = SessionStore(db_path=db)
        # Bootstrap via the ordinary app anchor so we're unlocked.
        assert s.legacy_bootstrap(PW) is True
        return s

    def test_list_sessions_excludes_vault(self, store_with_vault):
        ids = [s.id for s in store_with_vault.list_sessions()]
        assert "vault-1" not in ids
        assert "app-1" in ids

    def test_mood_series_excludes_vault(self, store_with_vault):
        # (neither row has mood_before here; assert no crash + no vault leak)
        series = store_with_vault.mood_series()
        assert all("vault" not in str(d) for d in series)

    def test_get_vault_raises_retirement_required(self, store_with_vault):
        with pytest.raises(RetirementRequiredError):
            store_with_vault.get_session("vault-1")

    def test_load_vault_messages_raises_not_decrypts(self, store_with_vault):
        # Must NOT return plaintext and must NOT attempt workspace-key decrypt.
        with pytest.raises(RetirementRequiredError):
            store_with_vault.load_messages("vault-1")

    def test_close_vault_raises(self, store_with_vault):
        with pytest.raises(RetirementRequiredError):
            store_with_vault.close_session("vault-1", mood_after=5)

    def test_memory_opt_in_on_vault_raises(self, store_with_vault):
        with pytest.raises(RetirementRequiredError):
            store_with_vault.set_memory_opt_in("vault-1", True)

    def test_vault_ids_listed_for_retirement_tool(self, store_with_vault):
        assert store_with_vault.list_vault_session_ids() == ["vault-1"]


class TestVaultOnlyLegacyState:
    def test_vault_only_classified_and_unlock_refused(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _insert_legacy_vault_row(db, isolated_salt, with_app_anchor=False)
        s = SessionStore(db_path=db)
        assert s.classify() == WorkspaceState.VAULT_ONLY_LEGACY
        # No circular dependency: ordinary unlock refuses, pointing to recovery.
        with pytest.raises(WorkspaceStateError):
            s.unlock(PW)
        # And legacy_bootstrap also refuses (no app-key anchor exists).
        with pytest.raises(WorkspaceStateError):
            s.legacy_bootstrap(PW)
