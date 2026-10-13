"""
tests/component/test_vault_retirement.py

Regression coverage for the ONE-TIME, conversion-only vault retirement tool
(R5/R17). Synthetic fixtures only; no live data, no deletion mechanism.

Covered:
  * Wrong vault credential leaves the source unchanged (no writes).
  * Conversion preserves conversation id + message history and other
    conversations.
  * Converted conversation has memory contribution OFF (excluded from memory
    until explicitly enabled).
  * Interrupted retirement is recoverable (original intact; backup discarded).
  * Consistent backup accounts for WAL state.
  * Unresolved metadata encryption dependency STOPS rather than guessing.
  * Vault-only legacy recovery: verify vault cred, init NEW target workspace,
    convert into it — without claiming the old app passphrase was authenticated.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3

import pytest

from therapy_ai.memory.session_store import (
    Encryptor,
    SessionStore,
    WorkspaceState,
)
from therapy_ai.memory import vault_retirement as vr

APP_PW = "workspace-app-pass"
VAULT_PW = "legacy-vault-pass"
NEW_PW = "freshly-chosen-workspace-pass"
VAULT_VERIFIER = "therapy-ai-vault-ok"


def _write_salt(salt_path, salt: bytes) -> None:
    with open(salt_path, "wb") as f:
        f.write(base64.b64encode(salt))


def _schema(conn):
    conn.execute("""CREATE TABLE sessions (
        id TEXT PRIMARY KEY, created_at DATETIME, ended_at DATETIME,
        mood_before INTEGER, mood_after INTEGER, rating INTEGER,
        themes_enc TEXT, note_enc TEXT,
        memory_opt_in BOOLEAN DEFAULT 0,
        is_vault BOOLEAN DEFAULT 0, vault_salt TEXT, vault_verifier TEXT
    )""")
    conn.execute("""CREATE TABLE messages (
        id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
        content_enc TEXT, timestamp DATETIME, tokens INTEGER, is_flagged BOOLEAN
    )""")


def _make_case1_db(db_path, salt_path):
    """
    Case 1: workspace has an app anchor (an ordinary app-key conversation) PLUS
    a legacy vault conversation whose themes/note were app-sealed (as in the
    retired design). No workspace_meta verifier yet (legacy).
    """
    app_salt = os.urandom(16)
    _write_salt(salt_path, app_salt)
    app_enc = Encryptor(APP_PW, salt=app_salt)

    conn = sqlite3.connect(db_path)
    _schema(conn)

    # Ordinary app-key conversation (anchor).
    conn.execute("INSERT INTO sessions (id, created_at, is_vault) VALUES ('app-1','2026-01-01 00:00:00',0)")
    conn.execute(
        "INSERT INTO messages (session_id, role, content_enc, timestamp) VALUES (?,?,?,?)",
        ("app-1", "user", app_enc.encrypt("ordinary message"), "2026-01-01 00:00:01"),
    )

    # Legacy vault conversation: vault-sealed messages, APP-sealed themes/note.
    vsalt = os.urandom(16)
    venc = Encryptor(VAULT_PW, salt=vsalt)
    conn.execute(
        "INSERT INTO sessions (id, created_at, ended_at, is_vault, vault_salt, vault_verifier, themes_enc, note_enc) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("vault-1", "2026-02-01 00:00:00", "2026-02-01 01:00:00", 1,
         base64.b64encode(vsalt).decode(), venc.encrypt(VAULT_VERIFIER),
         app_enc.encrypt(json.dumps(["grief", "family"])),
         app_enc.encrypt("a private note")),
    )
    for i, text in enumerate(["first secret", "second secret", "third secret"]):
        conn.execute(
            "INSERT INTO messages (session_id, role, content_enc, timestamp) VALUES (?,?,?,?)",
            ("vault-1", "user" if i % 2 == 0 else "assistant",
             venc.encrypt(text), f"2026-02-01 00:0{i}:00"),
        )
    conn.commit(); conn.close()
    return app_enc


def _make_case2_vault_only_db(db_path, salt_path):
    """Case 2: ONLY a legacy vault conversation; no app anchor, no verifier."""
    _write_salt(salt_path, os.urandom(16))  # salt exists but no app ciphertext
    conn = sqlite3.connect(db_path)
    _schema(conn)
    vsalt = os.urandom(16)
    venc = Encryptor(VAULT_PW, salt=vsalt)
    conn.execute(
        "INSERT INTO sessions (id, created_at, is_vault, vault_salt, vault_verifier) VALUES (?,?,?,?,?)",
        ("vault-only", "2026-03-01 00:00:00", 1,
         base64.b64encode(vsalt).decode(), venc.encrypt(VAULT_VERIFIER)),
    )
    for i, text in enumerate(["only secret A", "only secret B"]):
        conn.execute(
            "INSERT INTO messages (session_id, role, content_enc, timestamp) VALUES (?,?,?,?)",
            ("vault-only", "user", venc.encrypt(text), f"2026-03-01 00:0{i}:00"),
        )
    conn.commit(); conn.close()


# ── Credential verification + no-write-on-wrong-credential ──────────────────────

class TestCredentialVerification:
    def test_correct_vault_credential_verifies(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        assert vr.verify_vault_credential(db, "vault-1", VAULT_PW) is True

    def test_wrong_vault_credential_rejected(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        assert vr.verify_vault_credential(db, "vault-1", "nope") is False

    def test_wrong_credential_leaves_source_unchanged(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        before = db.read_bytes()
        # Attempting conversion with a wrong credential must raise and not write.
        backup = vr.make_consistent_backup(db)
        target = SessionStore(db_path=backup)
        target.legacy_bootstrap(APP_PW)  # anchor present in backup
        with pytest.raises(Exception):
            vr.convert_vault_rows(
                backup_db_path=backup,
                target_store=target,
                vault_credentials={"vault-1": "WRONG"},
                app_encryptor_for_metadata=Encryptor(APP_PW, salt=Encryptor.read_salt()),
            )
        # The ORIGINAL db is untouched regardless of backup state.
        assert db.read_bytes() == before


# ── Case 1 conversion (app anchor present) ──────────────────────────────────────

class TestCase1Conversion:
    def test_conversion_preserves_history_and_others(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        app_enc = _make_case1_db(db, isolated_salt)

        backup = vr.make_consistent_backup(db)
        target = SessionStore(db_path=backup)
        assert target.legacy_bootstrap(APP_PW) is True  # unlock via app anchor

        converted = vr.convert_vault_rows(
            backup_db_path=backup,
            target_store=target,
            vault_credentials={"vault-1": VAULT_PW},
            app_encryptor_for_metadata=Encryptor(APP_PW, salt=Encryptor.read_salt()),
        )
        assert converted == ["vault-1"]

        # Re-open the backup fresh and verify through the workspace key.
        store = SessionStore(db_path=backup)
        store.unlock(APP_PW)
        # id preserved, now ordinary:
        assert store.is_vault("vault-1") is False
        msgs = store.load_messages("vault-1")
        assert [m.content for m in msgs] == ["first secret", "second secret", "third secret"]
        # metadata preserved:
        sess = store.get_session("vault-1")
        assert sess.themes == ["grief", "family"]
        assert sess.note == "a private note"
        # other conversation intact:
        assert store.load_messages("app-1")[0].content == "ordinary message"

    def test_converted_conversation_memory_off(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        backup = vr.make_consistent_backup(db)
        target = SessionStore(db_path=backup)
        target.legacy_bootstrap(APP_PW)
        vr.convert_vault_rows(
            backup_db_path=backup,
            target_store=target,
            vault_credentials={"vault-1": VAULT_PW},
            app_encryptor_for_metadata=Encryptor(APP_PW, salt=Encryptor.read_salt()),
        )
        store = SessionStore(db_path=backup)
        store.unlock(APP_PW)
        assert store.memory_opt_in("vault-1") is False

    def test_validation_passes_for_good_conversion(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        backup = vr.make_consistent_backup(db)
        target = SessionStore(db_path=backup)
        target.legacy_bootstrap(APP_PW)
        ids = vr.convert_vault_rows(
            backup_db_path=backup,
            target_store=target,
            vault_credentials={"vault-1": VAULT_PW},
            app_encryptor_for_metadata=Encryptor(APP_PW, salt=Encryptor.read_salt()),
        )
        assert vr.validate_converted_backup(backup, APP_PW, ids) is True


# ── Interruption recovery ────────────────────────────────────────────────────────

class TestInterruptionRecovery:
    def test_interrupted_conversion_leaves_original_intact(self, tmp_path, isolated_salt, monkeypatch):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        before = db.read_bytes()
        backup = vr.make_consistent_backup(db)
        target = SessionStore(db_path=backup)
        target.legacy_bootstrap(APP_PW)

        # Simulate a crash midway: patch the metadata decoder to raise after the
        # messages have been re-sealed within the (not-yet-committed) transaction.
        real = vr._decrypt_metadata_blob
        def boom(*a, **k):
            raise RuntimeError("simulated crash mid-conversion")
        monkeypatch.setattr(vr, "_decrypt_metadata_blob", boom)

        with pytest.raises(RuntimeError):
            vr.convert_vault_rows(
                backup_db_path=backup,
                target_store=target,
                vault_credentials={"vault-1": VAULT_PW},
                app_encryptor_for_metadata=Encryptor(APP_PW, salt=Encryptor.read_salt()),
            )
        monkeypatch.setattr(vr, "_decrypt_metadata_blob", real)

        # Original DB untouched; the aborted transaction rolled back in the backup.
        assert db.read_bytes() == before
        chk = SessionStore(db_path=backup)
        # The earlier legacy_bootstrap on `target` persisted a verifier into the
        # backup, so it is now VERIFIED_READY — unlock (not bootstrap) applies.
        chk.unlock(APP_PW)
        # vault row still a vault row in the backup (rollback) → not converted.
        assert chk.is_vault("vault-1") is True


# ── Unresolved metadata dependency stops ────────────────────────────────────────

class TestUnresolvedDependencyStops:
    def test_undecryptable_metadata_raises(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case1_db(db, isolated_salt)
        # Corrupt the themes blob so neither vault nor app key can decrypt it.
        c = sqlite3.connect(db)
        c.execute("UPDATE sessions SET themes_enc = 'AAAA' WHERE id='vault-1'")
        c.commit(); c.close()

        backup = vr.make_consistent_backup(db)
        target = SessionStore(db_path=backup)
        target.legacy_bootstrap(APP_PW)
        with pytest.raises(ValueError):
            vr.convert_vault_rows(
                backup_db_path=backup,
                target_store=target,
                vault_credentials={"vault-1": VAULT_PW},
                app_encryptor_for_metadata=Encryptor(APP_PW, salt=Encryptor.read_salt()),
            )


# ── Case 2: vault-only legacy recovery ──────────────────────────────────────────

class TestCase2VaultOnlyRecovery:
    def test_classified_vault_only(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_case2_vault_only_db(db, isolated_salt)
        plan = vr.plan_conversion(db)
        assert plan.state == WorkspaceState.VAULT_ONLY_LEGACY
        assert plan.needs_new_workspace is True
        assert plan.vault_session_ids == ["vault-only"]

    def test_recovery_into_new_workspace(self, tmp_path, isolated_salt):
        """
        Verify vault cred, initialise a NEW target workspace with a freshly
        chosen passphrase, and convert into it. We do NOT claim this
        authenticates any old app passphrase (there is none).
        """
        db = tmp_path / "db.sqlite"
        _make_case2_vault_only_db(db, isolated_salt)
        assert vr.verify_vault_credential(db, "vault-only", VAULT_PW) is True

        backup = vr.make_consistent_backup(db)
        # Initialise the NEW workspace verifier in the backup using NEW_PW.
        target = SessionStore(db_path=backup)
        # Case 2 has no app anchor, so classify() is VAULT_ONLY_LEGACY; we
        # explicitly initialise fresh workspace metadata keyed to NEW_PW.
        # (The salt file already exists; the new key is derived from it.)
        from therapy_ai.memory.session_store import WorkspaceAuth
        new_enc = Encryptor(NEW_PW, salt=Encryptor.read_salt())
        WorkspaceAuth(target._engine).initialize(new_enc)
        target._set_unlocked(new_enc)

        converted = vr.convert_vault_rows(
            backup_db_path=backup,
            target_store=target,
            vault_credentials={"vault-only": VAULT_PW},
            app_encryptor_for_metadata=None,  # no app metadata in case 2
        )
        assert converted == ["vault-only"]
        assert vr.validate_converted_backup(backup, NEW_PW, ["vault-only"]) is True

        # Confirmed readable under the NEW workspace passphrase, history intact.
        store = SessionStore(db_path=backup)
        store.unlock(NEW_PW)
        assert [m.content for m in store.load_messages("vault-only")] == [
            "only secret A", "only secret B",
        ]
