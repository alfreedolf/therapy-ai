"""
tests/component/test_workspace_auth.py

Acceptance tests for VERIFIED workspace access (R1–R4, R13, R16) and
locked-state enforcement + lifecycle revocation (R12).

All against synthetic SQLite workspaces in tmp. The autouse `isolated_salt`
fixture redirects the shared salt file to tmp.
"""

from __future__ import annotations

import base64
import sqlite3

import pytest

from therapy_ai.memory.session_store import (
    AuthenticationError,
    Encryptor,
    SessionStore,
    WorkspaceLockedError,
    WorkspaceState,
    WorkspaceStateError,
)

PW = "a-strong-workspace-pass"


# ── Initialization vs unlock distinction (R1) ──────────────────────────────────

class TestInitializeVsUnlock:
    def test_fresh_workspace_is_empty(self, tmp_path):
        s = SessionStore(db_path=tmp_path / "db.sqlite")
        assert s.classify() == WorkspaceState.EMPTY
        assert s.is_unlocked is False

    def test_initialize_sets_verifier_and_unlocks(self, tmp_path):
        s = SessionStore(db_path=tmp_path / "db.sqlite")
        s.initialize(PW)
        assert s.is_unlocked is True
        assert s.classify() == WorkspaceState.VERIFIED_READY

    def test_initialize_refuses_on_already_initialised(self, tmp_path):
        s = SessionStore(db_path=tmp_path / "db.sqlite")
        s.initialize(PW)
        s2 = SessionStore(db_path=tmp_path / "db.sqlite")
        with pytest.raises(WorkspaceStateError):
            s2.initialize(PW)

    def test_unlock_with_correct_passphrase(self, tmp_path):
        db = tmp_path / "db.sqlite"
        SessionStore(db_path=db).initialize(PW)
        reopened = SessionStore(db_path=db)
        reopened.unlock(PW)
        assert reopened.is_unlocked is True


# ── Wrong credentials fail closed (R2) ──────────────────────────────────────────

class TestWrongCredentialsFailClosed:
    def test_wrong_passphrase_raises_and_stays_locked(self, tmp_path):
        db = tmp_path / "db.sqlite"
        SessionStore(db_path=db).initialize(PW)
        s = SessionStore(db_path=db)
        with pytest.raises(AuthenticationError):
            s.unlock("wrong-pass")
        assert s.is_unlocked is False

    def test_wrong_passphrase_permits_no_writes(self, tmp_path):
        db = tmp_path / "db.sqlite"
        SessionStore(db_path=db).initialize(PW)
        s = SessionStore(db_path=db)
        with pytest.raises(AuthenticationError):
            s.unlock("wrong-pass")
        # Protected op must be refused while locked.
        with pytest.raises(WorkspaceLockedError):
            s.create_session(mood_before=5)

    def test_wrong_passphrase_does_not_reinitialise_metadata(self, tmp_path):
        db = tmp_path / "db.sqlite"
        SessionStore(db_path=db).initialize(PW)
        # Capture verifier before the bad attempt.
        before = sqlite3.connect(db).execute(
            "SELECT verifier FROM workspace_meta WHERE id=1"
        ).fetchone()[0]
        s = SessionStore(db_path=db)
        with pytest.raises(AuthenticationError):
            s.unlock("wrong-pass")
        after = sqlite3.connect(db).execute(
            "SELECT verifier FROM workspace_meta WHERE id=1"
        ).fetchone()[0]
        assert before == after  # unchanged


# ── Missing / corrupt metadata (R3) ─────────────────────────────────────────────

class TestFailClosedStates:
    def test_corrupt_verifier_refuses_unlock(self, tmp_path):
        db = tmp_path / "db.sqlite"
        SessionStore(db_path=db).initialize(PW)
        # Corrupt the verifier.
        c = sqlite3.connect(db)
        c.execute("UPDATE workspace_meta SET verifier='not-base64!!' WHERE id=1")
        c.commit(); c.close()
        s = SessionStore(db_path=db)
        assert s.classify() == WorkspaceState.CORRUPT
        with pytest.raises(WorkspaceStateError):
            s.unlock(PW)

    def test_missing_salt_with_verifier_fails_closed(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        SessionStore(db_path=db).initialize(PW)
        # Remove the salt file → key cannot be derived.
        from pathlib import Path
        Path(isolated_salt).unlink()
        s = SessionStore(db_path=db)
        with pytest.raises(WorkspaceStateError):
            s.unlock(PW)
        assert s.is_unlocked is False


# ── Legacy bootstrap (R4) ────────────────────────────────────────────────────────

def _make_legacy_app_db(db_path, salt_path, passphrase: str) -> None:
    """
    Build a legacy workspace: ordinary (non-vault) encrypted content but NO
    workspace_meta verifier. Simulates a DB created before verified-access.
    """
    salt = base64.b64decode(open(salt_path, "rb").read()) if __import__("os").path.exists(salt_path) else __import__("os").urandom(16)
    # Ensure the salt file holds exactly this salt.
    with open(salt_path, "wb") as f:
        f.write(base64.b64encode(salt))
    enc = Encryptor(passphrase, salt=salt)
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
    c.execute("INSERT INTO sessions (id, created_at, is_vault) VALUES ('leg-1','2026-01-01 00:00:00',0)")
    c.execute(
        "INSERT INTO messages (session_id, role, content_enc, timestamp) VALUES (?,?,?,?)",
        ("leg-1", "user", enc.encrypt("legacy secret"), "2026-01-01 00:00:01"),
    )
    c.commit(); c.close()


class TestLegacyBootstrap:
    def test_state_is_legacy_no_verifier(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_legacy_app_db(db, isolated_salt, PW)
        s = SessionStore(db_path=db)
        assert s.classify() == WorkspaceState.LEGACY_NO_VERIFIER

    def test_bootstrap_with_correct_pass_succeeds_and_persists_verifier(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_legacy_app_db(db, isolated_salt, PW)
        s = SessionStore(db_path=db)
        assert s.legacy_bootstrap(PW) is True
        assert s.is_unlocked is True
        # Verifier now present → next open is ordinary unlock.
        s2 = SessionStore(db_path=db)
        assert s2.classify() == WorkspaceState.VERIFIED_READY

    def test_bootstrap_with_wrong_pass_fails_and_persists_nothing(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_legacy_app_db(db, isolated_salt, PW)
        s = SessionStore(db_path=db)
        assert s.legacy_bootstrap("wrong") is False
        assert s.is_unlocked is False
        # No verifier fabricated.
        row = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM workspace_meta"
        ).fetchone()[0]
        assert row == 0

    def test_unlock_refuses_legacy_state(self, tmp_path, isolated_salt):
        db = tmp_path / "db.sqlite"
        _make_legacy_app_db(db, isolated_salt, PW)
        s = SessionStore(db_path=db)
        with pytest.raises(WorkspaceStateError):
            s.unlock(PW)


# ── Locked-state enforcement + lifecycle revocation (R12) ───────────────────────

class TestLockedStateEnforcement:
    def test_protected_ops_require_unlock(self, tmp_path):
        s = SessionStore(db_path=tmp_path / "db.sqlite")
        for op in (
            lambda: s.create_session(mood_before=5),
            lambda: s.load_messages("x"),
        ):
            with pytest.raises(WorkspaceLockedError):
                op()

    def test_lock_revokes_access(self, tmp_path):
        s = SessionStore(db_path=tmp_path / "db.sqlite")
        s.initialize(PW)
        sid = s.create_session(mood_before=5)
        s.lock()
        assert s.is_unlocked is False
        with pytest.raises(WorkspaceLockedError):
            s.add_message(sid, "user", "x")

    def test_lifecycle_token_changes_on_reunlock(self, tmp_path):
        db = tmp_path / "db.sqlite"
        s = SessionStore(db_path=db)
        s.initialize(PW)
        tok1 = s.lifecycle_token
        s.lock()
        s.unlock(PW)
        tok2 = s.lifecycle_token
        assert tok1 and tok2 and tok1 != tok2

    def test_late_result_from_revoked_lifecycle_refused(self, tmp_path):
        db = tmp_path / "db.sqlite"
        s = SessionStore(db_path=db)
        s.initialize(PW)
        sid = s.create_session(mood_before=5)
        old_tok = s.lifecycle_token
        # Simulate: a generation began under old_tok, user locks + re-unlocks.
        s.lock()
        s.unlock(PW)
        # Persisting the stale result must be refused even though we're unlocked.
        with pytest.raises(WorkspaceLockedError):
            s.add_message(sid, "assistant", "late token", lifecycle_token=old_tok)
        # The current token works.
        s.add_message(sid, "assistant", "fresh", lifecycle_token=s.lifecycle_token)
        assert s.load_messages(sid)[-1].content == "fresh"
