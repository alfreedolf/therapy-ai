"""
tests/component/test_vault_sessions.py

Component tests for vault-mode sessions — a session locked with its own
passphrase, separate from the app passphrase. The security guarantee under
test: someone who knows the APP passphrase still cannot read a vault session
without its VAULT passphrase.

Uses a real encrypted SQLite DB (tmp) and the real AES-256-GCM + PBKDF2 path.
The autouse `isolated_salt` fixture in conftest.py redirects the shared salt
file to tmp — vault sessions use their own per-session salt regardless.
"""

from __future__ import annotations

import sqlite3

import pytest

from therapy_ai.memory.session_store import SessionStore

APP_PASS = "app-level-passphrase"
VAULT_PASS = "a-separate-vault-secret"


class TestVaultCreation:
    def test_vault_flag_and_salt_persisted(self, store):
        sid = store.create_session(mood_before=5, vault_passphrase=VAULT_PASS)
        assert store.is_vault(sid) is True

    def test_ordinary_session_is_not_vault(self, store):
        sid = store.create_session(mood_before=5)
        assert store.is_vault(sid) is False

    def test_empty_vault_passphrase_rejected(self, store):
        with pytest.raises(ValueError):
            store.create_session(vault_passphrase="")


class TestVaultRoundTrip:
    def test_message_roundtrips_with_vault_passphrase(self, store):
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        secret = "Something I only want behind the vault."
        store.add_message(sid, "user", secret, vault_passphrase=VAULT_PASS)

        loaded = store.load_messages(sid, vault_passphrase=VAULT_PASS)
        assert len(loaded) == 1
        assert loaded[0].content == secret


class TestVaultIsolation:
    """The core guarantee of vault mode."""

    def test_app_passphrase_cannot_read_vault(self, store):
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        store.add_message(sid, "user", "private", vault_passphrase=VAULT_PASS)

        # No vault passphrase supplied -> refuse, do not leak ciphertext.
        with pytest.raises(PermissionError):
            store.load_messages(sid)

    def test_wrong_vault_passphrase_cannot_read(self, store):
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        store.add_message(sid, "user", "private", vault_passphrase=VAULT_PASS)

        with pytest.raises(PermissionError):
            store.load_messages(sid, vault_passphrase="guessed-wrong")

    def test_wrong_vault_passphrase_cannot_write(self, store):
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        with pytest.raises(PermissionError):
            store.add_message(sid, "user", "x", vault_passphrase="wrong")

    def test_verify_passphrase_true_and_false(self, store):
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        assert store.verify_vault_passphrase(sid, VAULT_PASS) is True
        assert store.verify_vault_passphrase(sid, "nope") is False

    def test_second_store_same_app_pass_still_blocked(self, store, tmp_path):
        """
        A fresh SessionStore opened with the SAME app passphrase (simulating
        the normal app) must still not read the vault without its passphrase.
        """
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        store.add_message(sid, "user", "private", vault_passphrase=VAULT_PASS)

        other = SessionStore(passphrase=APP_PASS, db_path=tmp_path / "sessions.db")
        with pytest.raises(PermissionError):
            other.load_messages(sid)
        # ...but with the vault passphrase it opens.
        assert other.load_messages(sid, vault_passphrase=VAULT_PASS)[0].content == "private"


class TestVaultCiphertextAtRest:
    def test_vault_plaintext_absent_from_raw_db(self, store, tmp_path):
        secret = "VAULT_SENTINEL_PLAINTEXT_SHOULD_NOT_APPEAR_98765"
        sid = store.create_session(vault_passphrase=VAULT_PASS)
        store.add_message(sid, "user", secret, vault_passphrase=VAULT_PASS)

        raw = sqlite3.connect(tmp_path / "sessions.db")
        try:
            rows = raw.execute("SELECT content_enc FROM messages").fetchall()
            verifier_rows = raw.execute(
                "SELECT vault_verifier FROM sessions WHERE id = ?", (sid,)
            ).fetchall()
        finally:
            raw.close()

        assert rows
        for (stored,) in rows:
            assert secret not in stored
        # The verifier must not be the known plaintext either.
        assert verifier_rows and "therapy-ai-vault-ok" not in (verifier_rows[0][0] or "")


class TestNonVaultUnaffected:
    def test_ordinary_session_still_works_without_passphrase(self, store):
        sid = store.create_session(mood_before=5)
        store.add_message(sid, "user", "ordinary message")
        loaded = store.load_messages(sid)
        assert loaded[0].content == "ordinary message"
