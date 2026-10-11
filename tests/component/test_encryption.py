"""
tests/component/test_encryption.py

Component tests for the encrypted SessionStore — the core privacy guarantee of
the product. These use a real SQLite file (in tmp) and the real AES-256-GCM +
PBKDF2 encryption path. The salt file is redirected to tmp by the autouse
`isolated_salt` fixture in conftest.py.
"""

from __future__ import annotations

import sqlite3

import pytest

from therapy_ai.memory.session_store import SessionStore


class TestRoundTrip:
    def test_message_survives_encrypt_decrypt(self, store):
        sid = store.create_session(mood_before=6)
        original = "I've been struggling to talk to my father lately."
        store.add_message(sid, "user", original)

        loaded = store.load_messages(sid)
        assert len(loaded) == 1
        assert loaded[0].content == original

    def test_note_and_themes_survive_roundtrip(self, store):
        sid = store.create_session(mood_before=5)
        store.close_session(
            sid,
            mood_after=7,
            rating=4,
            themes=["family", "self-worth"],
            note="Felt lighter after talking it through.",
        )
        session = store.get_session(sid)
        assert session.themes == ["family", "self-worth"]
        assert session.note == "Felt lighter after talking it through."


class TestPassphraseIsolation:
    def test_wrong_passphrase_cannot_decrypt(self, tmp_path):
        """
        The central privacy guarantee: a store opened with a different
        passphrase (same salt) must NOT be able to read the messages.
        """
        db_path = tmp_path / "sessions.db"

        writer = SessionStore(passphrase="the-real-passphrase", db_path=db_path)
        sid = writer.create_session(mood_before=5)
        writer.add_message(sid, "user", "A private thought.")

        # Different passphrase, same DB + same (isolated) salt
        attacker = SessionStore(passphrase="a-guessed-passphrase", db_path=db_path)

        # Decryption must fail rather than silently return garbage
        with pytest.raises(Exception):
            attacker.load_messages(sid)

    def test_same_passphrase_reopens_successfully(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        secret = "Something I only want to tell myself."

        writer = SessionStore(passphrase="stable-passphrase", db_path=db_path)
        sid = writer.create_session(mood_before=5)
        writer.add_message(sid, "user", secret)

        reopened = SessionStore(passphrase="stable-passphrase", db_path=db_path)
        loaded = reopened.load_messages(sid)
        assert loaded[0].content == secret


class TestCiphertextAtRest:
    def test_plaintext_not_present_in_raw_db(self, store, tmp_path):
        """
        Read the SQLite file directly (bypassing the store) and confirm the
        plaintext never appears — only ciphertext is persisted.
        """
        secret = "SENTINEL_PLAINTEXT_SHOULD_NOT_APPEAR_12345"
        sid = store.create_session(mood_before=5)
        store.add_message(sid, "user", secret)

        db_file = tmp_path / "sessions.db"
        raw = sqlite3.connect(db_file)
        try:
            rows = raw.execute("SELECT content_enc FROM messages").fetchall()
        finally:
            raw.close()

        assert rows, "expected at least one stored message"
        for (stored,) in rows:
            assert secret not in stored  # ciphertext only


class TestDataSovereignty:
    def test_delete_session_removes_messages(self, store):
        sid = store.create_session(mood_before=5)
        store.add_message(sid, "user", "temporary")
        store.delete_session(sid)

        with pytest.raises(ValueError):
            store.get_session(sid)

    def test_wipe_all_data_clears_sessions(self, store):
        sid = store.create_session(mood_before=5)
        store.add_message(sid, "user", "will be wiped")

        store.wipe_all_data()

        # After wipe, tables are recreated empty
        assert store.list_sessions() == []
