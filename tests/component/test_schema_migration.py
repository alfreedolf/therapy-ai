"""
tests/component/test_schema_migration.py

Regression test for the additive schema auto-migration in SessionStore.

A database created before the vault columns existed must upgrade itself when
opened — adding the missing columns without destroying existing rows — so that
create_session() no longer fails with "table sessions has no column named
is_vault".
"""

from __future__ import annotations

import sqlite3

from therapy_ai.memory.session_store import SessionStore


def _make_old_schema_db(db_path) -> None:
    """Create a pre-vault 'sessions' table with one legacy row."""
    c = sqlite3.connect(db_path)
    c.execute(
        """CREATE TABLE sessions (
            id TEXT PRIMARY KEY, created_at DATETIME, ended_at DATETIME,
            mood_before INTEGER, mood_after INTEGER, rating INTEGER,
            themes_enc TEXT, note_enc TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            content_enc TEXT, timestamp DATETIME, tokens INTEGER, is_flagged BOOLEAN
        )"""
    )
    c.execute("INSERT INTO sessions (id, created_at) VALUES ('legacy-1', '2026-01-01 00:00:00')")
    c.commit()
    c.close()


class TestAdditiveMigration:
    def test_vault_columns_added_on_open(self, tmp_path):
        db = tmp_path / "old.db"
        _make_old_schema_db(db)

        SessionStore(passphrase="pw", db_path=db)  # opening triggers migration

        cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(sessions)")}
        assert {"is_vault", "vault_salt", "vault_verifier"} <= cols

    def test_existing_rows_preserved(self, tmp_path):
        db = tmp_path / "old.db"
        _make_old_schema_db(db)

        SessionStore(passphrase="pw", db_path=db)

        count = sqlite3.connect(db).execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 1  # the legacy row survived

    def test_create_session_works_after_migration(self, tmp_path):
        db = tmp_path / "old.db"
        _make_old_schema_db(db)

        store = SessionStore(passphrase="pw", db_path=db)
        # The operation that originally failed:
        sid = store.create_session(mood_before=5)
        assert sid
        # And a vault session too:
        vid = store.create_session(mood_before=5, vault_passphrase="secret")
        assert store.is_vault(vid) is True

    def test_migration_is_idempotent(self, tmp_path):
        db = tmp_path / "old.db"
        _make_old_schema_db(db)

        # Opening twice must not error (columns already present on 2nd open).
        SessionStore(passphrase="pw", db_path=db)
        SessionStore(passphrase="pw", db_path=db)
