"""
tests/component/test_schema_migration.py

Regression test for the additive schema auto-migration in SessionStore.

Post vault-retirement the migration must additively add any ORM-declared column
missing from a legacy DB — including `sessions.memory_opt_in` and the
`workspace_meta` table — WITHOUT destroying existing rows and WITHOUT dropping
the retained (legacy) vault columns.
"""

from __future__ import annotations

import sqlite3

from therapy_ai.memory.session_store import SessionStore, WorkspaceState


def _make_pre_memory_schema_db(db_path) -> None:
    """
    Create a 'sessions' table from the vault era (has vault columns, but NO
    memory_opt_in and NO workspace_meta) plus one legacy row.
    """
    c = sqlite3.connect(db_path)
    c.execute(
        """CREATE TABLE sessions (
            id TEXT PRIMARY KEY, created_at DATETIME, ended_at DATETIME,
            mood_before INTEGER, mood_after INTEGER, rating INTEGER,
            themes_enc TEXT, note_enc TEXT,
            is_vault BOOLEAN DEFAULT 0, vault_salt TEXT, vault_verifier TEXT
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
    def test_memory_and_meta_added_on_open(self, tmp_path):
        db = tmp_path / "old.db"
        _make_pre_memory_schema_db(db)

        SessionStore(db_path=db)  # opening triggers migration

        raw = sqlite3.connect(db)
        session_cols = {r[1] for r in raw.execute("PRAGMA table_info(sessions)")}
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        # New column added additively…
        assert "memory_opt_in" in session_cols
        # …retained vault columns NOT dropped…
        assert {"is_vault", "vault_salt", "vault_verifier"} <= session_cols
        # …and the workspace_meta table created.
        assert "workspace_meta" in tables

    def test_existing_rows_preserved(self, tmp_path):
        db = tmp_path / "old.db"
        _make_pre_memory_schema_db(db)

        SessionStore(db_path=db)

        count = sqlite3.connect(db).execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 1  # legacy row survived

    def test_migrated_db_is_legacy_until_bootstrapped(self, tmp_path):
        """
        A migrated legacy DB has no verifier, so it classifies as a legacy state
        rather than magically being verified. (The 'legacy-1' row has no
        encrypted message anchor, so it is VAULT_ONLY_LEGACY-style ambiguous:
        here, no vault rows + no app anchor → CORRUPT/ambiguous, requiring a
        decision. We assert it is NOT silently VERIFIED_READY.)
        """
        db = tmp_path / "old.db"
        _make_pre_memory_schema_db(db)
        store = SessionStore(db_path=db)
        assert store.classify() != WorkspaceState.VERIFIED_READY

    def test_migration_is_idempotent(self, tmp_path):
        db = tmp_path / "old.db"
        _make_pre_memory_schema_db(db)
        SessionStore(db_path=db)
        SessionStore(db_path=db)  # second open must not error
