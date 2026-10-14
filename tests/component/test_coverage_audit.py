"""
tests/component/test_coverage_audit.py

Maps three previously-claimed guarantees to EXACT assertions (handoff section 5,
item 5). These pin down the honest boundary of each claim rather than relying on
a related test's existence.

Questions answered:
  Q1. Does quarantine leave plaintext summaries/vectors on disk?
      -> YES. Quarantine RENAMES the legacy table; the plaintext ROWS persist
         under the quarantined name. The guarantee is "not reused by retrieval",
         NOT "plaintext removed". This test documents that precisely.
  Q2. Does a dry run (plan_conversion) cause writes?
      -> plan_conversion itself only SELECTs, BUT it constructs a SessionStore,
         whose __init__ runs additive schema migration / create_all. On a DB
         missing columns/tables that constructor CAN write (ADD COLUMN / CREATE
         TABLE). So "dry run" is read-only only for an already-current schema.
  Q3. Does replacement/backup safely handle active connections + interruption?
      -> make_consistent_backup uses SQLite's online backup API and leaves the
         ORIGINAL untouched; conversion operates on the backup copy; an
         interrupted conversion leaves the original intact (covered in
         test_vault_retirement; re-asserted here at the backup-isolation level).

Synthetic fixtures only.
"""

from __future__ import annotations

import sqlite3

from therapy_ai.memory import vault_retirement as vr
from therapy_ai.memory.memory_retriever import MemoryRetriever

TEST_PASSPHRASE = "correct-horse-battery-staple"


# ── Q1: quarantine leaves plaintext rows on disk (renamed, not removed) ─────────

class TestQuarantineHonesty:
    def test_quarantine_renames_but_retains_plaintext_rows(self, store):
        db = store._engine.url.database
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE memory_chunks (session_id TEXT PRIMARY KEY, summary TEXT)")
        c.execute("INSERT INTO memory_chunks VALUES ('old', 'leaky plaintext summary')")
        c.commit()
        c.close()

        MemoryRetriever(db, encryptor_provider=store.encryptor_provider())

        conn = sqlite3.connect(db)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            assert "memory_chunks" not in tables
            assert "_legacy_plaintext_memory_chunks" in tables
            # HONEST BOUNDARY: the plaintext row still exists under the new name.
            rows = conn.execute(
                "SELECT summary FROM _legacy_plaintext_memory_chunks"
            ).fetchall()
            assert rows == [("leaky plaintext summary",)]
        finally:
            conn.close()

    def test_quarantined_plaintext_is_not_used_by_retrieval(self, store):
        """The retained plaintext is NOT read back into the live index."""
        db = store._engine.url.database
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE memory_chunks (session_id TEXT PRIMARY KEY, summary TEXT)")
        c.execute("INSERT INTO memory_chunks VALUES ('old', 'father work sleep leak')")
        c.commit()
        c.close()

        retr = MemoryRetriever(db, encryptor_provider=store.encryptor_provider())
        # No memory_sources were created, so retrieval sees nothing from the
        # quarantined plaintext.
        assert retr.retrieve("father", top_k=5) == []


# ── Q2: dry-run write behavior through the store constructor ────────────────────

class TestDryRunWriteBoundary:
    def test_plan_conversion_on_current_schema_does_not_change_table_set(self, store):
        """
        On an already-current schema, plan_conversion performs no schema changes
        (the table set is unchanged). This is the clean 'dry run is read-only'
        case; the DB already has every ORM-declared column/table.
        """
        db = store._engine.url.database

        def _snapshot():
            conn = sqlite3.connect(db)
            try:
                return {
                    r[0]: r[1] for r in conn.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                conn.close()

        before = _snapshot()
        vr.plan_conversion(db)
        after = _snapshot()
        assert before == after  # no schema writes on a current-schema DB

    def test_plan_conversion_constructs_store_which_can_migrate(self, tmp_path):
        """
        FINDING: plan_conversion builds a SessionStore, whose __init__ runs
        additive migration/create_all. On a DB MISSING tables, 'dry run' WILL
        create them. Documented so 'dry run = zero writes' isn't overclaimed.
        """
        db = tmp_path / "bare.sqlite"
        # A completely empty file: no tables at all.
        sqlite3.connect(db).close()

        vr.plan_conversion(db)  # constructs SessionStore -> create_all writes

        conn = sqlite3.connect(db)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        # The store constructor created its schema during the "dry run".
        assert "sessions" in tables and "messages" in tables


# ── Q3: backup isolation — original untouched by backup creation ────────────────

class TestBackupIsolation:
    def test_make_consistent_backup_leaves_original_unchanged(self, store):
        db = store._engine.url.database
        sid = store.create_session(mood_before=5)
        store.add_message(sid, "user", "original content")

        def _digest(path):
            conn = sqlite3.connect(path)
            try:
                return conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(LENGTH(content_enc)),0) FROM messages"
                ).fetchone()
            finally:
                conn.close()

        before = _digest(db)
        backup = vr.make_consistent_backup(db)
        after = _digest(db)
        assert before == after                 # original unchanged
        assert backup.exists() and str(backup) != str(db)  # separate artifact
