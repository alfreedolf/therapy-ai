"""
therapy_ai/memory/vault_retirement.py

ONE-TIME, conversion-only retirement of legacy per-conversation "vault" records
into the single confidential workspace. There is intentionally NO deletion
mechanism here (deletion, if ever wanted, is authorised separately).

Guarantees
──────────
* Verifies the vault credential against the row's existing verifier/ciphertext
  BEFORE touching anything. A wrong vault credential leaves the source byte-for
  -byte unchanged.
* Decodes the ACTUAL source format (per-row vault salt + AES-GCM) and re-encrypts
  into the CHOSEN target workspace format (the target store's current Encryptor),
  rather than assuming the target is "the old app key".
* Legacy notes/themes: validated against their actual stored format. In the
  retired design these were sealed with the APP key, not the vault key; the tool
  detects which key decrypts them and preserves them. If neither key decrypts a
  notes/themes blob, it STOPS and reports an unresolved dependency rather than
  guessing or discarding.
* Preserves the conversation id and per-message timestamps/order.
* Sets memory contribution OFF on the converted conversation.
* No plaintext intermediate files: all decryption happens in memory; the only
  on-disk artifact is a consistent ENCRYPTED SQLite backup copy.
* Crash-safe: operates on a validated backup copy; the original is preserved
  until the converted copy is validated; replacement is explicit and recoverable.

This module performs NO interactive prompting and holds NO credentials beyond
the function call. The CLI wrapper (scripts/retire_vault.py) is responsible for
prompting in the user's own terminal.

Everything here is exercised on SYNTHETIC fixtures only.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from therapy_ai.memory.session_store import (
    Encryptor,
    SessionStore,
    WorkspaceState,
)

logger = logging.getLogger(__name__)

_VAULT_VERIFIER_PLAINTEXT = "therapy-ai-vault-ok"  # the OLD per-vault token


# ── Result / plan types ──────────────────────────────────────────────────────

@dataclass
class ConversionPlan:
    db_path: Path
    vault_session_ids: list[str]
    state: WorkspaceState
    needs_new_workspace: bool  # True for VAULT_ONLY_LEGACY


@dataclass
class ConversionResult:
    converted_ids: list[str]
    backup_path: Path
    validated: bool


# ── Backup / consistency ──────────────────────────────────────────────────────

def make_consistent_backup(db_path: str | Path) -> Path:
    """
    Produce a SQLite-consistent ENCRYPTED backup copy of the workspace DB.

    This does NOT rely on a filesystem copy of a live DB (which can be
    inconsistent with its -wal/-shm). It uses SQLite's online Backup API, which
    captures a transactionally consistent snapshot even if other connections
    exist. The caller must still ensure the application is stopped / holds
    appropriate exclusive access before replacement; this function only creates
    the snapshot.

    Returns the backup path (db_path + '.retire-backup').
    """
    db_path = Path(db_path)
    backup_path = db_path.with_suffix(db_path.suffix + ".retire-backup")
    src = sqlite3.connect(str(db_path))
    try:
        # Fold any WAL into the main DB file for a clean snapshot. If not in WAL
        # or the checkpoint cannot run, the online backup API is still consistent.
        with contextlib.suppress(sqlite3.OperationalError):
            src.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)  # transactionally consistent online backup
        finally:
            dst.close()
    finally:
        src.close()
    logger.info("Consistent encrypted backup written: %s", backup_path)
    return backup_path


# ── Planning ──────────────────────────────────────────────────────────────────

def plan_conversion(db_path: str | Path) -> ConversionPlan:
    """
    Inspect the DB (no credentials) and describe what conversion would do.
    """
    store = SessionStore(db_path=db_path)
    state = store.classify()
    vault_ids = store.list_vault_session_ids()
    needs_new = state == WorkspaceState.VAULT_ONLY_LEGACY
    return ConversionPlan(
        db_path=Path(db_path),
        vault_session_ids=vault_ids,
        state=state,
        needs_new_workspace=needs_new,
    )


# ── Low-level vault decode (actual source format) ────────────────────────────

def _vault_encryptor_for_row(conn, session_id: str, vault_passphrase: str) -> Encryptor:
    row = conn.execute(
        "SELECT vault_salt FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"Session {session_id} has no vault salt; not a vault row.")
    salt = base64.b64decode(row[0])
    return Encryptor(vault_passphrase, salt=salt)


def verify_vault_credential(db_path, session_id: str, vault_passphrase: str) -> bool:
    """
    True iff `vault_passphrase` decrypts the row's existing vault verifier to the
    known token. Never raises on a wrong credential.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT vault_verifier FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return False
        try:
            venc = _vault_encryptor_for_row(conn, session_id, vault_passphrase)
            return venc.decrypt(row[0]) == _VAULT_VERIFIER_PLAINTEXT
        except Exception:
            return False
    finally:
        conn.close()


def _decrypt_metadata_blob(blob: str, venc: Encryptor, app_enc: Encryptor | None):
    """
    Legacy themes/note blob: validated against its ACTUAL format. Try the vault
    key first, then the app key (the retired design sealed these with the app
    key). Returns (plaintext, which_key). Raises if neither decrypts — an
    unresolved dependency the caller must surface, not guess.
    """
    for key_name, enc in (("vault", venc), ("app", app_enc)):
        if enc is None:
            continue
        try:
            return enc.decrypt(blob), key_name
        except Exception:
            continue
    raise ValueError(
        "A notes/themes blob could not be decrypted with the vault key or the "
        "workspace app key. Unresolved encryption dependency — stopping rather "
        "than guessing or discarding."
    )


# ── Conversion (operates on the BACKUP copy) ─────────────────────────────────

def convert_vault_rows(
    *,
    backup_db_path: str | Path,
    target_store: SessionStore,
    vault_credentials: dict[str, str],
    app_encryptor_for_metadata: Encryptor | None,
) -> list[str]:
    """
    Convert vault rows IN PLACE within `backup_db_path` so their content becomes
    sealed with `target_store`'s workspace key, then clear the vault markers.

    * `target_store` MUST be unlocked; its active Encryptor is the target format.
    * `vault_credentials` maps session_id -> vault passphrase (already verified).
    * `app_encryptor_for_metadata` is the ORIGINAL app-key encryptor if the
      legacy notes/themes were app-sealed (Case 1); None for vault-only (Case 2),
      in which case any undecryptable metadata triggers a stop.

    Returns the list of converted session ids. Raises on any unresolved
    dependency, leaving the backup copy to be discarded by the caller (original
    untouched).
    """
    target_enc = target_store.active_encryptor
    if target_enc is None:
        raise RuntimeError("Target workspace must be unlocked before conversion.")

    conn = sqlite3.connect(str(backup_db_path))
    converted: list[str] = []
    try:
        conn.execute("BEGIN")
        for sid, vpass in vault_credentials.items():
            # Re-verify against the backup copy's own verifier (defense in depth).
            vrow = conn.execute(
                "SELECT vault_verifier FROM sessions WHERE id = ? AND is_vault = 1",
                (sid,),
            ).fetchone()
            if vrow is None:
                raise ValueError(f"{sid} is not an unretired vault row in the backup.")
            venc = _vault_encryptor_for_row(conn, sid, vpass)
            if venc.decrypt(vrow[0]) != _VAULT_VERIFIER_PLAINTEXT:
                raise ValueError(f"Vault credential for {sid} failed verification.")

            # Re-seal each message: decrypt (vault key, in memory) -> encrypt (target key).
            msgs = conn.execute(
                "SELECT id, content_enc FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (sid,),
            ).fetchall()
            for mid, content_enc in msgs:
                plaintext = venc.decrypt(content_enc)       # in-memory only
                resealed = target_enc.encrypt(plaintext)
                conn.execute(
                    "UPDATE messages SET content_enc = ? WHERE id = ?", (resealed, mid)
                )

            # Re-seal metadata (themes/note) per its ACTUAL key.
            srow = conn.execute(
                "SELECT themes_enc, note_enc FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            themes_enc, note_enc = srow
            new_themes = new_note = None
            if themes_enc:
                plain, _ = _decrypt_metadata_blob(themes_enc, venc, app_encryptor_for_metadata)
                json.loads(plain)  # validate it is a JSON list as expected
                new_themes = target_enc.encrypt(plain)
            if note_enc:
                plain, _ = _decrypt_metadata_blob(note_enc, venc, app_encryptor_for_metadata)
                new_note = target_enc.encrypt(plain)

            # Clear vault markers; preserve id/history; memory contribution OFF.
            conn.execute(
                """UPDATE sessions
                   SET is_vault = 0, vault_salt = NULL, vault_verifier = NULL,
                       memory_opt_in = 0, themes_enc = ?, note_enc = ?
                   WHERE id = ?""",
                (new_themes, new_note, sid),
            )
            converted.append(sid)

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    logger.info("Converted %d vault row(s) in backup copy.", len(converted))
    return converted


# ── Validation ────────────────────────────────────────────────────────────────

def validate_converted_backup(
    backup_db_path: str | Path,
    workspace_passphrase: str,
    expected_ids: list[str],
) -> bool:
    """
    Open the converted BACKUP with the target workspace passphrase and confirm:
      * no unretired vault rows remain,
      * each converted conversation's messages decrypt with the workspace key,
      * converted conversations have memory contribution OFF.
    Returns True only if all checks pass.
    """
    store = SessionStore(db_path=backup_db_path)
    state = store.classify()
    if state == WorkspaceState.EMPTY:
        return False
    # The backup should now be a verified or legacy-bootstrappable workspace.
    try:
        if state == WorkspaceState.VERIFIED_READY:
            store.unlock(workspace_passphrase)
        elif state == WorkspaceState.LEGACY_NO_VERIFIER:
            if not store.legacy_bootstrap(workspace_passphrase):
                return False
        else:
            return False
    except Exception:
        return False

    if store.list_vault_session_ids():
        return False
    for sid in expected_ids:
        try:
            msgs = store.load_messages(sid)   # workspace-key decrypt must work
        except Exception:
            return False
        if store.memory_opt_in(sid) is not False:
            return False
        if not msgs:
            return False
    return True
