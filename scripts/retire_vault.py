#!/usr/bin/env python
"""
scripts/retire_vault.py

ONE-TIME, interactive, conversion-only retirement of legacy per-conversation
"vault" conversations into the single confidential workspace.

This is a thin CLI wrapper around therapy_ai.memory.vault_retirement. It is
NOT part of the running application and does nothing unless you run it
explicitly. There is deliberately NO deletion mode.

SAFETY / OPERATING RULES
  * STOP THE APP FIRST. Do not run this while the Streamlit app is open against
    the same database (we need exclusive, consistent access).
  * Credentials are prompted here, in YOUR terminal, via getpass. They are never
    echoed, never logged, and never passed on a command line.
  * The tool works on a SQLite-consistent BACKUP copy. Your original database is
    left untouched until you explicitly confirm replacement AFTER validation.
  * If anything cannot be verified or decrypted, the tool STOPS and explains,
    rather than guessing or discarding data.

Usage:
    python scripts/retire_vault.py --db data/sessions.db           # dry-run plan
    python scripts/retire_vault.py --db data/sessions.db --apply   # do it

Even with --apply, replacement of the original requires a final typed
confirmation after validation succeeds.
"""

from __future__ import annotations

import argparse
import getpass
import shutil
import sys
from pathlib import Path

from therapy_ai.memory import vault_retirement as vr
from therapy_ai.memory.session_store import (
    Encryptor,
    SessionStore,
    WorkspaceAuth,
    WorkspaceState,
)


def _prompt_secret(label: str) -> str:
    return getpass.getpass(f"{label}: ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-time vault retirement (conversion only).")
    ap.add_argument("--db", required=True, help="Path to the workspace SQLite DB.")
    ap.add_argument("--apply", action="store_true", help="Perform conversion (default: dry-run plan).")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 2

    plan = vr.plan_conversion(db_path)
    print("── Conversion plan ───────────────────────────────────────────")
    print(f"  DB:                 {plan.db_path}")
    print(f"  Workspace state:    {plan.state.value}")
    print(f"  Vault conversations: {len(plan.vault_session_ids)}")
    for sid in plan.vault_session_ids:
        print(f"      - {sid}")
    print(f"  Needs NEW workspace: {plan.needs_new_workspace}")
    print("──────────────────────────────────────────────────────────────")

    if not plan.vault_session_ids:
        print("No unretired vault conversations found. Nothing to do.")
        return 0

    if not args.apply:
        print("\nDry run only. Re-run with --apply to perform the conversion.")
        print("The application must be stopped before applying.")
        return 0

    print("\nEnsure the Therapy AI app is STOPPED before continuing.")
    if input("Type 'yes' to confirm the app is stopped: ").strip().lower() != "yes":
        print("Aborted.")
        return 1

    # 1) Consistent encrypted backup.
    backup = vr.make_consistent_backup(db_path)
    print(f"Consistent backup created: {backup}")

    # 2) Open the backup as the target workspace.
    target = SessionStore(db_path=backup)
    app_enc_for_metadata = None
    if plan.needs_new_workspace:
        # Case 2 — vault-only: user chooses a NEW workspace passphrase.
        print("\nThis workspace contains only locked (vault) conversations.")
        print("Choose a NEW workspace passphrase to hold the converted data.")
        new_pw = _prompt_secret("New workspace passphrase")
        confirm = _prompt_secret("Confirm new workspace passphrase")
        if new_pw != confirm or len(new_pw) < 8:
            print("Passphrases did not match or too short. Aborted.", file=sys.stderr)
            return 1
        new_enc = Encryptor(new_pw, salt=Encryptor.read_salt())
        WorkspaceAuth(target._engine).initialize(new_enc)
        target._set_unlocked(new_enc)
        workspace_pw = new_pw
    else:
        # Case 1 — existing workspace: verify the EXISTING workspace passphrase.
        workspace_pw = _prompt_secret("Existing workspace passphrase")
        state = target.classify()
        if state == WorkspaceState.VERIFIED_READY:
            target.unlock(workspace_pw)
        elif state == WorkspaceState.LEGACY_NO_VERIFIER:
            if not target.legacy_bootstrap(workspace_pw):
                print("Workspace passphrase did not verify. Aborted.", file=sys.stderr)
                return 1
        else:
            print(f"Unexpected workspace state '{state.value}'. Aborted.", file=sys.stderr)
            return 1
        app_enc_for_metadata = Encryptor(workspace_pw, salt=Encryptor.read_salt())

    # 3) Collect + verify each vault credential before converting.
    creds: dict[str, str] = {}
    for sid in plan.vault_session_ids:
        vpw = _prompt_secret(f"Vault passphrase for conversation {sid}")
        if not vr.verify_vault_credential(db_path, sid, vpw):
            print(f"Vault passphrase for {sid} did not verify. Aborted; nothing changed.",
                  file=sys.stderr)
            return 1
        creds[sid] = vpw

    # 4) Convert within the backup.
    try:
        converted = vr.convert_vault_rows(
            backup_db_path=backup,
            target_store=target,
            vault_credentials=creds,
            app_encryptor_for_metadata=app_enc_for_metadata,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Conversion stopped: {e}", file=sys.stderr)
        print("Original database unchanged. Backup copy left for inspection.")
        return 1

    # 5) Validate the converted backup.
    if not vr.validate_converted_backup(backup, workspace_pw, converted):
        print("Validation of the converted backup FAILED. Original unchanged.",
              file=sys.stderr)
        return 1
    print(f"Validation passed: {len(converted)} conversation(s) converted in backup.")

    # 6) Explicit replacement, recoverable.
    print("\nThe backup has been converted and validated.")
    print(f"  Original: {db_path}")
    print(f"  Converted backup: {backup}")
    if input("Type 'REPLACE' to swap the converted backup into place: ").strip() != "REPLACE":
        print("Not replaced. Your original is unchanged; the converted backup is kept.")
        return 0

    safety_copy = db_path.with_suffix(db_path.suffix + ".pre-retire")
    shutil.copy2(db_path, safety_copy)  # keep a recoverable copy of the original
    shutil.copy2(backup, db_path)
    print(f"Replaced. A copy of your original is at: {safety_copy}")
    print("Done. Verify in the app, then you may remove the backup files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
