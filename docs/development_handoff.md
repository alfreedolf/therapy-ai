# Development Handoff — verified workspace refactor & vault retirement

Branch: `refactor/retire-vault`. **Unfinished implementation checkpoint.**
This is NOT a merge approval and NOT a declaration that all security
requirements are satisfied. A focused final verification pass is still pending
(see end).

> No credentials, private conversation content, local sensitive paths, or
> runtime storage artifacts are included in this document by design.

---

## 1. Current architecture & status

Single confidential encrypted workspace (one SQLite DB). The per-conversation
"vault" tier has been retired as a product feature; legacy vault rows are
guarded in a transition state and migrated by a one-time conversion tool.

Key modules changed:
- `therapy_ai/memory/session_store.py`
  - `WorkspaceMeta` (versioned KDF/verifier metadata) + `WorkspaceAuth`.
  - Distinct operations: `initialize()`, `unlock()`, `legacy_bootstrap()`,
    `open_verified()`; `classify()` → `WorkspaceState`
    (EMPTY / VERIFIED_READY / LEGACY_NO_VERIFIER / VAULT_ONLY_LEGACY / CORRUPT).
  - Fail-closed on wrong passphrase / missing-corrupt verifier / missing salt.
  - Lock lifecycle: `lock()`, lifecycle token, `check_lifecycle()`,
    `_require_unlocked()`.
  - Transition guard: `is_vault` retained; vault rows excluded from
    `list_sessions`/`mood_series`, and raise `RetirementRequiredError` on
    get/load/close/opt-in. Vault content never decrypted with the workspace key.
    Vault columns retained (NOT dropped).
  - `memory_opt_in` column (contribution consent, default OFF).
- `therapy_ai/memory/memory_retriever.py`
  - Encrypted source summaries at rest (`memory_sources`); in-memory retrieval
    index rebuilt from decrypted sources; legacy plaintext `memory_chunks`
    quarantined (renamed), not reused.
- `therapy_ai/memory/context_builder.py`
  - Contribution (per-session `memory_opt_in`) vs receiving (global recall
    config) kept distinct; current conversation's own history always included.
- `therapy_ai/memory/vault_retirement.py` + `scripts/retire_vault.py`
  - Conversion-only, inert one-time tool (no deletion mechanism).
- `therapy_ai/ui/app.py`, `ui/pages/chat.py`, `ui/pages/progress.py`
  - Vault UX removed; verified init/unlock gate; lock-and-clear + inactivity
    auto-lock; buffered stream displayed/persisted only under a valid lifecycle;
    retirement-required screen; per-conversation memory opt-in toggle.

## 2. Stages completed
- A. Reconciliation + traceability matrix (`docs/requirements_matrix.md`).
- B. Verified workspace auth + unretired-record transition guard.
- C. Protected derived storage + authoritative memory policy.
- D. Vault UI/runtime removal with transition guard preserved.
- E. One-time conversion tool + regression tests (synthetic only).

## 3. Tests / checks actually run (last results)
- Full suite: **177 passed, 2 xfailed** (`.venv/bin/python -m pytest -o addopts="" -q`).
- `ruff check` on new modules: clean.
- UI import smoke test: `app`, `chat`, `progress` import; vault UI gone;
  retirement UI present.
- Retirement CLI dry-run against a SYNTHETIC vault-only DB: classified
  correctly, refused to act without `--apply`.
- NOTE: default `pytest` addopts include `--cov`; coverage plugin may not be
  installed in every env. Runs above used `-o addopts=""`.

## 4. Known gaps & deferred requirements
- **R14 Password change — DEFERRED.** Not implemented; no unsafe UI control
  exposed. Re-keying would require re-encrypting all app-key ciphertext +
  rewriting the verifier.
- **R18 Sensitive metadata partly plaintext.** Encrypted: message content,
  notes, themes, memory source summaries. NOT encrypted: `mood_before`,
  `mood_after`, `rating`, `is_flagged`, timestamps, vault marker columns.
- **Legacy plaintext-index handling** needs verification beyond the single
  quarantine unit test (behavior on real-world shapes, idempotency across opens).
- **UI lifecycle / inactivity-lock** behaviors are implemented and covered at
  the store level, but need focused UI-level tests (no-post-lock display,
  inactivity expiration + default, retriever/encryptor reference revocation).
- **Buffered generation** is the current reported behavior: tokens stream into a
  buffer and are only displayed/persisted if the lifecycle token still matches.
  This trades live token-by-token display under lock-race for safety; confirm
  this is the desired UX.

## 5. Pending verification tasks (the next task)
Focused final verification pass — do NOT restart the redesign or repeat
completed stages, and do NOT begin live conversion:
1. UI no-post-lock-display tests.
2. Inactivity expiration behavior and default (`ui.lock_timeout_minutes`, default 0).
3. Retained retriever/encryptor reference revocation on lock.
4. Legacy plaintext-index handling.
5. Authentication / dry-run side-effect guarantees (no writes on failure).
6. Retirement backup, replacement, and interruption recovery.
7. Accurate user-facing documentation and check results.

## 6. Issue references / tracking drafts
- `docs/github_changes.md` holds DRAFT GitHub issue/project changes.
  **These drafts have NOT been applied.** No remote issue/project writes made.

## 7. Setting up an isolated SYNTHETIC workspace for development
Never use the real workspace for development/tests.
```bash
# From repo root. Use a throwaway temp dir and a redirected salt file.
python - <<'PY'
import tempfile
from pathlib import Path
from therapy_ai.memory.session_store import Encryptor, SessionStore
tmp = Path(tempfile.mkdtemp())
Encryptor.SALT_FILE = str(tmp / ".salt")          # redirect salt away from data/
store = SessionStore(db_path=tmp / "sessions.db") # synthetic DB
store.initialize("a-test-passphrase")             # verified empty workspace
sid = store.create_session(mood_before=5)
store.add_message(sid, "user", "synthetic message")
print("synthetic workspace at:", tmp)
PY
```
Tests already isolate automatically: the autouse `isolated_salt` fixture
redirects `Encryptor.SALT_FILE` to a tmp dir, and the `store` fixture uses a
throwaway initialized DB. Run: `.venv/bin/python -m pytest -o addopts="" -q`.

## 8. Resuming on another machine
```bash
git fetch origin
git checkout refactor/retire-vault
# create venv / install deps (uv on a non-native volume may need copy mode):
export UV_LINK_MODE=copy
uv sync --extra dev        # or install pytest/pytest-cov into the venv
.venv/bin/python -m pytest -o addopts="" -q
```

## 9. Code handoff vs. personal-data transfer (IMPORTANT)
This push transfers **development code and documentation only**.
- **A. Continuing development** → use SYNTHETIC fixtures (section 7). The other
  machine does NOT need and must NOT receive the real workspace.
- **B. Moving the real encrypted personal workspace** (the live `data/`
  database, salt, and crypto metadata) is a SEPARATE, explicitly approved
  transfer/backup procedure. It is NOT part of this GitHub push and must never
  be uploaded to GitHub. Do not initialize a replacement workspace on the other
  machine as if it held existing conversations.

## 10. Explicit status records
- No live conversion has been performed.
- Password change is deferred (no unsafe control exposed).
- Sensitive metadata remains partly plaintext (section 4, R18).
- Legacy plaintext-index handling still requires verification.
- UI lifecycle and inactivity-lock behavior still require focused tests.
- Buffered generation is the current reported behavior.
- GitHub tracking drafts have not been applied.
