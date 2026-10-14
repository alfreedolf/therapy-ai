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
- Full suite: **203 passed, 2 xfailed** (`.venv/bin/python -m pytest -o addopts="" -q`).
  (Was 177; +26 from the focused verification pass below.)
- `ruff check` on new/changed modules and new tests: clean. (Some PRE-EXISTING
  findings remain in `memory_retriever.py` (B904/B905/SIM105) and
  `session_store.py` (UP042 `WorkspaceState(str, Enum)`); not introduced by this
  pass and left untouched to avoid scope creep.)
- UI import smoke test: `app`, `chat`, `progress` import; vault UI gone;
  retirement UI present.
- Retirement CLI dry-run against a SYNTHETIC vault-only DB: classified
  correctly, refused to act without `--apply`.
- NOTE: default `pytest` addopts include `--cov`; coverage plugin may not be
  installed in every env. Runs above used `-o addopts=""`.

### 3a. Focused verification pass (this iteration) — new tests
- `tests/component/test_lock_lifecycle.py` — lifecycle-bound access revocation
  through the PRODUCTION `store.lock()` mechanism (no manual `_enc` clearing).
- `tests/component/test_ui_app.py` — `streamlit.testing.v1.AppTest` driving the
  REAL `app` functions (gate, bootstrap, lock-and-clear, inactivity) with mocked
  inference + synthetic storage; plus the generation reveal/persist guard.
- `tests/component/test_coverage_audit.py` — maps three previously-claimed
  guarantees (quarantine, dry-run writes, backup isolation) to exact assertions.

## 4. Known gaps & deferred requirements
- **R14 Password change — DEFERRED.** Not implemented; no unsafe UI control
  exposed. Re-keying would require re-encrypting all app-key ciphertext +
  rewriting the verifier.
- **R18 Sensitive metadata partly plaintext.** Encrypted: message content,
  notes, themes, memory source summaries. NOT encrypted: `mood_before`,
  `mood_after`, `rating`, `is_flagged`, timestamps, vault marker columns.
- **Legacy plaintext-index handling** — quarantine behavior now pinned (see 4c):
  it RENAMES, leaving plaintext rows on disk under `_legacy_plaintext_*` (not
  erased). Still NOT verified against real-world table shapes or multi-open
  idempotency beyond the synthetic cases; a dedicated plaintext-erasure step (vs.
  mere quarantine) remains a possible future requirement.
- **UI lifecycle / inactivity-lock** — now covered by `test_ui_app.py` via
  AppTest against the real functions, PLUS production-path guard tests. See
  section 4a for the revocation fix and findings. Remaining limitation: AppTest
  is single-threaded and cannot reproduce TRUE concurrent browser behavior
  (lock click landing while generation genuinely runs in parallel); that path is
  simulated by a mocked engine that revokes the lifecycle mid-stream and is
  additionally asserted at the server-side guard level.
- **Buffered generation** retained for THIS iteration (confirmed interim UX
  choice). Tokens stream into a buffer and are displayed only while unlocked
  under the same lifecycle; reveal + persist are gated by `check_lifecycle`.
  See section 4b for the honest limits of what buffering does and does not
  guarantee. Streaming restoration is explicitly NOT in scope this iteration.

## 4a. Lifecycle-bound revocation — PRODUCTION CHANGE (this iteration)
A real defect was found and fixed. Previously `MemoryRetriever` was handed a
RAW `Encryptor` reference at construction (`encryptor=store.active_encryptor`).
`store.lock()` nulled the store's key but NOT the retriever's captured reference,
so a retained retriever could still decrypt `memory_sources` and rebuild its
index AFTER lock. The earlier lock-lifecycle test only "passed" because it
manually did `retriever._enc = None` — a test crutch, not a production guarantee.

Fix (small, lifecycle-bound):
- `SessionStore.lifecycle_encryptor(token)` returns the live Encryptor ONLY if
  `token` matches the current unlocked lifecycle, else None.
- `SessionStore.encryptor_provider()` returns a zero-arg provider bound to the
  CURRENT lifecycle token.
- `MemoryRetriever` now takes `encryptor_provider` (preferred) and resolves the
  key per-use via a `_enc` property. After lock → provider returns None →
  retrieve returns [] (and drops any cached index), index/remove refuse. After a
  LATER unlock, a retriever bound to the OLD token stays dead (new lifecycle,
  new token); only a freshly constructed retriever works.
- `ContextBuilder` now constructs the retriever with `store.encryptor_provider()`.
- The deprecated `encryptor=` arg is retained (wrapped) for existing call sites
  but is NOT lifecycle-bound by design; production uses the provider.

Scope of the guarantee (explicit, no overclaiming): this governs access through
SUPPORTED interfaces. It does NOT zero memory already copied elsewhere and does
NOT defend against arbitrary in-process code reaching private attributes.
Clearing Streamlit `session_state` is a belt-and-suspenders cleanup, NOT the
security boundary.

## 4b. Inactivity-lock behavior — verified facts (not just "a check exists")
- **Default: DISABLED.** `config/default_config.yaml` has
  `ui.lock_timeout_minutes: 0`. No new default is enabled by this pass; enabling
  one is a product decision (see section 5).
- **Server-side expiration** is checked at the start of each authenticated rerun
  (`_enforce_inactivity_lock`), before page rendering/protected operations.
- **FINDING — reruns reset the idle clock.** `_enforce_inactivity_lock` sets
  `last_activity = now` on EVERY call. Because it runs on every authenticated
  rerun, ANY rerun (including automatic, non-user-initiated reruns) refreshes the
  idle timer. "Idle" therefore means "no rerun", not "no user action". Pinned by
  `test_ui_app.py::...::test_every_call_resets_activity_clock`.
- **Server access vs. rendered content.** Expiration revokes server-side access
  and clears sensitive `session_state`; it does not, by itself, scrub content
  already painted in an idle browser tab until the next rerun. Do not claim an
  already-open idle page visibly clears without an interaction/rerun.

## 4c. Coverage-claim audit (honest boundaries)
- **Legacy plaintext quarantine leaves plaintext ON DISK (renamed).** Quarantine
  RENAMES `memory_chunks`/`memory_vectors` to `_legacy_plaintext_*`; the rows are
  NOT removed. Guarantee = "not reused by retrieval", NOT "plaintext erased".
  Pinned by `test_coverage_audit.py::TestQuarantineHonesty`.
- **"Dry run" is read-only only on a current schema.** `plan_conversion` SELECTs
  only, but constructs a `SessionStore`, whose `__init__` runs additive
  migration / `create_all` — which CAN write (ADD COLUMN / CREATE TABLE) on a DB
  missing columns/tables. Pinned by `TestDryRunWriteBoundary`.
- **Backup isolates the original.** `make_consistent_backup` uses SQLite's online
  backup API and leaves the original unchanged; conversion runs on the backup
  copy; interruption leaves the original intact. Re-asserted by
  `TestBackupIsolation` (and `test_vault_retirement::TestInterruptionRecovery`).

## 5. Pending verification tasks — status after this pass
Still NOT a merge approval; the suite being green does not mean fully verified.
1. UI no-post-lock-display — ADDRESSED via AppTest + guard tests (reveal/persist
   gated by lifecycle). Limitation: true browser concurrency not reproducible in
   AppTest (documented, §4 and test module docstring).
2. Inactivity expiration behavior and default — VERIFIED + TESTED. Default stays
   DISABLED (`ui.lock_timeout_minutes: 0`). See the product decision below.
3. Retained retriever/encryptor reference revocation — DEFECT FOUND & FIXED
   (§4a); regression tested through the production `lock()` path.
4. Legacy plaintext-index handling — boundary PINNED (§4c): quarantine renames,
   plaintext retained on disk; not reused. Real-world-shape / multi-open
   idempotency still OPEN.
5. Auth / dry-run side-effect guarantees — PINNED (§4c): no writes on wrong
   credential; "dry run" read-only ONLY on a current schema (store constructor
   can migrate). 
6. Retirement backup / replacement / interruption recovery — covered
   (`test_vault_retirement`, re-asserted in `test_coverage_audit`).
7. Accurate user-facing docs / check results — THIS update + README note.

### Open product decision (do not silently enable)
`ui.lock_timeout_minutes` default is **0 (disabled)**. Enabling a nonzero default
(e.g. 15) would auto-lock idle sessions, but note the §4b finding that any rerun
resets the idle timer — so a nonzero default only locks when the tab is truly
idle (no reruns). Decision deferred to the owner; not changed here.

### Still deferred (unchanged): R14 password change, R18 metadata encryption.

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
- Legacy plaintext-index handling: quarantine PINNED (renames, retains plaintext
  on disk, not reused); real-world-shape / multi-open idempotency still open.
- UI lifecycle / inactivity-lock: now covered by AppTest + production-path guard
  tests (§4a/4b). AppTest cannot reproduce true browser concurrency (documented).
- Retained-reference revocation: a real defect was FOUND AND FIXED this pass
  (§4a) — lifecycle-bound encryptor provider; retained retrievers die on lock.
- Buffered generation: retained for this iteration (interim UX); reveal/persist
  gated by lifecycle token. Buffering alone does NOT guarantee safe revocation —
  the lifecycle gate does.
- Inactivity default remains DISABLED; enabling a default is an open product
  decision (section 5), not changed here.
- GitHub tracking drafts have not been applied; nothing pushed/merged.
- The suite is green (203 passed, 2 xfailed) but this is NOT a declaration of
  full verification or merge approval.

## 11. Interim UX note — buffered generation (this iteration)
- Buffering is an INTERIM choice; streaming restoration is deliberately out of
  scope here.
- "Can the lock button be processed while generation runs?" — Streamlit executes
  a script run to completion single-threaded; a lock interaction is handled on
  the NEXT rerun, not truly concurrently. The safety property does not rely on
  interrupting an in-flight generation: the completed buffer is revealed/persisted
  ONLY if `check_lifecycle(token)` still matches, so a lock that lands before the
  reveal causes the result to be discarded.
- Implemented & tested cancellation/expiration behaviors: (a) display suppressed
  once the lifecycle token no longer matches mid-stream; (b) reveal + persist
  refused for a revoked lifecycle (`check_lifecycle`, `add_message(lifecycle_token=)`);
  (c) old-lifecycle result still rejected after a later re-unlock. See
  `test_ui_app.py::TestGenerationRevealPersistGuard` and
  `test_workspace_auth.py::TestLockedStateEnforcement`.
