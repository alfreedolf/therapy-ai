# Requirement → Implementation Traceability (vault retirement refactor)

Branch: `refactor/retire-vault`. Synthetic-fixture implementation only.
No live-data access, no GitHub remote writes. Updated as work proceeds.

> NOTE: The existence of a single live "vault" conversation is **user-reported**,
> not independently verified. This refactor does not open or inspect the live
> `data/sessions.db` / `data/.salt`.

## Legend
- ✅ implemented + tested (synthetic)
- 🔶 in progress
- ⏳ planned this task
- ⛔ deferred / needs decision (documented)

| # | Requirement | Design / location | Status | Tests |
|---|---|---|---|---|
| R1 | Verified workspace access: initialization vs unlock distinction; verify before authenticated state | `WorkspaceAuth` in `session_store.py` (workspace verifier + KDF meta) | ✅ | `test_workspace_auth.py` |
| R2 | Wrong credentials: no writes, no replacement-metadata creation | `unlock()` verifies before any write | ✅ | test_workspace_auth |
| R3 | Missing/corrupt verifier on non-empty DB → fail closed; report ambiguous state | `unlock()` fail-closed + `WorkspaceState` | ✅ | test_workspace_auth |
| R4 | Legacy bootstrap: verify entered passphrase against existing app-key ciphertext (never fabricate verifier as proof) | `legacy_bootstrap()` distinct op | ✅ | test_workspace_auth |
| R5 | Vault-only legacy workspace recovery (verify vault cred, init NEW workspace, convert; preserve originals until validated) | `WorkspaceState.VAULT_ONLY_LEGACY`; `vault_retirement.py` + `scripts/retire_vault.py` | ✅ | test_transition_guard; test_vault_retirement::TestCase2VaultOnlyRecovery |
| R6 | Private by default (unconditional encryption) | existing `Encryptor` (AES-256-GCM/PBKDF2) retained | ✅ (retained) | test_encryption |
| R7 | No per-conversation passwords (remove vault create/unlock) | store vault API removed; `chat.py` UI removed; retirement guard UI | ✅ | UI import smoke; test_transition_guard |
| R8 | Transition guard | `is_vault` read kept; guard in load/get/close/opt-in; exclusion in list/mood | ✅ | test_transition_guard |
| R9 | Protected derived storage: encrypt source summaries; in-memory index | `memory_retriever.py` rework | ✅ | test_memory_policy |
| R10 | Memory CONTRIBUTION opt-in (authoritative, persisted, default OFF); distinct from RECEIVING | `memory_opt_in` column + policy | ✅ | test_memory_policy |
| R11 | Export/training separate; must not govern memory indexing | indexing gated by R10 only; export unchanged | ✅ | test_memory_policy::TestExportSeparation |
| R12 | Lock lifecycle: manual lock, inactivity lock, service revocation, cache/index cleanup, no post-lock writes/results, revoked-lifecycle token | store: lifecycle token + `_require_unlocked` + `check_lifecycle`; UI: `_lock_and_clear`, `_enforce_inactivity_lock`, buffered-then-guarded stream display | ✅ (store tested; UI logic exercised via import + store-level lifecycle tests) | test_workspace_auth::TestLockedStateEnforcement |
| R13 | Versioned, workspace-local KDF/format metadata | `workspace_meta(format_version, kdf_algo, kdf_iterations)` | ✅ | test_workspace_auth |
| R14 | Password-change behavior: specify+test OR document deferred and hide unsafe control | documented below | ⛔ (deferred, documented) | — |
| R15 | Pre-existing persistent plaintext index artifacts quarantined, not reused | `memory_retriever._setup_db` quarantine | ✅ | test_memory_policy |
| R16 | Path isolation (store honors injected paths; salt isolation) | injected `db_path`; `isolated_salt` fixture | ✅ | test_workspace_auth/test_memory_policy use tmp paths |
| R17 | Conversion-only retirement; preserve ids/history; contribution OFF; no plaintext intermediate; SQLite-consistent backup; interruption recovery | `vault_retirement.py` (`make_consistent_backup` online-backup API, `convert_vault_rows` in-txn, `validate_converted_backup`) + `scripts/retire_vault.py` (inert CLI, getpass, explicit REPLACE) | ✅ | test_vault_retirement (10 tests) |
| R18 | Sensitive-metadata coverage honesty | documented below | ✅ (documented) | — |

## Documented decisions / deferred items

### R14 — Password change
DEFERRED this task. Changing the workspace passphrase requires re-deriving the
key and re-encrypting all app-key ciphertext (messages, notes, themes, summaries)
plus rewriting the workspace verifier. Not implemented here. **No password-change
UI control will be exposed** (an unsafe/no-op control would risk data loss).
Documented as an unresolved requirement.

### R18 — Metadata encryption coverage (honest scope)
Encrypted at rest (app key): message content, notes, themes, and (new) memory
source summaries. **NOT encrypted**: `mood_before`, `mood_after`, `rating`,
`is_flagged`, `timestamp`, `created_at`, `ended_at`, and vault marker columns.
Protecting these remaining columns is **deferred** and called out as a gap, not
silently claimed.

### Memory semantics (R10 vs receiving)
`memory_opt_in` = may this conversation CONTRIBUTE to future cross-session
context. It does NOT control whether the current conversation RECEIVES context.
Receiving is governed by the existing global `ContextConfig` recall settings
(`include_themes`, `use_semantic_memory`), retained and documented separately.
The current conversation's OWN history is always preserved regardless.
New and converted conversations default to contribution OFF.
