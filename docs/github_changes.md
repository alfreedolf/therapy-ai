# Proposed GitHub issue / project changes (DRAFT — NOT APPLIED)

No remote writes have been made. This file is for your review; apply manually
or authorise application separately. Branch: `refactor/retire-vault` (local,
not pushed).

## Epic to close / retire
- **#16 (vault isolation epic)** — the per-conversation vault tier is retired as
  a product feature. Close with a note: superseded by verified single-workspace
  model + one-time conversion tool. (Do not delete history.)

## New issues to open
1. **Verified workspace authentication** — initialize/unlock/legacy-bootstrap
   distinction, fail-closed states, versioned KDF/verifier metadata.
   (Implemented on `refactor/retire-vault`.)
2. **Retire per-conversation vault** — remove create/unlock UX + store API;
   keep transition guard + retained columns; add retirement tool.
   (Implemented.)
3. **Protected derived memory + opt-in contribution policy** — encrypted source
   summaries, in-memory index, `memory_opt_in`, legacy plaintext quarantine.
   (Implemented.)
4. **Lock lifecycle hardening** — manual/inactivity lock, lifecycle token,
   post-lock write/display prevention, index cleanup. (Implemented.)
5. **[Deferred] Workspace password change** — re-key + re-encrypt all app-key
   ciphertext + rewrite verifier. NOT implemented; no unsafe UI exposed.
   (Tracking issue only.)
6. **[Deferred] Encrypt remaining sensitive metadata** — mood/rating/flag/
   timestamp columns are currently plaintext. (Tracking issue only.)
7. **[Not in scope] Multi-user profiles** — explicitly out of scope for this
   refactor per product decision; placeholder only.

## PR (when authorised — not pushed/merged here)
- Title: "Retire per-conversation vault; add verified workspace + protected memory"
- Base: `main`  Head: `refactor/retire-vault`
- Body: summarise R1–R18 from docs/requirements_matrix.md; note live conversion
  NOT performed; call out deferred R14/R18.
