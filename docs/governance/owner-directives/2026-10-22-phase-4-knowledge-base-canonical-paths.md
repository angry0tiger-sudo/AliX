# Owner Directive: Phase 4 canonical MRD and roadmap paths

- **directive_id:** ALIX-OD-2026-10-22-PHASE-4-KB-PATHS
- **owner:** angry0tiger-sudo
- **timestamp:** 2026-10-22
- **approval_status:** Explicit owner instruction from the Phase 4 handover; PR review and merge remain required
- **phase:** 4 — FreqAI baseline and quantitative walk-forward
- **supersedes:** Only the previous path designation for canonical MRD and roadmap references in `ALIX-OD-2026-10-22-PHASE-4`; it does not change the underlying architecture or roadmap requirements.

## Instruction

The stable, owner-requested reference paths for the two comprehensive project documents are:

- `docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md`
- `docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md`

Populate both files from the complete source documents already present on the Phase 4 branch. Preserve their wording, section structure, terminology, and detail; do not summarize or silently rewrite the source. Keep the earlier paths as compatibility copies unless a separately reviewed owner instruction authorizes their removal:

- `docs/architecture/AliX_MRD_Dynamic_Risk_FA.md`
- `docs/roadmap/AliX_Roadmap_Dynamic_Risk_FA.md`

The document index must identify the `docs/knowledge_base/` copies as the canonical stable paths. The project documentation validator must require both copies, and CI must execute that validator. Update affected links and Phase 4 acceptance/status evidence so the new paths are discoverable and auditable.

## Scope and safety

- Apply changes only on `feat/phase-4-freqai-baseline` through GitHub.
- Do not change `main`, merge PR #4, or create a replacement PR.
- Do not weaken quality, security, dependency, coverage, or formatting gates.
- This path directive does not grant economic acceptance, Champion promotion, or trading authorization.
- Keep `Champion = null`, `promotion_status = BLOCKED_NO_ECONOMIC_GATE`, economic acceptance **NO-GO**, and trading authorization **NO-GO**.
- No order, exchange connection, market/provider request, notification send, live/dry-run/paper session, or network configuration change is authorized or required.

## Acceptance evidence

- The new canonical paths exist and retain the complete source contents.
- `docs/PROJECT_DOCUMENTS.md` clearly links both paths.
- `scripts/validate_project_docs.py` requires both paths and is executed by `.github/workflows/ci.yml`.
- Exact-head CI and artifact results are recorded in the Phase 4 acceptance/status documents and PR #4.
- This directive is committed on the Phase 4 branch; no merge is implied.
