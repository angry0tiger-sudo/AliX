# ADR-0011: Main Workflow Integrity — Option B

- Status: Accepted for branch-local containment; main-side limitation remains unresolved.
- Date: 2026-11-12.
- Active branch: `research/nobitex-hf-strategy`.
- Previous active-branch SHA: `877b0c5f31d2a2ef9f5a08697a94e1fcf416d7a6`.
- Frozen main SHA: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (must remain untouched).
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/42.
- Related PR: https://github.com/angry0tiger-sudo/AliX/pull/1 (closed/merged; not an Issue).

## Context

The workflow files reference code that is not present in the frozen `main` tree:

- `ai-artifact-training.yml` references `scripts/train_ai_artifacts.py`,
  `services/strategy_selection/model_io.py`, and
  `config/strategy_selection/hmm-model-approval.json`.
- `ai-artifact-shadow-validation.yml` references `scripts/run_ai_shadow_validation.py`.
- `audit-label-transitions.yml` references `scripts/fetch_binance_public_ohlcv.py`
  and `scripts/audit_ml_core_label_transitions`.

At the verified active-branch starting point, the training and shadow-validation modules were
absent from both `main` and `research/nobitex-hf-strategy`, and present on `staging`.
The V4-D label-audit source is on `research/ml-core-v4d-label-semantics`.

The three workflow definitions on `main` were inspected directly:
- Training: blob `2630c999ab6bb8af3a942736f69e371f6ff2143f`; job condition only checks the confirmation input.
- Shadow validation: blob `808546955b159e8d1e40ad6dff395fe3a3becf4f`; job condition only checks the confirmation input.
- Label audit: blob `1399daca7a028c51ee77d62074fece08c54fb6b4`; job is already restricted to the V4-D branch.

**Important GitHub Actions execution semantics:** each workflow run uses the workflow version present in the event's associated commit/ref. For `workflow_dispatch`, the selected branch is the `GITHUB_REF`; the workflow file must also exist on the default branch. Therefore, a guard committed only to this research branch does not change the YAML used by a run dispatched against `main`. See:
- https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows

The standing owner directive freezes `main`. It cannot be repaired in this workstream.

## Decision — Option B

Apply explicit, fail-closed branch guards and explanatory comments only on
`research/nobitex-hf-strategy`:

1. Public AI artifact training executes from this branch's workflow definition only when the
   selected ref is `refs/heads/staging` and `confirm_public_read_only == true`.
2. AI shadow validation executes from this branch's workflow definition only when the selected
   ref is `refs/heads/staging` and `confirm_shadow_only == true`. Its existing staging checkout
   remains unchanged.
3. Label-transition audit remains pinned to
   `refs/heads/research/ml-core-v4d-label-semantics`; its existing branch allowlist is
   documented. The same V4-D restriction is already present in the frozen main copy.

These are branch-local containment guards. They prevent the active branch's workflow versions
from executing against refs that do not contain their dependencies.

## Explicit limitation — main is not protected by this branch-only change

The frozen `main` copies of training and shadow validation remain unchanged and still have no
branch-ref condition. A manual dispatch whose selected ref is `main` uses those unchanged main
workflow definitions; those two jobs may therefore still execute on main. The active-branch
guards do **not** block that case.

This limitation is intentionally not hidden or described as fixed. Under the current owner
directive, no change to `main` is authorized. Issue #42 remains open to track this residual
risk and the owner decision required to resolve it.

## Consequences and trade-offs

- The active Nobitex HF branch cannot accidentally execute training/shadow jobs whose dependencies
  are absent from that branch.
- Training/shadow jobs remain available from the dependency-complete `staging` ref through the
  active-branch workflow definition.
- The V4-D label audit remains constrained to its source branch.
- Main-side training/shadow dispatch remains a known, unresolved risk until owner-authorized
  action can be taken on `main` or the repository-level workflow enablement is changed by an
  authorized owner.
- No merge to main and no removal of main-side workflows is authorized by this ADR.

## Acceptance and safety

- Standard CI must pass on the exact active-branch SHA with existing gates unchanged.
- No test is disabled, no static/security/coverage gate is weakened, and no fail-closed path is bypassed.
- `Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO` remain unchanged.
- `ALIX_DRY_RUN=true`, `ALIX_ALLOW_LIVE_TRADING=false`, `private_api_calls=0`, and
  `real_order_attempts=0` remain mandatory.
- No secret, private API call, real order, live execution, or model promotion is authorized.
- `main` remains frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
