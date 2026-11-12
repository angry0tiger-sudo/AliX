# ADR-0011: Main Workflow Integrity — Option B

- Status: Accepted (owner-directed; active-branch implementation only).
- Date: 2026-11-12.
- Active branch: `research/nobitex-hf-strategy`.
- Active-branch starting SHA: `0e5e09d1b6cf6cbc36268b5cd0883fd3710cf821`.
- Frozen main SHA: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (must remain untouched).
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/42.
- Related PR: https://github.com/angry0tiger-sudo/AliX/pull/1 (closed/merged; not an Issue).

## Context

The three workflows exist in the frozen `main` tree, but the main tree does not contain all modules referenced by them:

- `ai-artifact-training.yml` references `scripts/train_ai_artifacts.py`,
  `services/strategy_selection/model_io.py`, and
  `config/strategy_selection/hmm-model-approval.json`.
- `ai-artifact-shadow-validation.yml` references `scripts/run_ai_shadow_validation.py`.
- `audit-label-transitions.yml` references `scripts/fetch_binance_public_ohlcv.py`
  and `scripts/audit_ml_core_label_transitions`.

At the active-branch starting SHA, the training and shadow-validation modules are also absent
from `research/nobitex-hf-strategy`; the training/shadow implementation dependencies are
available on `staging`. The label-transition audit remains tied to its V4-D source branch.

The standing owner directive freezes `main`. Repairing the workflow/module mismatch on `main`
or merging unrelated branches wholesale is therefore out of scope.

## Decision — Option B

Apply guards and explanatory comments only on `research/nobitex-hf-strategy`:

1. Public AI artifact training may execute only when manually dispatched against
   `refs/heads/staging` and `confirm_public_read_only == true`. The selected ref must contain
   the training script, model IO, and approval record.
2. AI shadow validation may execute only when manually dispatched against
   `refs/heads/staging` and `confirm_shadow_only == true`. Its existing staging checkout
   remains unchanged.
3. Label-transition audit remains restricted to
   `refs/heads/research/ml-core-v4d-label-semantics`; its exact branch allowlist is documented
   as a fail-closed guard. Dispatches from `main` or unrelated refs skip the job.

No workflow, source file, or commit on `main` is changed by this decision. No referenced
workflow is treated as proof that the missing modules have been ported to `main`.

## Consequences and trade-offs

- Manual dispatches from `main` or the active Nobitex HF branch cannot execute training/shadow
  jobs whose dependencies are absent there.
- Training/shadow jobs remain available only on the branch that contains their dependencies.
- The V4-D label audit remains available only on its own source branch.
- This is a containment measure, not a repair of `main` and not authorization to merge.
- A future owner directive must explicitly authorize unfreezing `main`; a separate reviewed PR
  must then port the complete dependency set and tests before any guard is removed.

## Acceptance and safety

- Standard CI must pass on the exact active-branch SHA with existing gates unchanged.
- No test is disabled, no static/security/coverage gate is weakened, and no fail-closed path is bypassed.
- `Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO` remain unchanged.
- `ALIX_DRY_RUN=true`, `ALIX_ALLOW_LIVE_TRADING=false`, `private_api_calls=0`, and
  `real_order_attempts=0` remain mandatory.
- No secret, private API call, real order, live execution, or model promotion is authorized.
- `main` remains frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
