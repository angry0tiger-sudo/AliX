# Owner Execution Checklist — Four-Arm Kronos/Chronos Experiment

**Registration:** NOBITEX-FOUR-ARM-KC-20261113  
**Related decisions:** ADR-032, ADR-033, ADR-034  
**Blocker register:** FOUR_ARM_BLOCKERS.md  
**Current state:** OWNER-APPROVED DEFAULTS RECORDED; REAL EXECUTION BLOCKED BY INCOMPLETE IMMUTABLE LOCK AND MISSING REAL RUNNER.

> This checklist distinguishes Owner-approved design defaults from technical authorization to execute. The Owner has authorized the pre-hack research scope and the defaults below. That approval does not waive immutable artifact, fail-closed, or reproducibility requirements.

## Stage 1 — Owner defaults recorded in ADR-034

The following defaults were selected by the explicit Owner directive received on 2026-11-13:

- [x] **Kronos-Small repository/default revision:** `Salesforce/chronos-small`, requested revision `main`.
- [x] **Chronos-2 repository/default revision:** `amazon/chronos-2-small`, requested revision `main`.
- [x] **Arm D forecast combination:** simple arithmetic average (equal-weight mean).
- [x] **Fold design default:** three-fold chronological split; target allocation 70% train / 15% validation / 15% test.
- [x] **Deterministic seed:** `42`.
- [x] **Triple-Barrier parameters:** TP=2×ATR(14), SL=2×ATR(14), maximum holding=24h.
- [x] **Purge and embargo:** 24h each.
- [x] **DSR trial-count disclosure:** `GLOBAL_N=UNKNOWN`; no guessed global count is permitted.
- [x] **Historical exposure decision:** accept the known final-12.5% pre-hack exposure as a disclosed risk; results from this segment are not independent evidence and cannot support an untouched-holdout claim.
- [x] **Owner instruction:** pre-hack-only research scope is approved; post-hack holdout remains excluded.

### Stage 1 — Evidence still required before execution

The choices above do not supply the content-addressed evidence required by the execution-lock validator:

- [ ] Kronos-Small exact immutable model revision (40-hex Hugging Face commit), distinct from the mutable alias `main`.
- [ ] Chronos-2 exact immutable model revision (40-hex Hugging Face commit), distinct from the mutable alias `main`.
- [ ] Exact Kronos-Small model-weight artifact manifest and SHA-256 (64 hex).
- [ ] Exact Chronos-2 model-weight artifact manifest and SHA-256 (64 hex).
- [ ] Runtime/library lock, model license/provenance, and pretraining-corpus overlap disclosure.
- [ ] B and C forecast-to-decision adapters; arm D common scale/units, horizon/as-of semantics, and invalid/missing forecast behavior.
- [ ] Exact fold-plan artifact with UTC timestamps, row-index ranges, common eligible rows, and purge/embargo boundaries. The 70/15/15 allocation alone is not three immutable fold definitions.
- [ ] Executable triple-barrier contract: decision-price anchor, ATR as-of semantics, dual-touch rule, gap-through behavior, fill price, and vertical-barrier expiry.
- [ ] Exact reproducible DSR method/formula, parameters, implementation SHA, and conservative sensitivity under `GLOBAL_N=UNKNOWN`. If evidence cannot support DSR>0, record `INSUFFICIENT_EVIDENCE / NO-GO`.
- [ ] Frozen cost/fill simulator for the separate 50 bps and 90 bps scenarios.

**Important model identity rule:** `main` is recorded as the Owner's requested default label, but the lock validator requires a 40-character immutable revision and a separate 64-character weight SHA-256 for each model. A repository identifier or branch name must never be substituted for a weight digest.

**Owner directive record:** received 2026-11-13. The directive requested a confirmation date of 2026-11-14; that date is future relative to this record and is not falsely recorded as an already completed signature.

## Stage 2 — Execution lock and authorization

Owner authorization for the bounded pre-hack research scope is recorded. The operational lock remains fail-closed until every required field below is present and verified.

- [x] Owner explicitly requested the four-arm pre-hack experiment and authorized the intended value `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=true` for that scope.
- [x] Required safety environment values are specified: `ALIX_DRY_RUN=true` and `ALIX_ALLOW_LIVE_TRADING=false`.
- [x] Exact permitted dataset identity is recorded:  
  `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- [ ] Do not set the runtime authorization environment flag until the complete immutable lock validates. Current lock file status is `NOT_AUTHORIZED`; all model identity/weight digests and contract fingerprints must be populated first.
- [ ] Immutable Owner authorization commit SHA (40 hex) is recorded in the lock after the decision/lock commit exists.
- [ ] Lock validator is the first action before any dataset/model/artifact I/O on every real entrypoint.
- [ ] Integration tests prove the real entrypoint cannot read data or load models when any authorization flag, dataset hash, model revision, weight digest, or contract fingerprint is absent/invalid.
- [ ] Cost/fill simulator is frozen for both 50 bps and 90 bps and covered by deterministic synthetic tests.
- [ ] Complete execution-lock JSON is committed and independently reviewed.
- [ ] Owner approval is bound to the exact lock/workflow commit SHA.

**Current execution-lock file:** `config/experiment/four_arm_execution_lock.json`  
**Current lock status:** `NOT_AUTHORIZED`  
**Reason:** model repository revisions, weight SHA-256 values, and required contract fingerprints are null. The existing validator requires a 40-hex revision and 64-hex artifact digest for each model, plus all contract fingerprints.

Execution-lock commit SHA: PENDING  
Owner authorization reference: this directive, 2026-11-13  
Runtime authorization flag: NOT ENABLED

## Stage 3 — Real workflow and dispatch

- [ ] Implement and review an actual pre-hack four-arm execution entrypoint. `scripts/run_four_arm_experiment.py` is absent at the current HEAD.
- [ ] Create a separate `workflow_dispatch`-only real-experiment workflow with `contents: read`, an 480-minute timeout, immutable lock validation before I/O, and no post-hack, private-API, or order path.
- [ ] Confirm the real workflow uses only the pre-hack dataset SHA recorded above and all four registered trial IDs under identical eligible rows and execution assumptions.
- [ ] Confirm artifact output includes all four arm manifests and structured logs, without raw holdout values or secret material.
- [ ] Owner reviews the exact workflow/runner/lock commit and confirms the runtime authorization scope.
- [ ] Dispatch the real workflow only after every Stage 1 evidence item and Stage 2 technical requirement passes.
- [ ] Workflow Run ID: PENDING
- [ ] Dispatched commit SHA: PENDING
- [ ] Artifact IDs / evidence: PENDING

**Current workflow warning:** `.github/workflows/four-arm-experiment.yml` is mock-only. It runs synthetic fixtures, hard-codes `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false`, and cannot produce an economic result. It must not be treated as the real experiment. No real four-arm workflow has been committed or dispatched.

## Stage 4 — Results, Trial Registry, and verdict

This stage is authorized as the required reporting procedure, but it must remain uncompleted until a genuine, authorized pre-hack run has finished and its artifacts have been reviewed.

- [ ] Append the actual run evidence to `docs/research/TRIAL_REGISTRY.md`; preserve its append-only history.
- [ ] For each arm A/B/C/D, record `arm_id`, `netPF_50bps`, `netPF_90bps`, `trades`, `maxDD`, DSR status/value, fold-level evidence, and winner flag.
- [ ] Retain all four trial IDs in multiple-testing accounting if evaluated: NOBITEX-KC26-A/B/C/D.
- [ ] Report Cost-50 and Cost-90 separately; no gross/net substitution or scenario averaging.
- [ ] If `GLOBAL_N=UNKNOWN` prevents a defensible DSR result, record `INSUFFICIENT_EVIDENCE / NO-GO`; do not invent a numeric DSR.
- [ ] Issue PASS only if NetPF≥1.10 at both costs, trades≥100, MaxDD≤15%, DSR>0 on defensible evidence, and at least three chronological folds pass.
- [ ] If an arm passes, it becomes eligible for a separately authorized paper phase only. Keep holdout sealed until the Owner issues distinct Release Authorization.
- [ ] If no arm passes, record documented NO-GO and the redesign/stop decision.
- [ ] Comment on Issue #43 only after the measured outcome is known; do not pre-announce a winner or NO-GO before a valid experiment result.
- [ ] Disable `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED` after the run, including failure/abort paths; retain dry-run true and live-trading false.

Economic verdict: PENDING  
Issue #43 comment: NOT POSTED (no economic result exists)  
Execution-lock disablement: NOT APPLICABLE — authorization flag has not been enabled

## Holdout and safety invariants

- Post-hack V2 status: `FETCHED_LOCKED`.
- `config/holdout/DISPATCH_TRIGGER`: `PROBE`.
- No post-hack holdout data may be read, copied, materialized, or supplied to any model/run in this task.
- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- main = FROZEN

**This checklist records Owner-approved defaults and execution boundaries. It does not convert missing model artifacts, missing contract hashes, or the mock-only workflow into a real experiment.**
