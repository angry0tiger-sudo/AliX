# Owner Execution Checklist — Four-Arm Kronos/Chronos Experiment

**Registration:** NOBITEX-FOUR-ARM-KC-20261113  
**Related decisions:** ADR-032, ADR-033  
**Blocker register:** FOUR_ARM_BLOCKERS.md  
**Current state:** NOT AUTHORIZED — all four trial IDs remain PREREGISTERED / NOT_STARTED.

> This checklist is an approval record template. Filling it in does not by itself authorize execution. The Owner must issue a separate, explicit run directive after the immutable execution lock and the real-research workflow have been reviewed.

## Stage 1 — Owner decisions for intrinsic blockers

Complete every field and attach the referenced immutable evidence before signing.

- [ ] **Kronos-Small weight SHA-256:** ______________________________________
- [ ] **Chronos-2 weight SHA-256:** _________________________________________
- [ ] **Immutable model revisions** (full Hugging Face commit or local artifact SHA; include runtime/version manifest):  
  __________________________________________________________________________
- [ ] **Forecast-to-decision mapping for arm B (Kronos-Small):**  
  __________________________________________________________________________
- [ ] **Forecast-to-decision mapping for arm C (Chronos-2):**  
  __________________________________________________________________________
- [ ] **Arm D prediction-combination rule** (average / fixed weighted average / consensus / stacking; exact formula and invalid-input behavior):  
  __________________________________________________________________________
- [ ] **Exact chronological Fold boundaries** (UTC timestamps and dataset row indices; train / validation / OOS per fold):  
  __________________________________________________________________________
- [ ] **Fixed deterministic seed** (and controlled random sources): ___________________
- [ ] **Triple-Barrier contract confirmed** (TP=2×ATR(14), SL=2×ATR(14), max holding=24h; executable-price anchor, dual-touch collision/tie, gap and fill semantics):  
  __________________________________________________________________________
- [ ] **Final DSR method approved** (formula/version, parameters, implementation SHA-256, trial-count sensitivity):  
  __________________________________________________________________________
- [ ] **GLOBAL_N decision** (candidate-level reconciliation plan; keep UNKNOWN if unproven; define the INSUFFICIENT_EVIDENCE/NO-GO condition):  
  __________________________________________________________________________
- [ ] **Historical final-12.5% exposure decision** (confirm that 24h purge + 24h embargo are only temporal-overlap controls, and specify whether the known-exposed segment remains non-independent or is excluded from confirmatory scoring):  
  __________________________________________________________________________

**Owner decision record / evidence links:**  
____________________________________________________________________________

**Stage 1 approval:**  
Owner: ______________________________  Date (UTC): _________________________  
Decision: APPROVED / REJECTED / RETURN FOR REVISION  
Signature or immutable approval reference: __________________________________

## Stage 2 — Complete and approve the immutable execution lock

All intrinsic decisions must be resolved and all technical blockers in FOUR_ARM_BLOCKERS.md must have verified evidence before this stage.

- [ ] Execution-lock validator is connected to every real training/evaluation entrypoint.
- [ ] Integration tests prove fail-closed behavior and prove no dataset/model I/O occurs before authorization.
- [ ] Cost/fill simulator is frozen for separate 50 bps and 90 bps scenarios and has deterministic synthetic tests.
- [ ] Complete execution-lock JSON is committed with immutable model revisions, weight SHA-256 values, fixed seed, data identity, and every required contract fingerprint.
- [ ] Owner reviews and signs the exact execution-lock commit.
- [ ] Owner-controlled authorization is issued for this specific pre-hack research run. The required flag is ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=true; the flag alone is not sufficient authorization.
- [ ] Safety flags remain ALIX_DRY_RUN=true and ALIX_ALLOW_LIVE_TRADING=false.
- [ ] Exact pre-hack dataset SHA-256 is confirmed:  
  0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e

Execution-lock commit SHA: _________________________________________________  
Owner approval reference: __________________________________________________  
Approval date (UTC): _______________________________________________________

## Stage 3 — Workflow dispatch authorization

**Important current-state warning:** the committed .github/workflows/four-arm-experiment.yml is currently a mock-only structural-validation workflow. It runs synthetic fixtures only and has ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false. Dispatching the current file does not train/evaluate models and does not produce an economic result. Do not dispatch it expecting a real four-arm run.

- [ ] Owner reviews the exact workflow commit and confirms it is a separately authorized **pre-hack research** workflow, not the current mock-only workflow.
- [ ] The workflow proves it validates the complete execution lock before any pre-hack data/model I/O.
- [ ] The workflow has no post-hack holdout fetch, no private API access, no live trading, and no real-order path.
- [ ] Owner explicitly authorizes dispatch of that reviewed workflow.
- [ ] Owner dispatches the authorized workflow.
- [ ] Run ID recorded: ______________________________
- [ ] Dispatched commit SHA recorded: ______________________________
- [ ] Workflow URL / evidence: ______________________________________

Optional infrastructure check only (not an economic experiment):
- [ ] Owner dispatches the currently committed mock-only workflow for synthetic structural validation.
- [ ] Mock-only Run ID: ______________________________
- [ ] Confirm report says MOCK_ONLY and trial_evaluation_performed=false.

## Stage 4 — Record results and issue the economic decision

Complete only after an authorized pre-hack research run has finished and its artifacts have been independently reviewed.

- [ ] Append run evidence and all four arm outcomes to TRIAL_REGISTRY.md; preserve append-only history.
- [ ] Retain all four trial IDs in multiple-testing / DSR accounting if evaluated:
  - NOBITEX-KC26-A
  - NOBITEX-KC26-B
  - NOBITEX-KC26-C
  - NOBITEX-KC26-D
- [ ] Report Cost-50 and Cost-90 separately for every arm.
- [ ] Report fold-level and pooled OOS metrics; retain failed, partial, rejected, and incomplete evidence.
- [ ] Record the reconciled DSR trial count or keep GLOBAL_N=UNKNOWN with the required conservative sensitivity and explain any INSUFFICIENT_EVIDENCE outcome.
- [ ] Issue and record the economic verdict: PASS / NO-GO / INSUFFICIENT_EVIDENCE.
- [ ] If no arm passes every preregistered gate: record NO-GO and keep the post-hack holdout sealed.
- [ ] If an arm passes: request separate Owner Release Authorization before any holdout access. A PASS does not automatically open the holdout.
- [ ] If and only if separate Release Authorization is granted, use a fresh one-time nonce under the approved protocol. Never reuse a consumed or rotated nonce.
- [ ] Record the separate Release Authorization reference and the holdout acquisition run ID; do not expose holdout values in stdout or reports.

Economic verdict: ______________________________  
Review / evidence reference: ____________________  
Owner: _______________________________________  
Date (UTC): ___________________________________

## Final authorization statement

- [ ] I confirm all Stage 1 intrinsic decisions are explicit and versioned.
- [ ] I confirm all Stage 2 technical and execution-lock requirements are verified.
- [ ] I confirm the dispatched workflow commit matches the approved execution-lock contract.
- [ ] I authorize only the specific pre-hack research action recorded above.
- [ ] I understand that this checklist does not authorize post-hack holdout access, Paper Trading, Real Trading, private API calls, or real orders.

Owner signature / immutable approval reference: ______________________________  
Approved scope: ___________________________________________________________  
Approved commit SHA: _____________________________________________________  
Date (UTC): ______________________________________________________________

## Non-negotiable safety invariants

- Champion = null
- order_permission = false
- trading_authorization = NO-GO until a separate authorization changes the permitted scope
- private_api_calls = 0
- real_order_attempts = 0
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- main = FROZEN

**Do not execute any real training, fit, inference, backtest, economic evaluation, holdout access, or trading action merely because this checklist exists or has been partially completed.**
