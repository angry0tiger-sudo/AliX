# ADR-042: Four-Arm Run #10 Mixed Economic Verdict and Safety Closeout

- **Status:** Accepted — no candidate promoted; mixed cohort classified as insufficient evidence
- **Date:** 2026-11-15 UTC
- **Branch:** `research/nobitex-hf-strategy`
- **Execution HEAD:** `71bf644dc0dd0437cd577f1f2df515e9db95c5e1`
- **Owner directive:** `OWNER-2026-11-14-FOURARM` (pre-hack research only)
- **Experiment:** Run #10 / `43712086361`; Job `132805651365`
- **Results artifact:** `14163527266`
- **Registry:** `docs/research/TRIAL_REGISTRY.md` (append-only)

## Context and execution verification

Run #10 completed successfully. The twelve-trial step passed in 1:08:07 and the upload step for the twelve manifests and forecast-cache evidence passed. The result ZIP has 12 unique expected trial manifests, the run report, and the cache artifact. All required fields and three fold records per trial are present.

The evaluation used only the verified pre-hack dataset SHA `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`: 27,825 confirmatory rows from 31,801 source rows, excluding 3,976 tail rows. No post-hack holdout access occurred.

## Measured evidence

- Forecast cache: 1,138 verified entries/anchors; cache SHA-256 `aea6c537f3ca79a28e75faf1a118bd57d5dea4fb58b45e477eb401df03d01868`; identity SHA-256 `f879e5c179cf6451485ef1ce432c7d921f5f4344146d80f08026f7f8cfcf2427`.
- Model inference: 1,138 calls per model; 2,276 total across Kronos and Chronos. One shared cache was computed once; nine forecast-driven B/C/D cells reused it. No per-trial inference was repeated.
- Arm A control: the threshold-120/180/240 manifests disclose the same Donchian control. Its trade series was computed once and referenced three times; the three manifests have identical metrics and folds.
- `GLOBAL_N=UNKNOWN`; audited historical lower bound 7,912. Every manifest has `dsr=null`; sensitivity multipliers [1, 2, 5, 10] are disclosed, but all numeric DSR values are **MISSING** because `numeric_dsr_computed=false`. No N or DSR estimate is invented.
- Every trial has three folds and zero passing folds. The complete aggregate and per-fold metrics are appended to the Trial Registry.

## Decision

Apply the registered 100-trade sample gate before interpreting below-sample candidates:

1. **FAIL (3 cells):** NOBITEX-KC26-FD-120-A, -180-A, and -240-A. Each is the same repeated control. It has 291 trades, NetPF@50=0.340813, NetPF@90=0.144630, MaxDD=94.32%, and 0/3 stable folds. These are observed failures of the baseline control, not three independent forecast strategies.
2. **INSUFFICIENT_SAMPLE (9 cells):** all B/C/D forecast-driven cells have fewer than 100 aggregate trades. They cannot be declared economic failures or promotion candidates from this run.
3. **PASS: 0 cells.**

The cohort is mixed: 3 FAIL and 9 INSUFFICIENT_SAMPLE. The all-FAIL condition for a directional-family NO-GO is not met; the all-INSUFFICIENT_SAMPLE condition is also not literally met because the repeated control failed. The cohort-level decision is therefore **INSUFFICIENT_SAMPLE / MIXED_COHORT — no candidate qualifies for progression**. The nine forecast-driven candidates remain under-sampled, so a family-wide directional NO-GO would overstate the evidence. The conditional TTM-R2 volatility pivot is not triggered by this mixed result and is not executed.

No Champion is selected. No holdout release or paper/live permission follows from this evaluation.

## Registry and audit trail

The twelve actual manifest records are appended to `docs/research/TRIAL_REGISTRY.md`; prior preregistration/history is preserved with zero deletions. The smoke remains technical evidence only and is not registered as a trial. Issue #43 receives the result table, adjudication, and run/artifact identities.

## Safety closeout

- Execution lock: `NOT_AUTHORIZED`; `owner_authorization_commit_sha=null`.
- Workflow authorization environment: `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false`.
- `ALIX_DRY_RUN=true`; `ALIX_ALLOW_LIVE_TRADING=false`.
- `EXECUTION_TRIGGER=PROBE`; `ACQUISITION_TRIGGER=PROBE`.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`.
- `private_api_calls=0`; `real_order_attempts=0`; `posthack_holdout_access=false`.
- Frozen main remains `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`. V2, Spec v3, public fetcher, and `data/nobitex/prehack/*` are not changed.

This ADR records an evaluation and safety closeout only. It authorizes no additional trials, tuning, pivot execution, holdout access, or trading.

---

## Append-only correction — run-report verdict integrity (2026-11-15)

A code review identified that the original `economic_verdict` and
`directional_family_verdict` fields in `four-arm-run-report.json` were assigned
constant strings by `_build_run_report()`. They were not derived from the
twelve per-trial decisions and therefore overstated the semantic authority of
those field names.

The runner now emits `cohort_verdict`, `promotion_gate_status`, and
`cohort_decision_counts`, all derived from the actual `trial_manifests`
decision values. The misleading constant fields have been removed. A unit
contract test varies the input decision mix and asserts that both verdict
fields change accordingly, including mixed, all-insufficient, and all-economic-
failure fixtures.

**Run #10's registered economic adjudication is unchanged:** 0 PASS, 3 FAIL
(the same threshold-invariant Arm-A control referenced three times), and 9
INSUFFICIENT_SAMPLE. All twelve original manifest records and all registry
entries remain unchanged. `GLOBAL_N=UNKNOWN`; numeric DSR and all numeric
sensitivity values remain MISSING. This correction repairs report semantics
only; it does not recompute, relabel, or reinterpret Run #10.
