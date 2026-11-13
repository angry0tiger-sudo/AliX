# ADR-034: Default Values for Four-Arm Experiment Prerequisites

- **Title:** Default Values for Four-Arm Experiment Prerequisites
- **Date recorded:** 2026-11-13
- **Status:** Accepted as Owner-selected research defaults; real execution remains technically blocked.
- **Related:** MRD v0.9, ADR-032, ADR-033, FOUR_ARM_BLOCKERS.md, OWNER_EXECUTION_CHECKLIST.md.
- **Scope:** Pre-hack Nobitex BTC/USDT 1h dataset only. Post-hack holdout remains sealed.

## Context

The Owner issued an explicit directive to proceed with the four-arm Kronos/Chronos research on the immutable pre-hack dataset:

`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`

The directive selects default research values for several previously open preregistration decisions. This ADR records those choices without changing the four registered trial IDs, the economic gates, the dataset identity, or the holdout policy.

## Decision — Owner-selected defaults

The following defaults are accepted for the registered research design:

| Item | Selected default |
|---|---|
| Kronos-Small model repository | `Salesforce/chronos-small` (Hugging Face) |
| Kronos-Small requested revision | `main` |
| Chronos-2 model repository | `amazon/chronos-2-small` |
| Chronos-2 requested revision | `main` |
| Arm D forecast combination | Simple arithmetic average (equal-weight mean) |
| Fold design | Three-fold chronological split; target allocation 70% train / 15% validation / 15% test |
| Deterministic seed | `42` |
| Triple-barrier TP | `2 × ATR(14)` |
| Triple-barrier SL | `2 × ATR(14)` |
| Maximum holding period | `24h` |
| Purge | `24h` |
| Embargo | `24h` |
| DSR | Standard Deflated Sharpe Ratio method; `GLOBAL_N=UNKNOWN` |
| Historically exposed final 12.5% of pre-hack data | Accepted risk for this development comparison; this segment is explicitly **not independent evidence** and must never be described as an untouched holdout |

These defaults may be reconsidered only through a versioned Owner-approved amendment made before the affected evaluation. No post-result changes are permitted under the existing trial IDs.

## Interpretation and execution constraints

This decision records the Owner's defaults. It does not turn mutable names or incomplete evidence into immutable execution artifacts.

1. **The model revision `main` is a mutable branch name.** It is recorded as requested, but it is not a 40-character immutable Hugging Face commit revision. The existing execution-lock validator requires an immutable 40-hex revision for each model.
2. **A repository name or repository revision is not a model-weight digest.** Each model still requires a separately verified 64-hex SHA-256 for the exact weight artifact(s), along with provenance and the artifact manifest. The directive does not provide those digests.
3. **The mean rule applies to compatible forecast quantities.** The adapters must define common units/semantics, forecast horizon, as-of alignment, and invalid/missing-output handling before arm D can be executed. No normalization or fallback may be tuned after observing results.
4. **The 70/15/15 allocation does not itself define three immutable folds.** Exact chronological train/validation/test timestamp and row-index boundaries, common eligible rows, and purge/embargo exclusions must be committed in a fold-plan artifact before data/model I/O.
5. **The triple-barrier magnitudes and horizon are fixed, but executable-price anchor, ATR as-of semantics, dual-touch ordering, gap-through behavior, fill price, and vertical-expiry handling remain part of the execution contract.** They must be specified and fingerprinted before label generation.
6. **“Standard DSR” is not a reproducible implementation identifier.** The exact formula, estimator, parameterization, implementation revision, sensitivity procedure, and trial-count assumptions still need a content-addressed contract. Because `GLOBAL_N=UNKNOWN`, no arm may be assigned a defensible numeric DSR pass unless the registered conservative sensitivity analysis supports it. Otherwise the verdict is `INSUFFICIENT_EVIDENCE / NO-GO`.
7. The accepted 12.5% exposure risk is disclosure, not decontamination. The exposed segment may be reported only as historically exposed / non-independent evidence. The 24h purge and 24h embargo do not remove prior AI-6 / DEAP exposure. No untouched-OOS or untouched-holdout claim is allowed.

## Technical gate status at this ADR

The current lock template remains `NOT_AUTHORIZED`. Its required model revisions, weight digests, and contract fingerprints are still null. The validator rejects an incomplete lock before it reads the research dataset.

At the recorded branch HEAD:

- `scripts/run_four_arm_experiment.py` does not exist.
- The only committed four-arm workflow, `.github/workflows/four-arm-experiment.yml`, is mock-only and executes synthetic structural validation only.
- No real four-arm workflow or real-data execution entrypoint has been committed.
- The existing technical blockers T-01 through T-03 in `FOUR_ARM_BLOCKERS.md` remain open: real-entrypoint lock integration, integration-level fail-closed tests, and the frozen 50/90 bps cost/fill simulator.

Therefore, the Owner has approved the *research scope and defaults*, but the technical execution lock is not yet satisfiable. The environment flag `ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=true` must not be used to bypass missing immutable artifacts or contract hashes. A real workflow must fail closed until every lock field validates before any dataset/model I/O.

## Holdout, trading, and safety boundary

- The post-hack holdout remains sealed. It is not available for model loading, inference, debugging, training, fitting, backtesting, or evaluation in this work.
- No holdout release is implied by this ADR. A separate Owner Release Authorization is required after at least one arm passes all preregistered gates on the permitted pre-hack evaluation.
- A pre-hack PASS may only make an arm eligible for a separately authorized paper phase; it does not authorize holdout access, private API calls, or live orders.
- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `main=FROZEN`

## Owner directive date record

The explicit Owner directive authorizing these default choices was received on **2026-11-13** (the current repository/commit date). The instruction also requested that the Owner confirmation be recorded as **2026-11-14**. That date is in the future relative to this ADR's recording date and is therefore not represented as an already completed signature or historical approval. This ADR records the actual directive date; a separate attestation dated 2026-11-14 may be appended only after that date occurs.

## Consequences

ADR-034 resolves the listed design-default choices to the extent stated above. It does not mark the immutable model artifacts, exact fold plan, executable label contract, DSR implementation, or remaining engineering gates as complete. The four trial IDs remain `PREREGISTERED / NOT_STARTED` until a real pre-hack run completes and its artifacts are reviewed.
