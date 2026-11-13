# ADR-033: Four-Arm Mock-Only Infrastructure and Execution Boundary

- **Date:** 2026-11-13
- **Status:** Accepted for mock-only infrastructure; economic execution is not authorized.
- **Supersedes:** None.
- **Related:** ADR-032, MRD v0.9, QWEN.md §10.2, four-arm tracking issue.

## Context

The four-arm family was preregistered under ADR-032 as
NOBITEX-KC26-A/B/C/D. Its execution prerequisites remain unresolved.
The research branch then accumulated seven consecutive failed CI runs
(#2357–#2363). The branch was restored to the last green tree from #2355 by
a normal, non-force-push commit; the recovery commit was verified green by
CI #2364.

The follow-on mock implementation is deliberately limited to deterministic
synthetic fixtures, pure gate functions, and an independent fail-closed
authorization contract. This decision does not extend ADR-032 into an
execution authorization.

## Decision

1. Keep the four registered trial IDs unchanged and PREREGISTERED /
   NOT_STARTED. Do not create candidate results or change the DSR trial
   cohort through mock validation.
2. Generate synthetic OHLCV fixtures in memory with a fixed seed. The mock
   report may contain the seed, row count, and a SHA-256 fingerprint only;
   it must not serialize or print OHLCV values.
3. Keep gate evaluation pure and side-effect free:
   - both 50 bps and 90 bps NetPF inputs must meet the preregistered
     threshold of 1.10;
   - completed trades must be at least 100;
   - maximum drawdown must not exceed 15%;
   - DSR and every supplied DSR sensitivity value must be strictly positive;
   - at least three fold-result flags must be passing;
   - if multiple arms pass, select the highest DSR; equal values use the
     lexicographically smaller registered trial ID as a deterministic tie
     resolution.
4. The fold helper consumes already-computed fold pass/fail flags. It does
   not define fold boundaries or the metric calculation within a fold.
   The DSR helper checks supplied sensitivity values; it does not calculate
   DSR or resolve GLOBAL_N. Those methods remain mandatory execution-lock
   items under ADR-032.
5. Any future non-mock research path must first call the independent
   execution-lock validator. The owner flag
   ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=true is necessary but not
   sufficient: ALIX_DRY_RUN=true, ALIX_ALLOW_LIVE_TRADING=false,
   an owner authorization commit SHA, the exact preregistered pre-hack data
   hash, immutable model revisions and weight hashes, and all contract
   fingerprints must also validate. Authorization is checked before the
   lock file is read. Failure is closed.
6. The committed execution-lock template is NOT_AUTHORIZED and contains
   no model revisions, weight hashes, seed, contract hashes, or owner
   authorization SHA. The mock workflow hard-codes the owner execution flag
   to false.
7. The dedicated GitHub Actions workflow is workflow_dispatch only.
   Its only execution path runs the synthetic mock module and uploads a
   structural mock report. It has no push trigger, market-data fetch,
   model download, training, inference, backtest, or economic evaluation.
   The workflow is not dispatched by this decision.

## Consequences

The following remain hard blockers before any actual four-arm research run:

- Immutable Kronos-Small and Chronos-2 repository revisions, weight SHA-256
  values, runtime versions, licenses, provenance, and pretraining-corpus
  overlap disclosure.
- Frozen forecast-to-decision adapters for arms B and C and the exact
  combined rule for arm D.
- Exact chronological fold boundaries and deterministic execution seed.
- Triple-barrier executable-price anchor, fill/collision/tie convention,
  and vertical-barrier expiry semantics.
- Frozen fee, slippage, fill, and execution simulator assumptions for the
  separate 50 bps and 90 bps scenarios.
- Exact DSR implementation and auditable historical-trial reconciliation;
  GLOBAL_N remains UNKNOWN until proven.
- A complete immutable execution-lock record and a later explicit Owner
  directive authorizing the pre-hack research run.

The post-hack holdout remains sealed. It may only be considered after a
preregistered arm passes all gates and the Owner separately issues Release
Authorization. Paper Trading and Real Trading remain NOT AUTHORIZED.

## Safety invariants

Champion=null; order_permission=false;
trading_authorization=NO-GO; private_api_calls=0;
real_order_attempts=0; ALIX_DRY_RUN=true;
ALIX_ALLOW_LIVE_TRADING=false; main=FROZEN.


## Appendix — Remaining blockers for real four-arm execution (2026-11-13)

This appendix is additive. The existing ADR-033 decision and its prior text remain unchanged.

### Verified mock-only infrastructure state

The mock-only infrastructure is complete for its documented scope:

- deterministic in-memory synthetic fixture generation;
- pure fold-stability, DSR-sensitivity, gate, and winner-selection contract helpers;
- independent fail-closed execution-lock validation contract;
- tests for positive and negative boundaries and authorization-before-lock-read behavior;
- a workflow_dispatch-only structural workflow that runs only the mock validator and keeps ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=false.

Standard CI #2371, Run ID 43467235675, passed on commit 2346f027fd6bd8149f78b00d8e3b0c86030b6d93. Quality, PostgreSQL integration, and Freqtrade runtime jobs all succeeded. The quality job reported 965 passed, 5 skipped, and 90.26% coverage (required minimum 90%). The two real-data research jobs were skipped under their existing conditions. The mock workflow itself was not dispatched.

This CI result validates the committed mock-only infrastructure and repository quality gates. It does not validate any economic hypothesis, model forecast, real-data execution path, or holdout operation.

### Remaining intrinsic Owner decisions

The following are still OPEN/BLOCKED and require explicit Owner decisions or immutable Owner-approved evidence:

1. Kronos-Small and Chronos-2 exact model-weight SHA-256 values.
2. Immutable source revisions / local artifact identities and pinned runtime versions for both models.
3. Forecast-to-decision mapping for arms B/C and the exact combination method for arm D.
4. Exact chronological three-fold (or greater) cut points, represented by UTC timestamps and immutable row indices.
5. Fixed deterministic seed and controlled random sources.
6. Triple-barrier executable-price anchor and final dual-touch collision, gap, fill, and vertical-expiry semantics; preregistered TP=2×ATR(14), SL=2×ATR(14), and 24h maximum holding remain the baseline parameters.
7. Exact DSR formula, parameters, implementation revision, and historical trial-count inputs.
8. Owner decision on the historically exposed final 12.5% of pre-hack data. The preregistered 24h purge and 24h embargo are temporal boundary controls; they do not erase prior AI-6 / DEAP exposure or make that segment an untouched holdout.
9. GLOBAL_N remains UNKNOWN pending candidate-level reconciliation. Heterogeneous historical counts must not be blindly summed.

The detailed status, options, recommendations, and resolution evidence are recorded in docs/research/preregistration/FOUR_ARM_BLOCKERS.md.

### Remaining technical blockers requiring separate Owner approval

1. Connect the execution-lock validator to every actual training/evaluation entrypoint and prove authorization occurs before any dataset/model/artifact I/O.
2. Add integration tests that prove fail-closed behavior across every real entrypoint and invalid/missing authorization condition.
3. Freeze and implement the shared deterministic fill/cost simulator for separate 50 bps and 90 bps round-trip scenarios.

These tasks may be implemented by the agent only after the Owner approves the engineering scope. No real-data or economic operation is included in resolving them.

### Owner checklist

The fillable approval and execution sequence is recorded in docs/research/preregistration/OWNER_EXECUTION_CHECKLIST.md. It separates intrinsic decisions, execution-lock approval, workflow authorization, and post-run result registration.

Important workflow clarification: the current four-arm workflow is mock-only. Its manual dispatch runs synthetic contract validation only. It must not be treated as an economic run. A real pre-hack research workflow requires a separately reviewed and explicitly Owner-authorized change after all blockers and execution-lock evidence are complete.

### Execution and safety boundary

All four trial IDs remain PREREGISTERED / NOT_STARTED. The post-hack holdout remains sealed. No training, fit, inference, backtest, economic evaluation, post-hack holdout access, private API call, or real-order attempt is authorized by this appendix.

Champion=null; order_permission=false; trading_authorization=NO-GO; private_api_calls=0; real_order_attempts=0; ALIX_DRY_RUN=true; ALIX_ALLOW_LIVE_TRADING=false; main=FROZEN.

This appendix is documentation and governance only. Real execution remains in wait state pending all blocker resolutions and a separate explicit Owner directive.

