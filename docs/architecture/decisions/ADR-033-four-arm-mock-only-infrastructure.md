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
