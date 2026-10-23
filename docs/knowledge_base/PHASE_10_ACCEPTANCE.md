# Phase 10 Acceptance Checklist

Status: IN PROGRESS / NO-GO until exact-head CI, owner review and required external-evidence steps are complete.

## GitHub and safety
- [ ] Candidate exists on a dedicated branch from main SHA `7893c87a8eea7fc10cf91b7cac3d891797b1f9e5`.
- [ ] Issue #13, ADR-0010, Owner Directive, this checklist and status report are linked.
- [ ] CI passes on the exact candidate SHA without weakening existing quality, coverage, security, dependency or architecture gates.
- [ ] `Champion=null`, `order_permission=false`, `promotion_status=BLOCKED_NO_ECONOMIC_GATE`, economic acceptance NO-GO and trading authorization NO-GO.
- [ ] No merge occurs without separate explicit owner authorization.

## Scheduler
- [ ] 07:00 local wall time uses an IANA timezone.
- [ ] DST gaps/folds, invalid timezone, single-flight duplicate triggers, expired lease/restart, DB failure, retry delay and sanitized error status are tested.
- [ ] Catch-up window is bounded; later missed runs are recorded as skipped.
- [ ] Every run has stable ID, status, attempt, start/end timestamps and audit events.

## Reporting and sources
- [ ] Read-only Freqtrade schema adapter checks the pinned table contract.
- [ ] Synthetic Decision Ledger remains labelled SYNTHETIC_ONLY / NOT_RECONCILED.
- [ ] Missing account snapshot or decision/order linkage yields DEGRADED; uncertified PnL/equity/drawdown are null.
- [ ] report_id, revision, source fingerprints, cutoff, schema and snapshot hash are available.
- [ ] PNG output is versioned and unreconciled values are never plotted; text fallback exists.
- [ ] Bale sender remains fake/injected in tests; no real message is sent.

## Public historical market data
- [ ] Manually dispatched workflow only; bounded GET to documented Nobitex public OHLCV endpoint, no credentials.
- [ ] OHLCV schema, timestamp range, bar cadence, duplicate bars, gaps and minimum coverage validated with no interpolation.
- [ ] Dataset, SHA-256 manifest and walk-forward report are uploaded as Actions artifacts, not committed.
- [ ] Research report uses the exact workflow SHA, dataset hash, fee/slippage assumptions and fold boundaries.
- [ ] Win rate is accompanied by sample count/time range. Thresholds are evaluated only with >=100 trades: win rate >=60%, absolute maximum drawdown <=15%, positive net expectancy.
- [ ] Insufficient sample or failed thresholds mean NO-GO. Passing them never authorizes trading or creates Champion.
- [ ] Report explicitly states the fixed-horizon research rule is not a Freqtrade strategy / RiskGate E2E backtest.

## Failure injection and handover
- [ ] DB unavailable, stale/gapped data, reconciliation mismatch, restart/duplicate run, lease contention, transport timeout, rate limiting, chart failure and latched Kill Switch have mock tests or are explicitly blocked.
- [ ] Windows runbook is prepared with candidate commit verification and disposable-test-only boundaries.
- [ ] Windows/WSL2, production PostgreSQL roles/backup, real account reconciliation, real order lifecycle and Bale delivery are not claimed tested.
