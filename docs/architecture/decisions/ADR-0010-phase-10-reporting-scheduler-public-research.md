# ADR-0010: Restart-Safe Reporting and Public-Market Research

- Status: Proposed; implementation is confined to the Phase 10 branch.
- Date: 2026-10-23.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/13.
- Owner Directive: docs/governance/owner-directives/2026-10-23-phase-10.md.
- Initial main SHA: 7893c87a8eea7fc10cf91b7cac3d891797b1f9e5.

## Context

The original ReportingEngine accepts only a generic date/summary/metrics value. The Phase 9 ledger records synthetic simulation mechanics, not authoritative Freqtrade trades, balances, fills or reconciled account data. AliXStrategy and AliXFreqAIBaselineStrategy intentionally emit no trading signals.

The current Owner Directive permits public read-only market-data GET calls only in manually dispatched GitHub Actions workflows. It prohibits credentials, account/order endpoints, real orders, live trading sessions, notification sends, local-owner execution and auto-promotion.

## Decisions

1. Persist scheduler execution state per job and resolved schedule instant. Use a unique database key, lease token/expiry, attempt count and run-event history; failed runs receive bounded exponential retry delay. SQLite is confined to one host. Use PostgreSQL for shared multi-host scheduling.
2. Schedule with IANA timezone data. Non-existent DST wall time is advanced to the first valid minute; ambiguous time selects the earlier instant by default (the later instant is configurable). Runs missed by more than six hours are recorded as `SKIPPED_LATE`, not replayed in a burst.
3. Use stable run IDs per schedule slot, stable report IDs per scope/date/schema and snapshot hashes over report content. Content changes create a report revision, not a silent overwrite.
4. Read Freqtrade's pinned SQLite `trades` table through a read-only schema-reflected adapter. Fail closed if required columns are missing. Treat stored profit as reported-only and unreconciled unless linked account/decision snapshots independently prove reconciliation.
5. Read Phase 9 ledger only with `SYNTHETIC_ONLY` provenance. It cannot be used as a source for real PnL.
6. Compose reports from interfaces. Missing, stale, mismatched or non-reconciled data yields `DEGRADED`; PnL/equity/drawdown remain uncertified. PNG charts have a reconciliation guard and text fallback.
7. Fetch public BTCUSDT hourly candles from the documented Nobitex `GET /market/udf/history` endpoint in a manually dispatched Actions workflow. Validate OHLC invariants, timestamps, duplicate bars, gaps and row coverage. Upload CSV, manifest, data hash and research metrics as artifacts; do not commit market data.
8. Evaluate the pinned walk-forward regression baseline using predeclared fee/slippage assumptions. Report win rate with sample count, net expectancy, drawdown and the owner-specified thresholds; insufficient sample fails closed. Results are not trading authorization.
9. The optional public-price observer is bounded to 5/15/60 minutes, records `ABSTAIN` on every iteration and never invokes Freqtrade, order APIs or Bale.
10. Keep Champion unset, order permission false, promotion blocked and trading authorization NO-GO.

## Trade-offs

- Durable leases make concurrent triggers and restart recovery testable but cannot guarantee exactly-once external side effects.
- Public price-history research is more representative than synthetic mechanics but still cannot prove strategy profitability, actual venue fee tier, fills or account reconciliation.
- Because the strategy scaffold remains signal-free, results from the fixed-horizon rule must not be described as an AliXStrategy or complete Freqtrade/RiskGate backtest.
