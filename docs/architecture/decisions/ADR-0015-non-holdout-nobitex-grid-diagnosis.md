# ADR-0015: Non-Holdout Nobitex Grid Diagnosis and Deferred Holdout Dispatch

- Status: Accepted for the active research branch.
- Date: 2026-11-13.
- Active branch: `research/nobitex-hf-strategy`.
- Frozen main: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (unchanged).
- Owner governance source: AliX Master Reference Document v0.9.
- Tracking issue: [#43](https://github.com/angry0tiger-sudo/AliX/issues/43).

## Context

Post-hack acquisition Run #5 (`43379942240`) passed the nonce and `PENDING_DISPATCH` gates, then failed while the wrapper validated returned timestamps. The old `_validate_record_timestamps` raised immediately on the first off-grid timestamp, before `fetch_window` computed its complete missing-timestamp list. Consequently, the run did not produce a combined gap/off-grid diagnostic artifact.

The failure is an infrastructure/validation reportability problem until evidence establishes whether the provider grid is systematically offset. No holdout-window timestamp list was available from Run #5, and this ADR does not infer the anomaly's exact location.

## Decision: report both gap and grid anomalies

1. The post-hack wrapper collects all in-range off-grid timestamps and their offset `(timestamp - start_timestamp) % 3600` rather than aborting immediately.
2. It computes the complete missing grid list after the full fetch.
3. If either list is non-empty, it writes a structural report containing the complete missing and off-grid timestamps, UTC representations, and distinct offsets. It writes no OHLCV values and creates no CSV.
4. The operation remains fail-closed: any gap or off-grid anomaly returns a non-zero process status. The `ZERO_TOLERANCE` gap policy, no-backfill/no-interpolation/no-masking rules, holdout boundaries, expected count, and public fetcher are unchanged.

## Decision: separate non-holdout DIAGNOSE mode

A dedicated `DIAGNOSE` trigger value is handled before nonce parsing and before V2 status validation. This branch does not consume or read a dispatch nonce and never selects the acquisition step. The acquisition branch remains guarded by the existing nonce-history, push-event, and `PENDING_DISPATCH` checks.

The only permitted diagnostic interval is:

- Start: `2026-11-02T00:00:00Z` / `1793577600`.
- End: `2026-11-12T00:00:00Z` / `1794441600`.
- Timeframe: UTC 1h; 241 inclusive timestamp positions.
- It is strictly after the committed holdout end `2026-11-01T00:00:00Z` / `1793491200`, and strictly after the pre-hack reference window.

The previously copied values `1762041600` and `1762905600` correspond to November 2–12, 2025, which lies inside the committed 2025–2026 holdout. They are rejected by the fixed-range validator and are not used.

The diagnostic reuses `scripts.fetch_nobitex_posthack_ohlcv.fetch_window` with explicit CLI boundaries; it does not load the holdout spec to choose a range. Only HTTP status, candle count, off-grid count, distinct offsets, first/last timestamps, and gap count are printed or persisted. Candle values are transiently parsed in memory by the shared public-data parser but are not printed, returned in the diagnostic record, or persisted. No prices or OHLCV fields are written to any artifact.

## Safety and dispatch controls

- V2 remains `PENDING_DISPATCH`; its commitment fields are not modified by this ADR or by DIAGNOSE.
- Existing used nonces remain permanently retired. No new nonce is generated or consumed in this diagnostic phase.
- The dedicated workflow retains `contents: read` only, uses pinned GitHub Actions SHAs, and requires no secrets.
- No holdout acquisition, training, fitting, tuning, backtest, economic evaluation, promotion, or trading authorization is granted.
- `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, `private_api_calls=0`, and `real_order_attempts=0`.
- The frozen `main`, public fetcher, and committed holdout specification remain untouched.
