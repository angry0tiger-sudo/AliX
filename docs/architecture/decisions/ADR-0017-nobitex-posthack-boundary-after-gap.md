# ADR-0017: Adjust the Nobitex Post-Hack Holdout Start After a Documented Boundary Gap

- Status: Accepted for the active research branch under the explicit owner directive dated 2026-11-13.
- Date: 2026-11-13.
- Active branch: `research/nobitex-hf-strategy`.
- Frozen main: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (unchanged).
- Governance source: AliX MRD v0.9.
- Related decisions: [ADR-0014](ADR-0014-nobitex-posthack-holdout-push-nonce.md), [ADR-0015](ADR-0015-non-holdout-nobitex-grid-diagnosis.md), [ADR-0016](ADR-0016-nobitex-venue-grid-convention.md).
- Tracking: [Issue #43](https://github.com/angry0tiger-sudo/AliX/issues/43).

## Context and verified evidence

The v2 fixed :30-grid acquisition was executed once under the owner-authorized nonce:

- Workflow Run #12 / ID `43438007282`, commit `32a2b32df73d9fd1b135c112325639399e4fa9a9`, conclusion `failure`.
- Terminal manifest status: `GAP_DETECTED`.
- Expected rows: 11,713; actual rows: 11,700; `gap_count=13`; `off_grid_count=0`; `grid_anomaly=false`.
- Gap report artifact: ID `14045205264`, file `nobitex-btcusdt-posthack-1h.gaps.json`.
- The only missing timestamps are the first 13 expected :30 candle-open timestamps: 2025-07-01 00:30 UTC through 12:30 UTC inclusive. The first subsequent expected candle-open is 2025-07-01 13:30 UTC.
- The gap report contains timestamp metadata only. No price/OHLCV value was printed or persisted.
- The subsequent bounded PROBE Run #13 / ID `43438307809` succeeded; this establishes endpoint reachability for the probe only, not full-window coverage.

## Decision

Adopt post-hack holdout spec v3 with a boundary adjustment justified only by the documented leading gap:

- `spec_version=3`.
- `start_utc=2025-07-01T13:30:00Z` / `1751376600`.
- `end_utc=2026-11-01T00:30:00Z` / `1793493000` (unchanged).
- Inclusive expected candle count: `11,700`.
- Grid remains hourly candle-open timestamps at :30 UTC.
- Revision reason: `boundary adjustment: skip 13 missing candles at window start (2025-07-01 00:30-12:30 UTC); not window shopping — gap at boundary documented in Run #12`.
- Canonical spec SHA-256: `4c2b5a55691d00da67a6ccd875ad66b5559b5d06408ec2f7f08ed5eae4a9790b`, computed by Standard CI #2349 / Run ID `43440748067`.
- Pre-hack end remains `PREHACK_END_TIMESTAMP=1750192200`. Therefore the exact minimum separation is `1751376600 - 1750192200 = 1,184,400` seconds / 329 hours.

**Arithmetic clarification:** 1,139,400 seconds is not consistent with the specified v3 start and pre-hack end. The derived value is 1,184,400 seconds; the spec, wrapper and CI assert that exact value.

This revision is not window shopping: the sole selection criterion is the 13-point structural gap located at the beginning of the fixed window in Run #12. No trade performance, return, model metric, cost result, or other economic outcome was examined or used to choose a boundary. The end date remains unchanged.

## Integrity and preregistration controls

- Gap policy remains `ZERO_TOLERANCE`; backfill, interpolation, masking and synthetic fill remain forbidden.
- Cost scenarios (50 and 90 bps), economic gates, purge/embargo controls and `data_use_lock` remain unchanged.
- The wrapper still validates the exact inclusive :30 grid and fails closed on any further missing or off-grid timestamp.
- Standard CI #2349 passed on the v3 code/spec commit; 931 tests passed, 5 were skipped, and total coverage was 90.26%.
- The fifth nonce `5f5f3083a55f6c07bac0551473844853` was generated only in CI Workflow Run #14 / ID `43441496367`, artifact ID `14044658921`. Nonces #1–#4 are permanently retired.
- V2 remains `PENDING_DISPATCH` until the one-time v3 acquisition completes and the manifest is verified. No release authorization is implied by this ADR.

## Explicit non-observation and safety

No training, fitting, tuning, backtest, or economic evaluation was performed. The v3 acquisition has not been dispatched by this decision record. No private API or order action is authorized.

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`

The public fetcher remains unchanged at blob `01638aeeda04ee7d19584009780e88dc68a395bc`. Frozen `main` remains unchanged at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
