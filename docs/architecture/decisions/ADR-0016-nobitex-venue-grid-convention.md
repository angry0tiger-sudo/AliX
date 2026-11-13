# ADR-0016: Align the Nobitex Holdout Specification to the Venue's Minute-30 Hourly Grid

- Status: Accepted for the active research branch under the explicit owner directive dated 2026-11-13.
- Date: 2026-11-13.
- Active branch: `research/nobitex-hf-strategy`.
- Frozen main: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (unchanged).
- Governance source: AliX MRD v0.9.
- Related decisions: [ADR-0014](ADR-0014-nobitex-posthack-holdout-push-nonce.md), [ADR-0015](ADR-0015-non-holdout-nobitex-grid-diagnosis.md).
- Tracking: [Issue #43](https://github.com/angry0tiger-sudo/AliX/issues/43).

## Context and verified evidence

The committed pre-hack data contract defines the public Nobitex hourly candle-open boundaries:

- `PREHACK_START_TIMESTAMP=1635712200`; `timestamp % 3600 = 1800`.
- `PREHACK_END_TIMESTAMP=1750192200`; `timestamp % 3600 = 1800`.

The adjacent, explicitly non-holdout DIAGNOSE Run #7 (`43398908810`) independently observed the same convention. The public endpoint returned HTTP 200 and 240 candles. All 240 timestamps had a unique offset of 1,800 seconds relative to the prior :00 UTC hourly grid; that :00 grid consequently had 241 missing timestamps. The artifact (`14024246276`) contains structural metadata only.

This is evidence about the provider's timestamp convention from pre-hack source metadata and an adjacent non-holdout diagnostic. It does not rely on inspecting, evaluating, or optimizing the holdout data.

## Decision

The canonical post-hack holdout spec is revised to version 2 and defines:

- `grid_convention`: `hourly candles timestamped at :30 UTC (Iran UTC+3:30 local hour boundaries)`.
- Start: `2025-07-01T00:30:00Z` / `1751329800`.
- End: `2026-11-01T00:30:00Z` / `1793493000`.
- Inclusive expected candle count: `11,713`.
- Minimum separation after the last pre-hack candle-open: `1,137,600` seconds / 316 hours.

Both boundaries are shifted by the same 30-minute convention offset. The nominal calendar date span and total duration are unchanged; this is not window-shopping or a change selected from holdout outcomes. The timestamp grid is aligned to the observed venue convention.

The wrapper now validates both endpoints on the :30 grid and uses the spec-defined start as the hourly alignment origin. Mock-only tests cover :30 success, :00 off-grid failure with complete structural reporting, and gaps on the :30 grid with the full missing list. The zero-tolerance gap policy remains fail-closed.

## Integrity and preregistration controls

- `expected_candle_count=11713` is asserted by both the canonical CI hash script and the wrapper.
- The 316-hour minimum separation and existing purge/embargo logic remain unchanged in policy.
- Gap policy, backfill prohibition, interpolation prohibition, masking prohibition, cost scenarios, economic gates and data-use lock are unchanged.
- The canonical spec SHA-256 is computed in GitHub Actions; the latest `spec_sha256:` line in V2 must match that value before acquisition can pass its gate.
- V2 remains `PENDING_DISPATCH` until the one-time acquisition has been completed and the resulting artifact/manifest have been verified.
- Nonces #1 (`3764030460fa5cbdaebf522c97453b3b`), #2 (`d86c6da4e1df4e7b4b042a5be940df24`), and #3 (`c1c9297040cc4bfcb28ac1f414c4ad07`) are permanently retired. The fourth nonce is generated only by the CI `NONCE_ROTATION` path, checked against the retired set, and appended as a new parser-compatible line.
- The successful acquisition path computes the canonical dataset hash in memory and records the row count and SHA in the manifest. In accordance with the owner directive, no CSV or raw OHLCV/price values are persisted in artifacts or logs; the manifest records `dataset_persisted=false`.
- The acquisition remains one-shot and fail-closed. Any gap or technical failure leaves the V2 status pending and the trigger reset to `PROBE`; there is no automatic retry or window adjustment.

## Explicit non-observation and safety

**No holdout data was observed** during this revision. The holdout window was not requested, fetched, opened, or passed to any model or evaluation process. Only source constants, adjacent non-holdout structural metadata, spec metadata and CI evidence were used.

No training, fitting, tuning, backtest, or economic evaluation was performed. No private API or order action is authorized.

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`

The public fetcher, HF workflow, Binance/Bybit paths, and frozen `main` remain untouched. Pull requests targeting `main` were closed per owner directive; none was merged.
