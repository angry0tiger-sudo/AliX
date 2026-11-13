# Post-Hack Nobitex Holdout Dispatch Runbook

## Purpose and current state

This runbook governs the one-time, owner-authorized acquisition of the Nobitex BTC/USDT Spot hourly holdout. Spec v3 is the active dispatch contract; the v2 section below is retained as historical evidence.

- Branch: `research/nobitex-hf-strategy`
- Workflow: `.github/workflows/nobitex-posthack.yml`
- Trigger file: `config/holdout/DISPATCH_TRIGGER`
- Safe trigger after every run: `PROBE`
- Commitment: `HOLDOUT_COMMITMENT_V2.md`
- Commitment status before acquisition: `PENDING_DISPATCH`
- Tracking: [Issue #43](https://github.com/angry0tiger-sudo/AliX/issues/43)
- Decision: [ADR-0016](../architecture/decisions/ADR-0016-nobitex-venue-grid-convention.md)

`main` is frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`. No pull request may be merged to it.

## Historical spec v2: fixed window and venue grid (superseded by v3)

- Spec: `config/holdout/posthack_window_spec.json`
- Version: `2`
- Venue convention: hourly candle-open timestamps at `:30 UTC`, corresponding to Iran's UTC+3:30 local hour boundaries.
- Start: `2025-07-01T00:30:00Z` inclusive / `1751329800`.
- End: `2026-11-01T00:30:00Z` inclusive / `1793493000`.
- Expected rows: `11,713`.
- Minimum separation after the pre-hack end: `1,137,600` seconds / 316 hours.
- Endpoint: `https://apiv2.nobitex.ir/market/udf/history`.
- Timeframe: 1h; timestamps are inclusive candle-open Unix seconds.
- Pagination: maximum 400 candles; provider `to` boundary is exclusive.
- Exact timestamp coverage is checked relative to the spec's :30 start; sort ascending and deduplicate exact timestamp keys before coverage validation.
- Gap policy is zero tolerance. No backfill, interpolation, masking, synthetic fill, resampling, or automatic window change.
- Cost scenarios, economic gates, purge/embargo controls and data-use lock are unchanged from the prior preregistration.

The v2 convention is based on the pre-hack source timestamps and adjacent non-holdout DIAGNOSE Run #7. It is a structural convention correction: both timestamps move by the same 30 minutes, while the nominal date span, total duration and 11,713 row count remain the same. It is not window-shopping.

## Active spec v3: boundary adjustment after documented gap

- Spec: `config/holdout/posthack_window_spec.json`
- Version: `3`
- Revision reason: `boundary adjustment after documented gap`; the 13 leading :30 timestamps listed in Run #12 gap artifact `14045205264` were absent at the fixed window start. This is not window shopping and is not based on economic/model results.
- Start: `2025-07-01T13:30:00Z` inclusive / `1751376600`.
- End: `2026-11-01T00:30:00Z` inclusive / `1793493000` (unchanged).
- Expected rows: `11,700`.
- Minimum separation after the pre-hack end (`1750192200`): `1,184,400` seconds / 329 hours.
- Canonical spec SHA-256: `4c2b5a55691d00da67a6ccd875ad66b5559b5d06408ec2f7f08ed5eae4a9790b` (Standard CI #2349, Run ID `43440748067`).
- Gap policy, cost scenarios, economic gates, purge/embargo controls and data-use lock are unchanged. No backfill, interpolation, masking, synthetic fill, resampling, or automatic further window adjustment is permitted.

## CI verification and spec fingerprint

Standard CI logs the pre-hack start/end values and their modulo 3600, asserts both are 1800, validates the v2 boundaries/count/separation, and prints the canonical sorted-key JSON SHA-256. The dedicated posthack workflow independently writes the same canonical spec SHA to its structural artifact.

Before dispatch, the latest top-level `spec_sha256:` line in V2 must match the hash computed by the exact commit's GitHub Actions run. Do not calculate or substitute the hash manually.

## PROBE behavior

The exact trigger value `PROBE` performs one small public GET. It records only HTTP status, endpoint, latency, run ID, commit SHA, and spec SHA. It does not acquire the window.

- HTTP 200: the endpoint answered the bounded probe. This does not prove full-window coverage.
- HTTP 403 or 451: `GEOBLOCKED`; stop. No proxy, VPN, runner geography workaround, or bypass is allowed.
- Other status/network errors: stop and inspect structural probe metadata.

## Historical fourth nonce: CI-only rotation and commitment

Nonces #1, #2 and #3 are retired permanently and must never be reused:

- #1: `3764030460fa5cbdaebf522c97453b3b`
- #2: `d86c6da4e1df4e7b4b042a5be940df24`
- #3: `c1c9297040cc4bfcb28ac1f414c4ad07`
- #4: `e3fb1601e654341a21fa13f00cd6271b`

The fifth nonce is generated only in GitHub Actions; the following four previous nonces are permanently retired:

1. Set the trigger to the exact literal `NONCE_ROTATION` in a one-file trigger commit. This mode runs no PROBE or acquisition.
2. The dedicated workflow uses `secrets.token_hex(16)`, rejects any collision with the retired set, and writes the candidate to `artifacts/posthack/nonce_candidate.txt`. The value is not printed in CI logs.
3. Download the CI artifact and verify the candidate is exactly 32 lowercase hexadecimal characters, differs from all three retired values, and is the candidate emitted by the exact run.
4. Append a section `Spec Revision v2 (2026-11-13)` to V2 containing the CI-computed `spec_sha256:`, ADR-0016 reference, pre-hack modulo evidence, DIAGNOSE Run #7 reference, and an independent parser-compatible line `dispatch_nonce: <fourth nonce>`.
5. Do not edit or remove any older V2 history. Keep `status: PENDING_DISPATCH`, `dataset_sha256: PENDING_DISPATCH`, `row_count: PENDING_DISPATCH`, and `run_id: PENDING_DISPATCH`.
6. After Standard CI is green on the V2 append commit, replace `NONCE_ROTATION` with that fourth nonce in a separate one-file trigger commit. That push is the sole authorization event for the one-time acquisition.

The acquisition gate requires a push event, a fresh 32-hex nonce matching the latest V2 key, no use of that nonce in trigger-file history, V2 still pending, and the latest committed spec SHA matching the workflow-computed hash. Do not use the Actions Run workflow button for acquisition. Do not retry a failed acquisition or reuse its nonce.

## Fifth nonce: CI-only rotation and commitment

- Fifth nonce: `5f5f3083a55f6c07bac0551473844853`.
- Generated by dedicated Workflow Run #14 / ID `43441496367`; artifact ID `14044658921`.
- Standard CI #2350 / Run ID `43441496425` completed successfully on the trigger-rotation commit before this nonce was appended.
- The nonce is exactly 32 lowercase hexadecimal characters and differs from all four retired values.
- The latest parser-compatible `dispatch_nonce:` line in V2 is the fifth nonce; the status remains `PENDING_DISPATCH`.
- Before the acquisition push, Standard CI must pass on the V2 append commit. The separate trigger commit containing this nonce is the only acquisition authorization event.

## Acquisition result handling

The wrapper validates the exact 11,700-point v3 :30 grid. It has zero tolerance for missing or off-grid timestamps.

- If gaps or grid anomalies exist, the workflow writes a structural artifact with complete missing/off-grid timestamps, UTC representations, offset metadata and counts; it exits non-zero. There is no backfill, interpolation, masking, or window adjustment.
- If the data is complete, it computes the canonical dataset digest in memory and writes a structural manifest including actual row count and SHA-256.
- Per the owner directive, raw OHLCV and price values are never printed or persisted. The successful manifest records `dataset_persisted=false`; no CSV is written to the artifact.
- Any gap or technical failure leaves V2 unchanged at `PENDING_DISPATCH`. Reset the trigger to `PROBE` in a separate one-file commit and stop. A further gap after the v3 boundary correction is terminal for this authorization: STOP and ABORT per ADR-024; do not shift the window again or reuse the nonce.
- Only a fully successful acquisition with inspected artifact/manifest permits an append-only V2 transition to `FETCHED_LOCKED` with the actual dataset SHA, row count and acquisition run ID. Then reset the trigger to `PROBE` and append the final manifest reference to Issue #43.
- No training, fitting, tuning, backtest, or economic evaluation is permitted. The holdout remains sealed pending separate owner Release Authorization.

## After success: download the artifact before expiry

GitHub Actions artifacts have a 30-day retention period for this workflow.

Immediately after a successful acquisition:

1. Download the complete artifact ZIP from the exact acquisition run before its expiry.
2. Preserve the original ZIP, artifact ID, workflow run ID, commit SHA, and manifest alongside the project evidence.
3. Verify only structural metadata and the manifest digest/count. Do not print or extract OHLCV/price values into reports, logs, issue comments, or documentation.
4. Record the artifact download/preservation and manifest verification in the append-only V2 execution ledger and Issue #43.

The artifact is metadata-only under the no-OHLCV-persistence directive; the dataset SHA is computed from the canonical serialization in memory.

## Safety invariants

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- No credentials, private APIs, paper orders, real orders, or promotion.
