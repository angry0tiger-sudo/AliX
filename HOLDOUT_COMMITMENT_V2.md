# HOLDOUT COMMITMENT V2 — Nobitex BTC/USDT Spot 1h

status: PENDING_DISPATCH
spec_sha256: d5e18ebea68858ab8e9f1c1ccaf56577405e6b073cba6b880d8be9c22714c4c5
dataset_sha256: PENDING_DISPATCH
row_count: PENDING_DISPATCH
run_id: PENDING_DISPATCH
dispatch_nonce: 3764030460fa5cbdaebf522c97453b3b
probe_run_id: 43360654817
probe_http_status: 200
probe_artifact_id: 14007598627

## Immutable preregistration

- Active branch: `research/nobitex-hf-strategy`.
- Venue: Nobitex Spot; symbol: BTCUSDT; timeframe: 1h.
- Fixed inclusive candle-open window: 2025-07-01T00:00:00Z through 2026-11-01T00:00:00Z.
- Expected candles: 11,713.
- Window spec: `config/holdout/posthack_window_spec.json`.
- Spec hash is canonical sorted-key JSON, UTF-8 SHA-256, computed by GitHub Actions only.
- Pre-hack source hash: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- Purge/embargo separation from the last pre-hack candle-open: 1,135,800 seconds (315.5 hours).
- Gap policy: zero tolerance; no backfill, interpolation, masking, or synthetic fill.
- Cost scenarios: 50 and 90 bps.
- Economic gates: NetPF >= 1.10; trades >= 100; MaxDD <= 15%; DSR > 0; at least 3 stable folds.

## Sealing statements

هیچ دادهٔ واقعی post-hack دریافت نشده است.
هیچ مدل یا پژوهشگری این پنجره را ندیده است.
این پنجره فقط یک‌بار و با nonce ثبت‌شده باز می‌شود.
The nonce is a one-time dispatch identifier, not an API credential or secret.
The acquisition workflow must reject a replay by checking prior trigger-file history.

## Probe evidence

- Probe run ID: `43360654817` (workflow run #1).
- HTTP status: `200`.
- Endpoint: `https://apiv2.nobitex.ir/market/udf/history`.
- Latency: `1290.7 ms`.
- Structural artifact ID: `14007598627`.
- Commit SHA tested: `d2b2672955118291491afd64a731abc70522bd70`.
- Exactly one small GET was issued; the response body was not read, printed, or stored.
- This establishes endpoint reachability from the GitHub Actions runner for the probe request only. It does not establish complete historical-window coverage.

## Data-use lock and safety

- No model training, fitting, tuning, backtest, or economic evaluation is authorized by this commitment.
- Dataset access remains locked until a separate owner Release Authorization.
- CSV values, if ever acquired, are stored only in the Actions artifact; logs and manifests remain structural metadata only.
- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `private_api_calls=0`
- `real_order_attempts=0`

## Append-only execution ledger

No acquisition or holdout consumption event is recorded yet. Future evidence must be appended below;
do not erase the initial commitment or prior execution entries.

## Nonce Rotation (2026-11-13)

- The previous nonce `3764030460fa5cbdaebf522c97453b3b` was not used for data acquisition because the wrapper failed before fetching with `ModuleNotFoundError: No module named 'scripts'`.
- The previous nonce is permanently retired and must never be reused.
- Replacement `dispatch_nonce:` (active): d86c6da4e1df4e7b4b042a5be940df24

## Nonce Rotation #2 (2026-11-13)

- The second nonce `d86c6da4e1df4e7b4b042a5be940df24` was rejected because it was recorded inside a descriptive bullet rather than as a standalone parser-compatible `dispatch_nonce:` key. The workflow therefore retained the previous parsed nonce and rejected the mismatched trigger with `STALE_OR_REPLAYED_NONCE`.
- The third nonce below replaces the second nonce for this one-time acquisition dispatch. Both earlier nonces are permanently retired.
dispatch_nonce: c1c9297040cc4bfcb28ac1f414c4ad07
dispatch_nonce: e3fb1601e654341a21fa13f00cd6271b

## Spec Revision v3 (2026-11-13)

- Operation class: boundary adjustment after documented gap; structural availability correction only, with no economic evaluation or model selection.
- Prior spec v2 start: `2025-07-01T00:30:00Z` / `1751329800`.
- Run #12 evidence: [Workflow Run #12 / ID 43438007282](https://github.com/angry0tiger-sudo/AliX/actions/runs/43438007282), conclusion `failure`, status `GAP_DETECTED`.
- Structural gap report: artifact ID `14045205264`, file `nobitex-btcusdt-posthack-1h.gaps.json`; 13 missing leading timestamps from `2025-07-01T00:30:00Z` through `2025-07-01T12:30:00Z`, actual rows 11,700 against 11,713 expected, `off_grid_count=0`, no grid anomaly.
- New spec v3 start: `2025-07-01T13:30:00Z` / `1751376600`; end unchanged: `2026-11-01T00:30:00Z` / `1793493000`.
- Inclusive expected candle count: `11,700`.
- Minimum separation from `PREHACK_END_TIMESTAMP=1750192200`: `1,184,400` seconds / 329 hours. This is the exact arithmetic difference from the v3 start.
- Revision reason: boundary adjustment after documented gap; the first 13 expected :30 timestamps were absent at the beginning of Run #12. No return, price, strategy, or economic metric informed this boundary.
- Canonical spec SHA-256, computed by Standard CI #2349 (Run ID `43440748067`) on commit `93861880d1bd24e880ad4effb66fe0537feb4776`:

spec_sha256: 4c2b5a55691d00da67a6ccd875ad66b5559b5d06408ec2f7f08ed5eae4a9790b

- Previous nonce #4 `e3fb1601e654341a21fa13f00cd6271b` is permanently retired. Fifth nonce generated by CI-only rotation in Workflow Run #14 (Run ID `43441496367`), artifact ID `14044658921`:

dispatch_nonce: 5f5f3083a55f6c07bac0551473844853

## Acquisition v3 Result (2026-11-13)

- Acquisition Workflow: [Run #15 / ID 43443886254](https://github.com/angry0tiger-sudo/AliX/actions/runs/43443886254).
- Acquisition commit: `845fe23c019f945a5d5d22c610871317d7d76eb2`.
- Result: `COMPLETE`; all v3 structural gates passed.
- Spec version: `3`; spec SHA-256: `4c2b5a55691d00da67a6ccd875ad66b5559b5d06408ec2f7f08ed5eae4a9790b`.
- Dispatch nonce: `5f5f3083a55f6c07bac0551473844853` (fifth nonce; permanently consumed).
- Start: `2025-07-01T13:30:00Z` / `1751376600`.
- End: `2026-11-01T00:30:00Z` / `1793493000`.
- Expected and actual rows: `11,700`; missing timestamps: `0`; off-grid: `0`; duplicates: `0`; grid anomaly: `false`.
- Dataset SHA-256 (canonical serialization computed in memory): `435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee`.
- `dataset_persisted=false`; artifact contains structural manifest and spec fingerprint only, no CSV or raw OHLCV.
- Artifact ID: `14046363033`; artifact ZIP SHA-256: `4053be1a4e0af8ca73d05ae0c6f2524933fd231d60a1b0b4b6a6c0518db7b290`.
- Manifest: `nobitex-btcusdt-posthack-1h.manifest.json`.
- Standard CI #2352 / Run ID `43443886284`: `SUCCESS` on acquisition trigger commit.
- Safety recorded by manifest: `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, `private_api_calls=0`, `real_order_attempts=0`.

status: FETCHED_LOCKED
dataset_sha256: 435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee
row_count: 11700
run_id: 43443886254
