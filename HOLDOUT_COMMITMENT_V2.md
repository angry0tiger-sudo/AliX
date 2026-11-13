# HOLDOUT COMMITMENT V2 — Nobitex BTC/USDT Spot 1h

status: PENDING_DISPATCH
spec_sha256: PENDING_CI_COMPUTATION
dataset_sha256: PENDING_DISPATCH
row_count: PENDING_DISPATCH
run_id: PENDING_DISPATCH
dispatch_nonce: 3764030460fa5cbdaebf522c97453b3b
probe_run_id: PENDING_PROBE
probe_http_status: PENDING_PROBE
probe_artifact_id: PENDING_PROBE

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
