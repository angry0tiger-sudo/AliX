# ADR-0013: Bybit Spot Kline Probe — DATA-BLOCKED from GitHub Actions

- Status: Accepted — do not build the Bybit fetcher on current evidence.
- Date: 2026-11-12.
- Active branch: `research/nobitex-hf-strategy`.
- Decision commit parent: `fb9bacf88bab0ae0ee74f7b134e00164e44da0d7`.
- Probe workflow: `.github/workflows/bybit-spot-kline-probe.yml`.
- Probe workflow blob SHA: `0cf91410653ca0b64213b9df6befe20168d47241`.
- Probe commit: `fb9bacf88bab0ae0ee74f7b134e00164e44da0d7`.
- Probe run: [GitHub Actions #1 / 43284092706](https://github.com/angry0tiger-sudo/AliX/actions/runs/43284092706).
- Probe job ID: `131412768886`.
- Full probe report: `docs/research/BYBIT_SPOT_KLINE_PROBE.md`.

## Context

The Binance Spot 15m fetch in HF Research #41 failed closed on a 90-minute timestamp discontinuity
between 2023-03-24 12:30 and 14:00 UTC. Bybit Spot V5 Kline was proposed as a research-only
alternative, but prior runner exploration indicated geographic HTTP 403 blocking. A full fetcher
must not be built until the exact public Spot endpoint is proven reachable and the required window
is complete.

## Evidence

The bounded probe made exactly two unauthenticated public GET requests to:

`https://api.bybit.com/v5/market/kline`

Request A:

- `category=spot`
- `symbol=BTCUSDT`
- `interval=15`
- `limit=5`

Observed: HTTP 403, zero records, no timestamps. Response excerpt:
`The Amazon CloudFront distribution is configured to block access from your country`.

Request B:

- `category=spot`
- `symbol=BTCUSDT`
- `interval=15`
- `start=1679661000000` (2023-03-24 12:30:00 UTC)
- `end=1679666400000` (2023-03-24 14:00:00 UTC)
- `limit=1000`

Observed: HTTP 403, zero records, no earliest/latest timestamp, and no usable historical
payload. The exact seven-bar grid cannot be validated.

Run evidence:

- Run ID `43284092706`, run number #1, commit
  `fb9bacf88bab0ae0ee74f7b134e00164e44da0d7`.
- Job ID `131412768886`.
- Both requests and report generation completed; the workflow reports success because blocked
  access is a valid diagnostic outcome, not because Bybit returned data.
- `private_api_calls=0`; `real_order_attempts=0`.

## Decision

1. Mark the Bybit Spot HF proxy path **DATA-BLOCKED from the current GitHub Actions runner**.
2. Do not build or add a complete Bybit fetcher under the current evidence.
3. Do not retry through alternate geography, proxies, tunnels, VPNs, or any workaround for the
   provider's geographic restriction.
4. Do not splice Binance and Bybit data, mask gaps, interpolate, backfill, synthesize candles,
   or shift the requested historical window.
5. Reconsider only after a new, separately authorized and policy-compliant public-data access
   path is independently demonstrated. Any change in venue/window is a new hypothesis and must
   be preregistered before evaluation.

## Scope and limitation

This is not a claim that Bybit is globally unavailable. The evidence establishes that both
requests were blocked from the GitHub Actions runner used in this probe. Because no payload was
returned, the historical seven-candle completeness question remains unobservable.

The native Nobitex 1h dataset remains a separate source and is not affected by this Bybit
runner-path result. Binance remains a separate proxy with its own documented gap. No cross-venue
transferability is assumed.

## Consequences

- The HF proxy path is `DATA-BLOCKED` for Bybit from the current Actions runner.
- No Bybit fetcher or related pagination/manifest implementation is authorized by this ADR.
- The four-arm Kronos/Chronos experiment is governed separately by the holdout-integrity decision
  recorded in `HOLDOUT_COMMITMENT.md`; the existing native dataset cannot be relabeled as an
  untouched holdout.
- A successful CI job for this probe must not be represented as successful data access.

## Safety invariants

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `private_api_calls=0`
- `real_order_attempts=0`
- No credentials, no orders, no live trading, no model promotion.
- `main` remains frozen and untouched.
