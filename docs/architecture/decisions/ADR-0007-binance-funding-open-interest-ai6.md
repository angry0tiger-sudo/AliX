# ADR-0007: Binance Public Funding/OI Research and V4-AQ Meta Features

- Status: Accepted for bounded internal research implementation; full-history economic evaluation blocked until data coverage is evidenced.
- Date: 2026-11-08.
- Owner authorization: `docs/governance/owner-directives/2026-11-08-ai6-binance-public-market-data.md`.
- Base: V4-AP SHA `798ef2d49bcbc715d35b16862a2cf83aa0865486`.
- Branch: `research/ml-core-v4aq-ai6`.

## Context

V4-AP ran all 36 preregistered Primary-only combinations across ETHUSDT/BTCUSDT, 15m/1h/4h, two Primary strategies and three exit geometries at 30 bps round trip. The exact Step Summary concluded:

`NO STATISTICAL EDGE FOUND IN CURRENT FEATURE SET.`

The next diagnostic investigates whether derivatives positioning context improves Meta-Labeling. The initial source is Binance USD-M Futures public market data only.

## Decision

1. Implement an unauthenticated read-only adapter for public Funding Rate and 4h Open Interest History.
2. Keep the adapter isolated from Freqtrade, order lifecycle, account credentials, Risk Gate and Kill Switch.
3. Persist raw observation times, conservative `available_at` assumptions, source hashes, CSV hashes and window-coverage diagnostics.
4. Add causal `funding_rate_zscore` and `oi_change` features using backward as-of joins only.
5. Add a fixed V4-AQ LightGBM Meta-Model feature schema that reuses V4-X OOF, purge and calibration mechanics. The Primary remains Donchian20.
6. Keep V4-AP's BALANCED geometry, 30 bps round-trip costs and 1825-day target window.
7. Block the model fit and economic replay unless required external coverage is complete. No missing OI history is interpolated or backfilled.

## Provider limitation

The public Open Interest History endpoint is limited to the latest 30 days. Consequently this provider alone cannot satisfy the five-year OI data requirement. A partial 30-day series must not be presented as five-year training data. An alternate archive/provider requires separate rights review and owner authorization.

## Consequences

- A blocked coverage diagnostic is a valid successful workflow outcome and must state `BLOCKED_INSUFFICIENT_EXTERNAL_HISTORY`.
- Funding and OI coverage are reported separately.
- A single selected cell does not receive a DSR claim. Net PF >=1.10 alone cannot promote a model.
- `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, and `private_api_calls=0` remain mandatory.
- The existing Phase 6 directive is superseded only for the exact unauthenticated public GET endpoints and symbols listed in the new Owner Directive. All other restrictions remain effective.
- No provider/commercial/redistribution rights are inferred from public endpoint availability.
