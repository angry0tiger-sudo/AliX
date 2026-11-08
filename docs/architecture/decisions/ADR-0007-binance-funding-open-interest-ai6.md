# ADR-0007: Binance USD-M Funding-Only Research and V4-AQ Meta Features

- Status: Accepted for bounded internal Funding-only research; no production or trading promotion.
- Date: 2026-11-08.
- Owner authorization: docs/governance/owner-directives/2026-11-08-ai6-binance-public-market-data.md.
- Branch: research/ml-core-v4aq-ai6.

## Context

V4-AP tested 36 preregistered Primary-only combinations across BTCUSDT/ETHUSDT, 15m/1h/4h, two Primary strategies, and three exit geometries at 30 bps round-trip cost. Its measured result was NO STATISTICAL EDGE FOUND IN CURRENT FEATURE SET.

AI-6 initially proposed both Binance Funding Rate and Open Interest History. The public OI Statistics endpoint only supports a short recent window and cannot provide the required 1825-day series. A 30-day OI slice cannot be joined to five years of OHLCV without severe coverage bias.

## Decision

1. Drop Open Interest from this phase. The active adapter, fetcher, manifests, feature engineering, model schema, workflow, and tests must not fetch or use OI.
2. Retain only Binance USD-M public Funding Rate History at /fapi/v1/fundingRate, using unauthenticated GET requests for BTCUSDT and ETHUSDT.
3. Persist raw funding settlement times, conservative available_at proxies, content hashes, CSV hashes, exact requested window, and coverage diagnostics.
4. Derive causal funding_rate_zscore and funding_rate_ma features with a 30-settlement rolling window, minimum 20 observations, a 48-hour gap reset, and a 48-hour maximum as-of age.
5. Keep Primary signal generation independent of Funding. Funding is strictly Meta-Model context and may filter existing Primary BUY events only.
6. Reuse V4-X purged OOF and calibration contracts and V4-AP's existing trade simulator, RSI/EMA and Donchian20 Primary definitions, TIGHT/BALANCED/TRAILING geometries, documented market-gap policy, and exact 30 bps round-trip cost model.
7. Evaluate the complete 36-cell V4-AP grid. Fit 12 Meta models grouped by asset/timeframe/Primary strategy, and reuse each out-of-sample filtered event stream for the three exit geometries.
8. Freeze the Meta threshold from in-training calibration. Score ROC-AUC on forward labels over the 70%-85% interval. The final 15% remains untouched.
9. Require >=50 trades, >=40% win rate, Net PF >=1.10 and DSR Z >0 for a passing cell. DSR uses 36 nominal trials; independence is not verified.

## Consequences

- Funding history must independently demonstrate at least 90% estimated expected settlements, boundary coverage within one day, and maximum event gap <=48 hours before the model fit.
- Insufficient Funding coverage is a blocked diagnostic, not evidence of no economic edge. No fit or economic replay may occur.
- A Net PF threshold pass alone does not validate or promote a candidate.
- AI-6 workflow scripts execute with python -m scripts.<module> so the repository root remains available for package imports.
- Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, and real_order_attempts=0 remain mandatory.
- Provider authorization is limited to bounded internal research. Public endpoint availability does not establish commercial-use or redistribution rights.
