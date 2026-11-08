# Owner Directive V4-AP: Final Comprehensive Primary-Only Diagnostic Sweep

- **Directive:** V4-AP
- **Status:** Implemented on the dedicated research branch; economic verdict pending GitHub Actions evidence.
- **Base:** `research/ml-core-v4ao-donchian` at `9a17aa082e22742e29c2866e6afc834103bc622f`
- **Branch:** `research/ml-core-v4ap-diagnostic-sweep`
- **Scope:** Research only. No model promotion and no trading authorization.

## Preregistered matrix

The sweep contains exactly 36 unique cells:

- Assets: ETHUSDT and BTCUSDT.
- Timeframes: 15m, 1h, and 4h.
- Primary signal rules: RSI(14) strictly below 30 while Close is strictly above EMA(200); or a close strictly above the highest high of the prior 20 completed bars.
- Exit geometries: Tight (SL 1.0 ATR / TP 1.5 ATR), Balanced (SL 1.5 ATR / TP 3.0 ATR), and Trailing (SL 2.0 ATR / TP 10.0 ATR / activation 1.0 ATR / step 1.5 ATR).
- Costs: 10 bps fee and 5 bps slippage per side, exactly 30 bps round trip.
- Market data: one shared 1825-day UTC interval per run, Binance Spot public 1m candles, aggregated into complete UTC epoch-aligned analysis bars.

## Explicit implementation assumptions

The owner directive does not specify a vertical time barrier. To keep elapsed exposure comparable across timeframes and preserve the V4-AO reference duration, the sweep uses a nominal 24-hour horizon: 96 complete 15m bars, 24 complete 1h bars, and 6 complete 4h bars. Incomplete analysis bars are dropped, never interpolated. An open position is flattened at the first available open after an analysis-bar data discontinuity. For ETHUSDT, only the already documented 80-minute source gap from 2023-03-24 12:40Z through 13:59Z is accepted; both sides of the gap are retained.

Signals are evaluated on completed candles and executed at the next available analysis candle open. Entry events that would cross a missing analysis interval are suppressed. The last 24 hours of new entries are suppressed so each admitted entry has the full nominal horizon inside the dataset. Positions do not pyramid; the simulator remains Spot-long only.

DSR is calculated on net per-trade returns, not annualized. The nominal trial count is fixed at 36. The existing V4-AO expected-maximum-Sharpe method is applied using the available valid Sharpe estimates to estimate the Sharpe distribution and all 36 preregistered trials for the expected-maximum term. Trial independence is not verified and is stated in the report. This is a multiple-testing diagnostic, not proof of independent trials or a guarantee of future performance.

## Decision rule

A cell qualifies only if all four owner thresholds pass:

1. Trades >= 50.
2. Win Rate >= 40%.
3. Net Profit Factor >= 1.10 after 30 bps round-trip costs.
4. DSR Z > 0.0.

The generated Step Summary must contain all 36 cells. When no cell passes all four conditions, it must print exactly:

`CONCLUSION: NO STATISTICAL EDGE FOUND IN CURRENT FEATURE SET.`

## Safety and controls

- Meta-Model: disabled.
- HMM: not imported or used.
- CUSUM: not imported or used.
- LSTM: not imported or used.
- RL: not imported or used in this sweep.
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- No weakening or bypass of Standard CI, test coverage, lint, format, or type-check gates.
- All execution and research run in GitHub Actions; no local backtest or local test execution is part of this directive.

## Pivot mapping and MRD traceability

The current MRD explicitly documents External Data & Social Media connectors, provider approval, point-in-time provenance, legal/rate-limit review, External Feature Store separation, and Data Quality Gate requirements. These are consistent with a future external-data pivot, including approved funding and order-book features. The current MRD and roadmap files reviewed for this change do not contain a clearly labeled section named `Phase AI-6`; this directive therefore records the owner's AI-6 reference as an owner-supplied phase label rather than misquoting the current document.

The repository contains an isolated RL research area, but RL is not part of this diagnostic sweep. Any pivot requires a separate architecture/research directive, dataset and label design, causal feature availability, acceptance gates, and isolated validation.

## Evidence required before closure

- Exact commit SHA and green Standard CI for that SHA.
- Successful V4-AP economic workflow run.
- Downloadable report JSON, full trade CSV, and Step Summary artifact.
- Validation that all 36 candidate IDs are unique and present, all costs are 30 bps, and safety invariants remain closed.
- Explicit publication of the preregistered outcome. Workflow success alone is not evidence of statistical edge.
