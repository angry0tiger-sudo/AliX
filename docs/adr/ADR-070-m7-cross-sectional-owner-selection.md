# ADR-070 — Owner Selection of Cross-Sectional Lead-Lag and M7 Preregistration

- **Status:** OWNER-APPROVED — M7 IMPLEMENTATION AND PUBLIC PRE-HACK DATA QUALIFICATION AUTHORIZED
- **Date:** 2026-11-27
- **Branch:** research/nobitex-hf-strategy
- **Owner directive:** select R&D Path B from ADR-069 and start M7.

## 1. Decision

The Owner has selected **R&D Path B: Cross-Asset Lead-Lag / Cross-Sectional Rotation**. The prior single-asset BTC direction-forecasting line remains closed. AliX itself remains OPEN.

M7 uses one interpretable pipeline only: StandardScaler + Ridge Regression with alpha=1.0. It predicts the next four-hour return for a fixed set of Nobitex Spot/USDT assets and rotates into at most the two highest strictly positive predicted returns. All unused capital remains in USDT.

Two logical candidates are registered prospectively and differ only by fixed universe size:
1. M7-5: ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT, XRPUSDT.
2. M7-10: the same five plus DOGEUSDT, TRXUSDT, LTCUSDT, LINKUSDT and DOTUSDT.

BTCUSDT is a contextual feature source, never a selected holding. The priority list is fixed before acquisition; no symbol replacement is permitted after observing coverage or outcomes.

## 2. Historical trial accounting

The focused inventory remains 12 rejected unique logical identities across HGBQ-CQR (3), M3 OFI (2), Regime T Triple-Barrier (1), M5.2 Sentiment (4), and M6 Triple-Barrier three-class (2). These are the focused identities reconciled in ADR-068, not the complete AliX historical trial count. GLOBAL_N remains UNKNOWN and GLOBAL_DSR remains NOT_COMPUTABLE.

M7 registers exactly two new logical identities, bringing the focused inventory to 14 identities (12 closed NO-GO + 2 M7 candidates). The two M7 candidates must not be described as successful or evaluated until their fixed data and folds have been processed by the approved workflow. Four folds and two cost scenarios are not additional trials.

## 3. Data authorization and qualification

The Owner authorizes public read-only acquisition of the fixed 1h PRE-HACK Nobitex Spot history for the ten candidate symbols on the fixed BTCUSDT window:
- Start: 2021-10-31T20:30:00Z (1635712200)
- End: 2025-06-17T20:30:00Z (1750192200)
- Expected rows per symbol: 31,801 inclusive hourly candles.

Only public /market/stats and /market/udf/history endpoints may be used. No API key, private endpoint, account or order access is permitted. The market must be listed and not marked closed in the public market-stats response. A symbol qualifies only if its hourly CSV has exactly the same timestamps as BTCUSDT, 31,801 rows, no gap, no duplicate, no off-grid event and valid OHLCV invariants. A candidate is data-blocked if any required symbol is absent, closed, lacks the full history, or fails integrity. There is no pair substitution, interpolation, resampling or gap filling.

Each qualified CSV and per-symbol manifest is immutable and SHA-256 bound. An aggregate qualification manifest records every requested symbol including unsuccessful/blocked requests. The BTCUSDT source remains bound to SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e.

The existing PRE-HACK history is research-exposed. As documented in HOLDOUT_COMMITMENT.md, rows [22308,31801) are contaminated by prior AI-6/DEAP use and may not be relabelled as an untouched holdout. M7 is exploratory only and must not access Holdout or Post-Hack data.

## 4. Model and decision policy

- Estimator: Pipeline(StandardScaler(), Ridge(alpha=1.0)).
- Features: BTC four-hour lagged log return; leave-one-out cross-sectional median four-hour return of other assets; rolling 24-hour beta to BTC from completed hourly returns; current completed-hour base-asset volume relative to the median of the prior 24 hourly volumes.
- Target: ln(open[t+5] / open[t+1]) after a decision at completed hour t, corresponding to a four-hour open-to-open return.
- Rebalance: every four hours anchored to the first fixed BTCUSDT timestamp.
- Selection: top two finite predictions strictly above zero, deterministic alphabetical tie-break; no shorting or leverage.
- Sizing: 50% initial equity per selected asset. If only one prediction is positive, only 50% is invested and the remainder remains in USDT; if none are positive, all capital remains in USDT.
- Model scaler and regressor are fit only on prior training blocks. No tuning, threshold sweep, feature search, model escalation, dynamic universe selection or post-result change is allowed.

## 5. Evaluation and economic gates

Four expanding chronological walk-forward folds are formed from five contiguous blocks. Training labels that cross the test boundary are purged, with a fixed four-hour embargo. No random splits or fitting inside test windows are allowed.

Cost scenarios are 50 bps and 90 bps round-trip. They remain registered modelling/stress assumptions, not verified realized Nobitex execution costs. Every selected asset held for one four-hour interval is one completed OOS round-trip trade; partial or overlapping holdings are not separately counted.

Each candidate must independently satisfy both cost scenarios:
- NetPF >= 1.10.
- At least 100 completed OOS asset-level trades.
- MaxDD <= 15% on the portfolio NAV curve.
- At least 3 valid folds (>=20 completed trades each).
- At least 3 stable folds, each with >=20 trades, positive net PnL and fold NetPF >1.00.

Both registered candidates must be reported, even if one is data-blocked or statistically ineligible. A blocked candidate is not a pass. Economic success on PRE-HACK remains exploratory, not confirmatory.

## 6. Safety and authority

This ADR authorizes only public pre-hack data qualification, the fixed M7 implementation, mock-only CI tests and the registered GitHub Actions research evaluation. It grants no further data source or execution permissions.

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- paper_orders = 0
- Holdout = SEALED
- holdout_bytes_read = false
- posthack_data_read = false
- local_execution = false

main remains frozen and must not be modified.

## 7. Change control

Any material deviation from the two fixed candidate universes, exact data interval, model, alpha, features, target, decision frequency, long-only selection, sizing, folds, cost scenarios, acceptance gates, or logical trial count requires a superseding preregistration before results are examined.

**Decision: R&D Path B selected; M7 research execution is authorized within the exact scope above. No Champion or Paper/Live permission is created.**


## 8. Pre-evaluation accounting clarification (M7-CS-G1.1)

During implementation review and before any fitting or economic output was observed, the return contract was clarified to maintain dimensional consistency: Ridge trains on the preregistered four-hour forward **log return**, while realized portfolio PnL and the 50/90 bps costs are calculated in **simple-return space**. Each asset's loss is floored at -100% of its own notional. The selected positions each use 50% of current NAV; NAV compounds every non-overlapping four-hour interval and trade-level PF contributions are measured in initial-equity terms using NAV immediately before the interval. This is a mechanical accounting correction, not a model, feature, threshold, universe, target-horizon, fold, or trial change. It was made before the first M7 outcome was inspected.
