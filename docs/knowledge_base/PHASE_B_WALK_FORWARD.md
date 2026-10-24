# Stage B: Gap-Aware Multi-Year Walk-Forward Protocol

## Status and scope

This protocol evaluates the fixed Phase 4 LightGBM baseline as an offline, fixed-horizon long-only research rule. It does not run the signal-free Freqtrade strategy, simulate the complete Risk Gate/order lifecycle, submit orders, or authorize a model. Any threshold result is research-only.

The immutable input is the Stage A coverage artifact from run [40077609244](https://github.com/angry0tiger-sudo/AliX/actions/runs/40077609244), coverage commit `8212da14e987fa10d7ab28f9aff190e9ec7d6fa6`, CSV SHA-256 `6b3f3ae006b8dc58d9f0f4092250cb484073984fb2249ed795bc0e12e823994c`.

## Data isolation and gap policy

- Segment A: 58,655 hourly candles from 2018-10-09 06:30 UTC to 2025-06-18 04:30 UTC.
- Segment B: 11,514 hourly candles from 2025-07-01 13:30 UTC to 2026-10-24 06:30 UTC.
- Gap: 320 missing hourly slots. The workflow never inserts synthetic candles, forward-fills, backfills, resamples, or interpolates.
- A and B are parsed, validated, featured, and evaluated independently. Every fold gets a freshly constructed model; no estimator, indicator state, target, or trading state transfers between A and B. A feature undefined on otherwise usable rows within a segment is excluded from that segment's model input and explicitly recorded; it is never imputed.

## Pre-registered fold protocol

- Target horizon: 12 hourly candles.
- OOS window: 90 days (2,160 hourly rows), expanding training; each fold purges 12 rows before test.
- Segment A development: minimum 3 years of valid training rows. Full 90-day OOS windows are used; a terminal partial window is explicitly labelled and included in development OOS evidence.
- Segment A holdout: the final 90 calendar days are reserved and evaluated separately, after development and B diagnostics. No parameter or model selection may use the holdout result.
- Segment B diagnostic: independent 180-day training warm-up followed by 90-day OOS windows. Because B is only about 1.31 years long, it is not comparable to the three-year A training gate and does not replace A's multi-year evidence.
- Model: pinned Phase 4 LightGBM regressor and existing past-only feature schema, fixed parameters, no hyperparameter search.
- Trade rule: at most one non-overlapping long-only research trade per 12-hour horizon if predicted return is strictly greater than modeled round-trip cost.
- Costs: 25 bps fee and 5 bps slippage per side. These are pre-registered assumptions, not verified live quotes.
- No real fill simulation, position-sizing/risk-engine integration, or execution calls occur. Results are not Freqtrade strategy backtest evidence.

## Economic criteria

Every result is evaluated independently for A development, A holdout, and B diagnostic:

1. At least 100 OOS trades.
2. Win rate strictly greater than 60%.
3. Positive net expectancy per trade after modeled costs.
4. Absolute maximum drawdown strictly less than 15%.

A missing/failed gate cannot be converted into acceptance by pooling A and B. A result can be marked `THRESHOLDS_MET_RESEARCH_ONLY`, `NO_GO_THRESHOLDS_NOT_MET`, or `INSUFFICIENT_SAMPLE`; none changes project authorization.

## Reproducibility and artifacts

The workflow verifies the upstream CSV and manifest hashes, run ID, commit, time ordering, OHLCV geometry, exact segment boundaries, and missing-slot count before evaluation. A failed verification stops the run. Results are uploaded to a GitHub Actions artifact containing the input artifact, per-fold report, OOS forecast CSV, taken-trade ledger, provenance, progress/failure state, and strict safety invariants.

## Immutable safety state

`Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, and `economic_acceptance=NO-GO`. Model promotion and strategy registry updates remain prohibited.
