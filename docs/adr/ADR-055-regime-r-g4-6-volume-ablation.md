# ADR-055 — Regime R G4.6 Volume-Derived Feature Ablation

- **Status:** PREREGISTERED_LOCKED — NOT YET EXECUTED
- **Date:** 2026-11-18
- **Branch:** `research/nobitex-hf-strategy`
- **Experiment:** G4.6 — one controlled volume-derived feature ablation
- **Preregistration:** `config/regime_r/hgbq_cqr_preregistration_ablation.json`
- **Preregistration SHA-256:** `83d9dffc0e3332eaddd5ee35bf846b13b049039eda6df087b9ccb734ce9f9de1`
- **Feature-contract SHA-256:** `6bcd91dcfdd3e48cfbe64adb714f432ae8dc610f136ed2731967c9aed5282250`
- **G3 frozen configuration SHA-256:** `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- **Baseline:** G4.5 Run #11 / artifact 14416704508

## 1. Hypothesis

Removing the volume-derived features, which are constant or zero in the fixed pre-hack dataset, will improve calibrated lower-bound returns and reduce conformal interval width relative to the G4.5 baseline.

This is one binary ablation. No additional feature profiles, parameter changes, target changes, fold changes, cost changes, or selective reruns are authorized by this registration.

## 2. Single controlled feature change

The baseline model receives 13 predictors. G4.6 removes exactly these two predictors:

- `log_volume_change_1`
- `volume_zscore_24`

The remaining 11 predictors retain their original formulas and order:

`log_return_1`, `log_return_4`, `log_return_24`, `realized_volatility_6`, `realized_volatility_24`, `range_fraction_1`, `atr_fraction_14`, `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`.

The raw Volume column remains in the hashed source dataset and schema validation. It is not provided as a predictor to the estimator. No other code path or feature family is changed.

## 3. Frozen G3 and evaluation protocol

The G3 preregistration remains byte-identical. HGB estimator, hyperparameters, random state, quantiles (0.05/0.50/0.95), MAPIE CQR 1.6.0, 90% calibration, target definition, 1h/4h/24h horizons, three rolling purged walk-forward folds, six-month training, one-month calibration, one-month embargo, one-month test, 24-hour purge, and 50/90 bps costs are unchanged.

The experiment uses the exact pre-hack Nobitex BTC/USDT 1h dataset:

- Path: `data/nobitex/prehack/nobitex-btcusdt-1h.csv`
- SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Rows: 31,801
- Scope: historically exposed exploratory PRE-HACK data; not a confirmatory holdout.

The existing 30 logical trial slots remain for lineage (three unique horizon configurations and 27 exact duplicate slots). Expected accounting is 90 fits and 90 diagnostic records. Duplicate slots are not independent candidates and cannot inflate the pass count.

## 4. Evaluation and acceptance criteria

Each unique horizon candidate (slot 01 for 1h, 4h, and 24h) is assessed independently at each cost across its three folds. For each cost scenario, all four gates must pass:

1. Pooled Net Profit Factor `>= 1.10`.
2. Pooled completed hypothetical Trades `>= 100`.
3. Pooled Max Drawdown `<= 15%`.
4. At least one forecast in each of the three folds must have calibrated Lower Bound strictly greater than that cost (3/3 qualifying folds).

A horizon candidate passes only when all four gates pass at both 50 bps and 90 bps. G4.6 passes only if at least one of the three unique horizon candidates passes this complete rule. All other outcomes are NO-GO.

If no lower bound clears cost in any horizon, fold, or cost scenario, the result is `COMPLETE_ABSTENTION_REPEATED` and the current HGBQ-CQR economic path is closed (Regime P), as directed by the Owner. If some qualifying signals exist but no candidate clears all gates, the result is `NO_GO_PARTIAL_SIGNALING`; no promotion or Holdout access is allowed.

The Owner's four explicit G4.6 gates govern this experiment. `GLOBAL_N` remains UNKNOWN and DSR remains NOT_COMPUTABLE; no numeric DSR is inferred or fabricated. G4.6 is exploratory evidence and not multiplicity-adjusted confirmation.

## 5. Diagnostics and comparison

The artifact must report, by unique horizon/fold and cost:

- median-positive forecast rate (not directional accuracy);
- mean calibrated conformal interval width;
- signed mean `Lower Bound - Cost` distance;
- count of forecasts strictly above cost;
- pooled and fold-level NetPF, trade count, and MaxDD;
- trial/fit/diagnostic accounting and duplicate lineage.

Aggregate diagnostic values use only the three unique horizon configurations and are weighted by prediction count. Results are compared against G4.5 baseline: median-positive 46.6788%, conformal width 5.3333%, and mean lower-bound-minus-cost distances -358.75 bps (50 bps) / -398.75 bps (90 bps).

## 6. Safety and non-goals

- PRE-HACK ONLY; no Holdout or post-hack path/bytes may be accessed.
- No private API or exchange network access.
- No real order and no Paper order.
- Hypothetical trade simulation is permitted only in the deterministic offline costed ledger to calculate the explicitly required economics.
- `Champion = null`; `order_permission = false`; `trading_authorization = NO-GO`.
- No model-family change, G3 parameter change, feature change beyond the two listed columns, target change, cost change, or fold change.
- This registration does not authorize deployment or Holdout access.

## 7. Decision rule

A successful G4.6 result means only that a unique horizon candidate passed the preregistered exploratory pre-hack gates and may be proposed for a separately authorized confirmatory evaluation. It does not authorize reading Holdout, candidate promotion, paper orders, or live trading.

A repeated complete abstention closes this HGBQ-CQR economic path under the Owner's explicit directive.

Architecture First. Evidence Always. Fail Closed.
