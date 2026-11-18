# ADR-055 — Regime R G4.6 Volume-Derived Feature Ablation

- **Status:** ACCEPTED — COMPLETE_ABSTENTION_REPEATED; HGBQ-CQR ECONOMIC PATH CLOSED (REGIME P)
- **Date:** 2026-11-18
- **Branch:** `research/nobitex-hf-strategy`
- **Experiment:** G4.6 — one controlled volume-derived feature ablation
- **Preregistration:** `config/regime_r/hgbq_cqr_preregistration_ablation.json`
- **Preregistration SHA-256:** `83d9dffc0e3332eaddd5ee35bf846b13b049039eda6df087b9ccb734ce9f9de1`
- **Feature-contract SHA-256:** `6bcd91dcfdd3e48cfbe64adb714f432ae8dc610f136ed2731967c9aed5282250`
- **G3 frozen configuration SHA-256:** `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- **Baseline:** G4.5 Run #11 / artifact 14416704508
- **Execution:** [G4.6 Run #1](https://github.com/angry0tiger-sudo/AliX/actions/runs/44307631442) — SUCCESS
- **Artifact:** [Artifact 14428742548](https://github.com/angry0tiger-sudo/AliX/actions/runs/44307631442/artifacts/14428742548)
- **Artifact name:** `regime-r-g4-g6-volume-ablation-prehack-0ca5402b208524d20d3a482227a63214551f945c`
- **Artifact ZIP SHA-256:** `e63a05c2b05886e42616d69b3e46ab4dfc59ecdcd2e3646690644ea39d80d6b6`
- **Evaluated Research commit:** `0ca5402b208524d20d3a482227a63214551f945c`
- **Standard CI for evaluated code:** [Run #2501](https://github.com/angry0tiger-sudo/AliX/actions/runs/44305925205) — SUCCESS

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


## 8. Final execution record and provenance

The single pre-registered G4.6 experiment completed successfully on GitHub Actions:

| Field | Observed |
|---|---|
| Workflow Run ID | `44307631442` |
| Run number | `#1` |
| Event | `workflow_dispatch` |
| Branch | `research/nobitex-hf-strategy` |
| Head SHA | `0ca5402b208524d20d3a482227a63214551f945c` |
| Conclusion | `success` |
| Artifact ID | `14428742548` |
| Artifact ZIP SHA-256 | `e63a05c2b05886e42616d69b3e46ab4dfc59ecdcd2e3646690644ea39d80d6b6` |

The run executed the locked registration and exact data/G3 hash verification. The run validator passed and the artifact was uploaded. The JSON has schema `alix.regime-r.g4.6-volume-ablation.v1`.

Observed accounting:

| Item | Value |
|---|---:|
| Registered logical slots | 30 |
| Unique horizon configurations | 3 |
| Duplicate slots | 27 |
| Fit calls | 90 |
| Diagnostic records | 90 |
| Unique horizon/fold slices | 9 |
| Forecast rows in the unique slices | 6,609 |
| Hypothetical trade rows | 0 |

The input dataset remained the committed PRE-HACK file, SHA-256 `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`, 31,801 rows. The artifact records 31,801 zero-volume rows and zero positive-volume rows. The frozen G3 commitment remained `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`; the G4.6 registration commitment remained `83d9dffc0e3332eaddd5ee35bf846b13b049039eda6df087b9ccb734ce9f9de1`.

## 9. G4.6 measured diagnostic results

All aggregate statistics below use only unique candidate slot 01 records, weighted by prediction count. The 27 duplicate slots are retained for accounting and reproduction only.

### 9.1 Aggregate

| Metric | G4.6 observed |
|---|---:|
| Forecast rows | 6,609 |
| Positive median forecasts | 3,085 / 6,609 |
| Median-positive rate | 46.678771% |
| Weighted mean forecast median | +10.2522 bps |
| Raw quantile interval width | 4.524927% (452.4927 bps) |
| Calibrated conformal interval width | 5.333342% (533.3342 bps) |
| Mean `Lower Bound - Cost`, 50 bps | -358.7484 bps |
| Mean `Lower Bound - Cost`, 90 bps | -398.7484 bps |
| Forecasts strictly above 50 bps cost | 0 / 6,609 (0%) |
| Forecasts strictly above 90 bps cost | 0 / 6,609 (0%) |

The 46.678771% figure is the percentage of model median forecasts greater than zero. It is **not** directional accuracy, realized hit rate, or win rate.

### 9.2 Horizon breakdown

| Horizon | Forecasts | Median positive | Mean median | Conformal width | LB − Cost at 50 bps | LB − Cost at 90 bps |
|---|---:|---:|---:|---:|---:|---:|
| 1h | 2,229 | 45.266936% | -0.8204 bps | 1.974032% (197.4032 bps) | -154.7903 bps | -194.7903 bps |
| 4h | 2,220 | 44.729730% | -0.0322 bps | 3.851546% (385.1546 bps) | -257.5849 bps | -297.5849 bps |
| 24h | 2,160 | 50.138889% | +32.2484 bps | 10.322920% (1,032.2920 bps) | -673.1954 bps | -713.1954 bps |

Every horizon/fold slice has zero forecasts with calibrated Lower Bound strictly above either cost threshold.

### 9.3 Direct comparison against G4.5

The G4.5 reference is Run #11 / artifact 14416704508. The raw G4.5 JSON and G4.6 JSON were compared record-by-record using the same `(trial_id, fold_id)` identity. **All 90 of 90 complete diagnostic statistics objects are exactly equal**, not merely close after rounding. This includes the median distribution, positive-median count, raw and calibrated interval widths, quantile summaries, conformal corrections, and lower-bound-to-cost distance summaries for every registered trial/fold record.

The unique-candidate weighted aggregate is consequently identical to G4.5:

| Metric | G4.5 | G4.6 | Change |
|---|---:|---:|---:|
| Median-positive rate | 46.678771% | 46.678771% | 0 |
| Mean conformal width | 533.3342 bps | 533.3342 bps | 0 |
| Mean LB − Cost, 50 bps | -358.7484 bps | -358.7484 bps | 0 |
| Mean LB − Cost, 90 bps | -398.7484 bps | -398.7484 bps | 0 |
| Forecasts above either cost | 0 | 0 | 0 |

The two volume-derived predictors had no measurable effect on the fitted forecast/calibration statistics on this fixed dataset. Removing them did **not** improve lower bounds and did **not** reduce conformal widths. This is consistent with their zero/constant values in the source data; it does not establish a general causal claim about volume features on other datasets or instruments.

## 10. Economic acceptance and final verdict

All three unique horizon candidates were tested under both frozen costs. The pooled offline hypothetical ledger results were:

| Horizon | Cost | NetPF | Trades | MaxDD | Qualifying LB folds | Candidate result |
|---|---:|---:|---:|---:|---:|---|
| 1h | 50 bps | N/A | 0 | 0.00% | 0 / 3 | FAIL |
| 1h | 90 bps | N/A | 0 | 0.00% | 0 / 3 | FAIL |
| 4h | 50 bps | N/A | 0 | 0.00% | 0 / 3 | FAIL |
| 4h | 90 bps | N/A | 0 | 0.00% | 0 / 3 | FAIL |
| 24h | 50 bps | N/A | 0 | 0.00% | 0 / 3 | FAIL |
| 24h | 90 bps | N/A | 0 | 0.00% | 0 / 3 | FAIL |

NetPF is N/A, not zero, because there were no completed trades. MaxDD of 0.00% is also not evidence of safety or profitability; no position was opened. The PF and trade-count gates failed, the lower-bound gate failed in every fold, and only the drawdown inequality was vacuously satisfied. No candidate passed all four gates at both costs.

The final machine verdict is:

`COMPLETE_ABSTENTION_REPEATED`

**Owner-directed decision: close the HGBQ-CQR economic path under Regime P.** The single authorized volume-feature ablation did not revive the path. No further HGBQ-CQR feature variants, parameter searches, cost relaxations, or selective reruns are authorized under this decision. The G4.6 evidence is exploratory PRE-HACK evidence only; there is no Holdout authorization.

## 11. Regime P closure, alternative paths, and safety

The HGBQ-CQR research code and artifacts remain in repository history for auditability, but this economic strategy is closed and must not be promoted or treated as a candidate Champion.

Two alternative research paths are recommended. Neither is authorized or executed by this ADR:

1. **Low-capacity quantile baseline:** evaluate a regularized linear quantile model (for example, scikit-learn `QuantileRegressor`) as an explicit low-complexity benchmark on the same target, data boundary, folds, and cost thresholds. It should be independently preregistered and held to the same economic gates. This tests whether any stable gross edge is present without the current HGBQ-CQR complexity.
2. **Target redesign:** separately preregister a volatility-scaled event/first-touch target, such as a leakage-safe triple-barrier formulation with explicit entry timing, barrier rules, costs, purging, and embargo. This tests whether fixed-horizon gross-return regression is the wrong decision target. It is not a parameter adjustment to G3 and must not reuse the current G4.6 outcomes as if they were confirmatory evidence.

The G4.6 run safety record is:

- `real_data_scope = PRE_HACK_ONLY`
- `holdout_bytes_read = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `hypothetical_trades_simulated = true` (offline deterministic costed ledger only)
- `exchange_network_access = false`
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`

No Holdout or post-hack bytes were read. No private exchange API was called. The zero-trade CSV is an empty offline ledger result, not a trading simulation that placed orders.

Architecture First. Evidence Always. Fail Closed.
