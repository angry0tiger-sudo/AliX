# ADR-053 — Regime R HGBQ-CQR G4/G5 Pre-hack Evaluation

- Decision: **NO-GO / INSUFFICIENT_EVIDENCE**
- Date: 2026-11-17
- Research branch: `research/nobitex-hf-strategy`
- Evaluation code SHA: `b5d69c9728f7a3ffd3989689fc948fcc96db419c`
- Evaluation workflow: [Run #10](https://github.com/angry0tiger-sudo/AliX/actions/runs/44072026184)
- Economic artifact: [Artifact 14322682950](https://github.com/angry0tiger-sudo/AliX/actions/runs/44072026184/artifacts/14322682950)
- Dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Frozen G3 configuration SHA-256: `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`

## 1. Decision

The owner-authorized exploratory G4/G5 evaluation on the fixed pre-hack BTC/USDT 1h dataset completed. All 30 registered trial slots ran, representing three unique candidate configurations and 27 duplicate slots, with 90 fit calls recorded.

No evaluated lower confidence bound exceeded its applicable round-trip cost threshold. Consequently, no hypothetical entry was opened and no trade was recorded. The decision is **NO-GO / INSUFFICIENT_EVIDENCE**. This result does not establish that every possible HGBQ-CQR design is economically ineffective; it establishes that this preregistered evaluation produced no qualifying signal and did not satisfy the acceptance gates.

## 2. Data boundary and frozen protocol

The only real-data input was `data/nobitex/prehack/nobitex-btcusdt-1h.csv`, with 31,801 rows spanning 2021-10-31T20:30:00Z through 2025-06-17T20:30:00Z. The workflow checked its byte-level SHA-256 before parsing. The G3 preregistration file was independently hash-checked and was not modified.

The run retained the preregistered HGB settings, quantiles 0.05/0.50/0.95, MAPIE 1.6.0, 90% confidence level, 1h/4h/24h horizons, three walk-forward folds, and 50/90 bps round-trip cost scenarios. The deterministic row-wise quantile rearrangement remained enabled. No Holdout/Post-hack data was read.

The pre-hack dataset is historically exposed and is exploratory rather than confirmatory. All 31,801 volume values are zero. Therefore, volume-change and volume-Z-score features provide no discriminatory signal in this dataset; this limitation does not invalidate the separately computed price-derived features.

## 3. Observed economic outcomes

The strict entry condition was:

`Lower Bound > Round-trip Cost`

Equality is an abstention. The actual artifact records the following outcomes for every trial slot, both cost scenarios, and all three folds:

| Horizon | Forecasts per fold | Signals above cost per fold | Abstentions per fold | Trades per fold | Fold stability |
|---|---:|---:|---:|---:|---:|
| 1h | 743 | 0 | 743 | 0 | 0/3 |
| 4h | 740 | 0 | 740 | 0 | 0/3 |
| 24h | 720 | 0 | 720 | 0 | 0/3 |

The same fold-level counts were recorded at both 50 bps and 90 bps. Across the full registered matrix:

- Registered trial slots: 30
- Unique candidate configurations: 3
- Duplicate slots: 27
- Fit calls: 90
- Completed hypothetical trades: 0
- Signals above cost: 0
- Full-gate candidates passing both costs: 0
- Skipped due to an occupied position: 0
- Unexecutable qualifying signals: 0

Net Profit Factor is **N/A**, not zero, because no trades exist from which to calculate it. Max Drawdown is 0.00% because no position was opened; this is not evidence of profitable or successful trading. DSR is not computable and is not imputed.

## 4. DSR and acceptance gates

`GLOBAL_N=UNKNOWN`. The historical trial multiplicity cohort and dependence structure have not been fully reconciled. A numeric Deflated Sharpe Ratio cannot therefore be calculated defensibly. No assumed trial count or fabricated DSR was substituted.

The positive DSR gate fails closed. In addition, zero trades and zero fold stability prevent the trade-count and fold-stability requirements from passing. No candidate passed all required gates at both costs.

## 5. Post-mortem: what the evidence does and does not establish

The directly supported explanation for zero trades is mechanical and exact: every test-fold lower bound was at or below the relevant cost threshold, so every forecast abstained under the frozen strict gate. The artifact does not contain the per-row lower/median/upper quantile values, conformal interval widths, or lower-bound-minus-cost distributions. It therefore cannot establish whether the non-qualifying lower bounds were mostly negative, slightly positive but below cost, or associated with unusually wide uncertainty intervals. Those explanations remain **UNKNOWN**, not interchangeable diagnoses.

Zero volume is a confirmed material feature limitation: volume-derived features were constant or neutral throughout the dataset. However, the evaluation also used price-derived return, volatility, range, ATR, and time features. The evidence does not isolate volume as the cause of the absence of qualifying lower bounds. It would be incorrect to attribute the zero-trade outcome solely to zero volume or to claim that high uncertainty was proven.

## 6. Safety invariants

- `real_data_scope = PRE_HACK_ONLY`
- `holdout_bytes_read = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `main = FROZEN @ 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`

This ADR records research evidence only. It does not authorize Holdout access, another real-data fit, paper orders, live orders, or candidate promotion.

Architecture First. Evidence Always. Fail Closed.
