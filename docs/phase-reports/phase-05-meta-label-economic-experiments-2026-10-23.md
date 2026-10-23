# Meta-Label Economic Experiments — 2026-10-23

## Status

**Research completed; economic acceptance remains NO-GO.** The work only retrained candidate classifiers inside chronological, purged walk-forward folds. It did not replace production model artifacts or modify the strategy registry.

- Research workflow: [Run #40004995169](https://github.com/angry0tiger-sudo/AliX/actions/runs/40004995169)
- Full CI: [Run #40004995134](https://github.com/angry0tiger-sudo/AliX/actions/runs/40004995134)
- Research branch head tested by both runs: `df106554104a4cb1ab1e634c60335e552ca1bcda`
- Evidence artifact: [alix-meta-label-economic-experiments-40004995169](https://github.com/angry0tiger-sudo/AliX/actions/runs/40004995169/artifacts/12524165970)
- Artifact digest: `sha256:1b9bd126ff3b13b393e18b4407041a44a478a1fb6bb0cf36592bc87bf5bb83df`

## Input and provenance

- Provider: Nobitex public UDF history API; public market data only.
- Dataset: 8,760 chronological hourly BTC/USDT candles.
- Dataset SHA-256: `1a9576510ceeac99a46e25186e1559ddc48fdc4d4c9c40b13e969b3b21e1508b`
- No credentials or account data; no order endpoint calls and zero real-order attempts.
- Shared data-quality checks passed; gaps are rejected rather than interpolated.

## Why the original Meta holdout produced no score >= 0.70

The original training run `39976295482` had 1,083 Meta training events and 112 holdout events. At the fixed research cutoff of 0.70, zero holdout events passed. Holdout Brier score was 0.14007, log loss 0.44532, baseline positive-label probability 0.22715, and mean net return across all proposals was -0.655% per event.

This establishes that the fixed filter produced no accepted holdout scores and the proposals were economically weak after the modeled costs. It does **not**, by itself, prove overfitting. The original metrics do not provide enough evidence to distinguish overfitting from a weak primary signal, poor score calibration, limited features, label/cost mismatch, or the deliberately narrow long-only event-generation rule.

The original feature set has nine OHLCV-derived features. Meta examples are generated only when the out-of-fold primary direction model predicts a long event, with a 12-bar triple-barrier label. The positive meta-label requires net return above zero after modeled fees and slippage (25 bps fee plus 5 bps slippage per side). These choices reduce the event set and make the target more difficult.

## Walk-forward experiment

The research harness evaluated 36 configurations on the same fixed one-year public dataset:

- Feature groups: current baseline; ATR + volume; ATR + volume + trend/momentum.
- Primary classifier: current unscaled LogisticRegression plus standardized LogisticRegression at C = 0.1, 1, 10.
- Meta LogisticRegression regularization C = 0.1, 1, 10.
- Expanding chronological primary OOF folds, purged mature Meta labels, and non-overlapping selected trades.
- Meta score threshold remained fixed at 0.70. The threshold was not optimized to manufacture signals.
- The adaptive policy could choose a candidate for a fold only using prior OOS folds with at least 30 selected trades, win rate strictly above 60%, and positive net expectancy. Otherwise it falls back to the baseline.

### Results

| Evaluation | OOS selected trades | Win rate | Net expectancy per trade | Maximum drawdown | Interpretation |
|---|---:|---:|---:|---:|---|
| Fixed baseline | 10 | 50.00% | -0.2490% | Not sufficient sample for acceptance | Negative; insufficient sample |
| Adaptive walk-forward policy | 10 | 50.00% | -0.2490% | Not sufficient sample for acceptance | Fell back to baseline; no prior candidate qualified |
| Best exploratory fixed candidate: `atr_volume|scaled_c1|meta_c=0.1` | 19 | 57.89% | +0.1156% | 9.13% | Promising diagnostic only; below acceptance sample and win-rate criteria |

Baseline weighted Brier score was 0.18756 with Brier skill 0.06843 versus a constant-probability baseline. The best exploratory fixed candidate had Brier skill 0.08348. These are modest improvements in probability quality, not proof of a stable trading edge.

**The best exploratory row is not the deployed policy.** Candidate summaries are ranked after examining the full set of fold results, so selecting the best of 36 configurations carries multiple-comparison/selection bias. The adaptive walk-forward policy avoids current-fold lookahead and did not switch candidates because no prior candidate met all selection conditions. Its 10 OOS trades are well below the required 100 trades, and its expectancy is negative.

## Decision

- Objective of win rate > 60% and positive expectancy: **not met by the adaptive policy**.
- Economic gate: `INSUFFICIENT_SAMPLE`.
- Production HMM / Meta artifacts: unchanged.
- Model approval records: remain `OWNER_APPROVED_TECHNICAL_ONLY`; no economic approval inferred.
- Strategy registry: remains empty; no strategy-level evidence is eligible for admission.
- `Champion = null`
- `economic_acceptance = NO-GO`
- `order_permission = false`
- `trading_authorization = NO-GO`

The next research iteration should prioritize independent, larger OOS trade samples, fold-level probability calibration and stability diagnostics, and strategy-level costed evidence. Do not lower the 0.70 research cutoff or change registry/promotion state merely to increase the number of emitted signals.
