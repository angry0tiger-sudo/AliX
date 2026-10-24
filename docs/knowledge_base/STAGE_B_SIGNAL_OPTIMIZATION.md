# Stage B Signal Optimization Research

## Decision from the existing run

Reference Stage B run: [40099106018](https://github.com/angry0tiger-sudo/AliX/actions/runs/40099106018), commit \`fa5d290762d5658dab2420f2d52231abdc436598\`.

On A walk-forward development, the runner produced 30,022 OOS forecasts but the pre-registered non-overlapping 12-hour rule created only 2,502 entry opportunities. It accepted 12 trades, a 0.48% take rate. The round-trip cost hurdle is 0.006 (0.60%). At eligible entry times, forecast return quantiles were:

| Statistic | Forecast return |
|---|---:|
| Median | 0.0679% |
| 90th percentile | 0.1581% |
| 95th percentile | 0.2200% |
| 99th percentile | 0.4655% |
| Maximum | 2.3080% |

The 99th percentile is below the unchanged 0.60% hurdle. This is the direct reason the rule emits so few trades: the forecasts generally do not exceed modeled round-trip costs. The 12 accepted trades had a 41.67% win rate, -1.071% net expectancy per trade, and 20.45% absolute maximum drawdown. The baseline does not satisfy the economic gate.

This runner does **not** use a Triple-Barrier probability cutoff. Stage B predicts a 12-hour forward return and accepts a long trade only when the predicted return is strictly greater than the fixed round-trip cost. Therefore lowering a Meta-Label probability threshold would not fix this runner's sparse signals; the two signal paths must not be conflated.

## Research design

The research job reads the immutable Stage A artifact and uses **segment A development only**. It removes the reserved final 90 days before feature engineering and target creation; the holdout is not evaluated or used for feature or parameter selection. Segment B is not used to select candidates.

The fixed, 12-candidate grid combines:

- Horizons: 6, 12, and 24 hours. Each candidate purges its own forward-label horizon.
- Feature sets: the existing Stage B features that are actually available on A, and those features plus RSI(14), distance from SMA(50), distance from SMA(200), 24-hour momentum, 24-hour volatility, and 24-hour range position.
- LightGBM profiles: the existing Phase 4 parameters, and a predeclared shallower/regularized profile.

All candidates use expanding, chronological 90-day OOS folds and freshly fitted models. The 0.60% transaction-cost hurdle and all strict economic acceptance thresholds remain fixed. All feature warm-ups are handled by complete-row selection; no forward-fill, backfill, interpolation, resampling, or imputation is allowed. Every candidate has per-fold predictions, opportunities, cost-qualified trades, and costed metrics in the artifact.

The adaptive diagnostic policy can choose a candidate for a fold using only prior A-development OOS trades, and only if those prior observations include at least 30 selected trades, win rate above 60%, positive net expectancy, and drawdown below 15%. Otherwise it falls back to the existing Stage B baseline. Candidate-grid aggregate rankings are descriptive and do not select a model. An apparently strong result from a small exploratory grid is not confirmatory evidence.

## Feature notes

- RSI(14), SMA(50), SMA(200), momentum(24), volatility(24), and range-position(24) use current closed-candle/past data only.
- Natural leading NaNs are retained until complete feature rows exist; no value is fabricated to shorten warm-up.
- A contains no positive-volume rows, so the existing volume-change feature remains unavailable and is explicitly excluded. The experiment does not invent volume-derived signals from zero-volume source data.
- The new features are research-only and are not added to the production/global feature contract by this PR.

## HMM and Meta-Labeling scope

The current Stage B run uses a LightGBM forward-return regressor; it does not use HMM, Triple-Barrier labels, or a Meta-Label probability threshold. HMM/Meta-Labeling remain separate Phase 5 components. The prior Phase 5 one-year experiment evaluated 36 exploratory configurations and kept its probability cutoff fixed at 0.70; its adaptive policy did not qualify a candidate with the required prior OOS evidence. It would be methodologically incorrect to claim that its probability threshold caused Stage B's 0.48% take rate or silently splice its result into this A-segment comparison. Those models need their own A-only purged walk-forward experiment if selected as the next research path.

## Safety and acceptance

All outputs remain research-only. The baseline and every candidate are evaluated against the unchanged strict gate: at least 100 OOS trades, win rate strictly above 60%, positive net expectancy after cost, and absolute maximum drawdown below 15%. Passing CI or producing a candidate summary does not imply economic acceptance.

The workflow asserts and records:

- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`
- \`economic_acceptance = NO-GO\`
- no production model or strategy-registry update
- zero private API calls and zero order endpoint calls


### Degenerate but valid indicator windows

A perfectly flat rolling window has no missing candle data but creates a zero denominator for RSI and range-position. The research feature contract defines those two edge cases explicitly: RSI is 50 when average gain and average loss are both zero; range position is 0.5 when the rolling high, rolling low, and close are identical. These are deterministic indicator definitions, not OHLCV imputation. Leading warm-up rows and genuine data gaps remain undefined and are never filled.
