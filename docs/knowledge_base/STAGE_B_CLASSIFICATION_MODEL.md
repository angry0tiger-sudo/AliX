# Stage B Binary Classification with a Volatility Gate

## Target definition

This is a research-only fork of the Stage B signal path. The positive class is the economically actionable long-side event:

forward close-to-close return over 12 hours > +0.006

The round-trip cost remains 0.60%: 25 bps per side in fees plus 5 bps per side in slippage. A large negative move is not a positive/buy label. Predicting absolute move magnitude would accidentally treat a sharp fall as a reason to buy.

A trade is taken only when all three predeclared conditions hold: classifier probability meets a candidate's fixed probability threshold, the timestamp is a non-overlapping 12-hour entry opportunity, and current closed-candle 24-hour volatility is at least the 75th percentile of volatility observed in that fold's training data.

## Experiment grid

The 12 predeclared candidates combine:

- Feature groups: Stage B baseline features available on A, and those features plus RSI(14), distance from SMA(50/200), 24-hour momentum, 24-hour volatility, and range position.
- Class weighting: unweighted binary LightGBM and fold-specific positive-class weighting derived from negative/positive label counts.
- Probability cutoffs: 0.35, 0.50, and 0.65.

The probability cutoffs are candidate configurations, not relaxed economic acceptance gates. If a candidate is selected for an OOS fold, selection uses only earlier OOS folds and requires at least 30 prior selected trades, win rate above 60%, positive net expectancy, and absolute drawdown below 15%. Otherwise the policy falls back to the unweighted baseline at p=0.50. Aggregate candidate rankings are descriptive only.

## Temporal and data-integrity rules

The workflow reads the immutable Stage A coverage artifact and removes the reserved final 90-day A holdout before feature engineering, target creation, training, or selection. It uses expanding chronological 90-day OOS folds, a 12-hour label purge, fresh classifiers per fold, and an identical contiguous timestamp grid across candidate feature groups.

The volatility cutoff is recalculated on each fold from training data only. Neither test-fold volatility distribution nor future target data influence the cutoff. No missing values are fabricated: incomplete warm-ups or internal gaps fail closed; no forward-fill, backfill, interpolation, resampling, or imputation is used. Segment A has no positive-volume rows, so no volume signal is invented.

## Economic acceptance and safety

Every candidate and the adaptive policy are scored after modeled transaction costs. Acceptance requires at least 100 OOS trades, win rate strictly above 60%, positive net expectancy per trade, and absolute maximum drawdown below 15%. A strong-looking small sample does not pass.

This experiment remains research-only. A model cannot be promoted or authorize orders even if the research metrics appear attractive.

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- economic_acceptance = NO-GO
- no production model or strategy registry updates
- zero private API calls and zero order endpoint calls
