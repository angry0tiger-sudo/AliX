# AI-6 PPO Risk Allocation Research

## Scope and authority

This phase evaluates a PPO agent as a **risk allocator**, not as a directional trading model. The existing Nobitex AI Ensemble remains the sole source of candidate entry signals. PPO selects target Spot exposure only. It cannot create a BUY signal, reverse direction, short, use leverage, access an exchange, or submit an order.

The experiment starts from the immutable AI Ensemble commit `e6c86b08da98ad28e479b45827fb8de6cd3cc8dd`. It consumes the pre-hack Nobitex BTC/USDT 1-hour data artifact produced by Standard CI run `42871937160`. It does not call Nobitex, Binance, or any other market-data API during PPO training.

## Data contract

The data loader requires all of the following:

- Exactly 31,801 hourly candles from 2021-11-01 through the approved pre-hack endpoint.
- Nobitex Public UDF provenance and `NOBITEX_ONLY_NO_BACKFILL`.
- An exact hourly timestamp grid, no unresolved intervals, and no post-cutoff observations.
- A SHA-256 match between the CSV and its producer manifest.
- Producer commit `e6c86b08da98ad28e479b45827fb8de6cd3cc8dd`.
- No external source overrides, credentials, private API calls, or order attempts.

Any mismatch fails closed. The source artifact is downloaded from Standard CI #2193 rather than fetched again.

## Primary signal and temporal protocol

The signal teacher is the current LightGBM + XGBoost + causal HMM + Logistic Regression Ensemble without Chronos-2. It is fitted once using only eligible samples in the first 45% of the historical sequence. Every teacher training label must end at least 24 hourly bars before the 45% boundary.

The teacher is frozen before any PPO training, validation, or final holdout observation. Its probabilities are generated causally over the remaining data. A probability threshold of 0.50 defines the independent Primary BUY signal.

| Chronological interval | Purpose | PPO visibility |
|---|---|---|
| First 45% | Fit primary signal teacher | Teacher fitting only |
| 45%–75% | PPO training | Training observations and rewards |
| 75%–87.5% | OOS validation | Evaluation only |
| 87.5%–100% | Final OOS holdout | Final evaluation only |

The validation and final holdout are not used to fit the teacher, train PPO, tune hyperparameters, or choose checkpoints. The evaluation environments start flat in each segment so baseline and PPO begin from identical conditions.

## Environment contract

The Gymnasium observation contains ATR, conservative drawdown, current Spot exposure, causal 1/6/24-bar returns, frozen primary-signal probability, the Boolean signal gate, and the signal-episode-decision-pending flag.

The action space is binary:

- SKIP_SIGNAL = 0% target exposure.
- TAKE_SIGNAL = 100% target exposure.

PPO makes exactly one decision at the start of each contiguous Primary BUY signal episode. That choice is latched until the signal turns off; the first signal-off candle forces a full exit at the next hourly open. This makes each completed trade correspond to one accepted contiguous signal episode. Actions emitted during the middle of an already-decided signal episode are ignored and explicitly counted for audit.

The previous model was not literally outputting 0% throughout: it generated 2,620 validation and 2,876 holdout position adjustments, but only one complete flat-to-flat inventory lifecycle. Its trade counter correctly saw one continuously open position, not 2,000+ completed trades. The one-trade PF/Win Rate were therefore not statistically meaningful. This revision aligns decision semantics, position lifecycle, and trade-count semantics.

Decisions are made after an hourly candle closes and simulated fills occur at the next hourly open. OHLC intrabar highs/lows are used conservatively for drawdown tracking. Any remaining inventory is liquidated at segment end, including its exit fee.

## Economic and reward contract

The only execution cost used for the principal experiment is:

- Maker fee: 25 bps per side.
- Slippage: 0 bps per side.
- Round trip: exactly 50 bps.

The reward remains a running Sortino-style net log-return signal, but the drawdown-shaping coefficients are reduced fivefold so the 15% threshold remains visible without dominating every training decision: incremental penalty 0.5, linear excess penalty 2.0, and quadratic excess penalty 4.0. The hard 15% drawdown promotion gate is unchanged. No artificial inactivity penalty or participation bonus is applied; such a bonus could force the policy to take losing baseline signals and manufacture activity without edge.

The fixed baseline takes every contiguous primary-signal episode and exits when the signal turns off. It uses the same market bars, next-open fill convention, cost model, episode boundaries, and evaluation segments as PPO.

## Promotion gate

A research result is eligible for owner review only when **both validation and the untouched final holdout** satisfy all of these conditions:

1. At least **51 completed taken signal episodes** for both PPO and baseline (strictly greater than 50).
2. PPO Net Profit Factor is at least 1.10 and higher than the baseline.
3. PPO maximum drawdown is below 15% and lower than the baseline.
4. Final-holdout trade-return Z-Score is positive.

A pass is recorded as `GO_FOR_OWNER_REVIEW`, not trading authorization. A failure is `NO-GO`; do not promote or relabel a weak result.

## Safety invariants

Every run and report must retain:

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `ALIX_DRY_RUN = true`
- `ALIX_ALLOW_LIVE_TRADING = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `nobitex_api_calls = 0`

No Paper Trading or live trading is performed by this workflow. A passing research gate requires a separate owner decision before any Paper Trading preparation or activity.
