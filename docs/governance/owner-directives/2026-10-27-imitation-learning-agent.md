# Imitation-learning research branch: teacher and behavioral cloning

Date: 2026-10-27

## Scope and safety

This branch explores a fresh, untrained LSTM trained by behavioral cloning on deterministic, causal OHLCV teacher labels. It does not load Recurrent PPO checkpoints. All work runs in GitHub Actions and historical simulation only.

Immutable safety invariants:
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- no private API, order, or execution endpoint; zero real-order attempts
- CI gates remain unchanged and must pass before training.

## Teacher limitations

The teacher uses completed 4-hour or daily trend bars, lagged swing/BOS levels, three-candle FVG heuristics, last opposing-candle order-block proxies, rolling VWAP, and rolling median-price proxy for volume profile. OHLCV cannot reveal true order-book liquidity, resting institutional orders, or a validated volume-at-price profile; these labels are heuristic research targets, not ground truth.

## Reward and evaluation

Behavioral cloning minimizes cross-entropy on teacher action labels. No hand-shaped RL reward is used. Evaluation uses chronological train/validation/final-holdout partitions; fit transforms and model training are training-only. Report action distribution, net return after configured fee/slippage, drawdown, win rate, profit factor, holding duration, and teacher/model label agreement. Promotion remains forbidden regardless of metrics.
