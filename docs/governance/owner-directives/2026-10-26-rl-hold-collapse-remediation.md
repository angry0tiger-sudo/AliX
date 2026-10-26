# Owner Directive: Recurrent PPO HOLD-Collapse Remediation

**Date:** 2026-10-26  
**Branch:** `research/rl-trading-agent`  
**Authority:** Explicit owner instruction for a new research-only training run. This supersedes the earlier low-volatility cash-HOLD bonus and entropy coefficient for this experiment; safety and evidence requirements remain unchanged.

## Required experiment

1. Remove the low-volatility cash-HOLD reward from the environment implementation. Cash HOLD must receive no bonus simply for waiting; the separate reward for holding an already-profitable open position remains distinct and must be reported as such.
2. Set the causal `opportunity_cost_penalty_when_flat_above_sma` to `0.01`. Apply it only when the action is HOLD, the portfolio is flat, and the completed candle close is above the causal 20-candle moving average. Do not expose the moving average or penalty signal in observations.
3. Use `ent_coef=0.5` for rollout updates starting before 50,000 training environment steps; use `ent_coef=0.1` afterward. Record the actual rollout-boundary transition in the report. High entropy is training exploration, not a guarantee of BUY/SELL actions and not randomization of deterministic evaluation.
4. Train Recurrent PPO for 1,000,000 steps on the verified public BTCUSDT Spot 1-minute dataset through GitHub Actions only, after exact-commit CI passes. No local training, exchange credentials, private APIs, execution endpoints, or real orders.
5. After training, evaluate every saved 100k checkpoint once on the exact Validation split with deterministic inference. Report discrete action counts and HOLD/BUY/SELL percentages, net profit, max drawdown, closed trades, fees, slippage, and checkpoint SHA-256. Do not score or recompute any checkpoint on the final Holdout, and do not select a checkpoint from Validation or Holdout.

## Acceptance and safety invariants

- A report must prove the cash-HOLD bonus was removed and the opportunity-cost penalty is `0.01`.
- Entropy configuration and the actual rollout-update transition must be explicit in the report.
- The ten Validation-only checkpoint rows must match the saved inventory and sum to the exact number of Validation steps. Report measured actions; do not make action activity a success criterion.
- `Champion = null`, `order_permission = false`, `trading_authorization = NO-GO`.
- `private_api_calls = 0`, `orders_or_execution_api_used = false`, `real_order_attempts = 0`. Production promotion remains prohibited.
- No CI, security, architecture, complexity, coverage, or dependency gate may be weakened.
