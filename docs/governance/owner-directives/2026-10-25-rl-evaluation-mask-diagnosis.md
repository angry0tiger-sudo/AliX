# Owner Directive: Diagnose zero-size BUY requests without retraining

**Date:** 2026-10-25  
**Branch:** `research/rl-trading-agent`  
**Authority:** Explicit user instruction in the Run #26 follow-up.

## Decision and evidence

1. Do not retrain the 1,000,000-step Recurrent PPO model for this diagnosis. Reuse the exact Run #26 checkpoint and verify its SHA-256 against the original report.
2. The prior report records `initial_equity = 10000` and `invalid_action_penalty_count = 0` for both validation and final holdout. Therefore the evidence does not support the hypothesis that the pre-action legal mask rejected BUY due to zero cash. The issue to diagnose is the execution path after the legal mask, including a zero or non-increasing target-size component.
3. Treat a BUY request with size zero or a target that would not increase the position as a non-executable no-op, not as an illegal action-mask block. Only a BUY rejected by the pre-action legal mask counts as `masked_buy_action_count`. Preserve fees, slippage, allocation limits, and fail-closed behavior.
4. Evaluation telemetry must report deterministic policy action signal and position-size summaries, zero-size BUY requests, no-op reasons, actual fills, PnL, return, drawdown, fees, and slippage. Do not modify actions to force a trade or silently clamp the learned action size.
5. Run the existing model on validation and the same Run #26 final holdout once in this evaluation-only workflow. This re-evaluation is authorized by the owner and uses the exact existing checkpoint; no training or model selection is performed. Because this repeats inspection of the same holdout, do not describe it as untouched for future model-selection work.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- Zero private API calls and zero real order attempts.
- No live exchange connection, automatic promotion, or weakened CI/quality gates.
