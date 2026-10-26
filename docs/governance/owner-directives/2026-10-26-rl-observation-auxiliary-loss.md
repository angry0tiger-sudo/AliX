# Owner Directive: State-Aware Observation and Auxiliary Guidance Loss

Date: 2026-10-26

## Decision

The archived Run #47 policy diagnostic identified 30,208 repeated BUY requests while the account was fully allocated. Portfolio state was absent from the policy observation. Retraining is prohibited until this architecture and its tests pass CI.

## Required observation contract

The default environment observation must append:

- `current_position_fraction`: marked BTC position value divided by total marked account equity, clamped to [0, 1].
- `unrealized_pnl_pct`: decimal return on remaining position cost basis, including entry costs; `0.05` means +5%; zero when no position exists.

The current portfolio fields are state inputs, not execution authority. Cash, total account value, private exchange information and technical indicators remain unavailable to the policy. A nine-feature legacy observation mode is permitted only for reproducing the immutable historical Run #47 checkpoint.

## Required guidance loss

During training, store a per-transition direction target from the causal slope of the 20-completed-candle simple moving average. BUY is a target only when the slope is rising and BUY is legal for the pre-action portfolio state; SELL is a target only when the slope is falling and SELL is legal; otherwise target HOLD.

Add cross-entropy between the policy direction distribution and this target to the PPO objective with `auxiliary_loss_weight=0.1`, linearly annealed to zero over 500,000 environment steps. The target must remain aligned with each recurrent rollout-buffer sample, including sequence padding and minibatch permutation. Fail closed on missing, malformed or out-of-range targets.

The auxiliary loss and reward guidance are training-only. Validation and final Holdout must have reward-guidance scale zero and must never optimize a supervised guidance loss.

## Acceptance and safety

- CI must pass on the exact training commit before a training-trigger commit is made.
- The research Actions run must train at least 1,000,000 steps, validate the observation contract and loss schedule, and evaluate Validation once and the final Holdout once.
- The quality gates are not to be relaxed to make the work pass.
- `Champion = null`, `order_permission = false`, `trading_authorization = NO-GO`.
- Zero private API calls and zero real order attempts. This owner directive does not authorize shadow/live routing.

Owner Directive supersedes conflicting architecture prose where the latter says portfolio state is deliberately absent from all observations.
