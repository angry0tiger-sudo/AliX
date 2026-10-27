# Owner Directive: RL diagnostics, bounded reward shaping, and causal trailing stop

Date: 2026-10-27
Scope: offline historical RL research only; no production or live-trading authorization.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- Private API calls and real order attempts remain zero.
- All evaluation remains offline and fail-closed; CI quality gates may not be weakened.
- The 15-candle minimum holding lock remains in both policy masks and the environment. A stop exit may bypass this lock as a risk exit.

## Diagnostics contract

Report `average_holding_period_candles`, `average_win_percent`, `average_loss_percent`, `profit_factor`, `early_exit_count`, and `flags.HOLD_COLLAPSE` for each evaluation split. Percent metrics are net of allocated entry/exit fees and slippage and measured against total entry costs for the trade. Average win uses positive closed-trade returns; average loss retains a negative sign. Empty averages and Profit Factor without gross losses are `null`. `early_exit_count` is the number of closed trades whose net return is below 1%, including loss exits as a diagnostic only.

## Reward changes

- Transaction-cost reward multiplier is 2.0; configured fee and slippage remain unchanged from the first training step.
- Base realized-profit reward remains 0.05 times closed net profit divided by initial equity. It is multiplied by 2.0 at net trade return >=3%, and 3.0 at >=5%, capped at 0.01 reward per closed trade.
- The early-exit penalty is 0.0001 and applies only to policy-selected full exits with non-negative net return below 1%. Losing exits and all trailing-stop exits are exempt so risk exits are never discouraged.

## Trailing-stop contract

The long-only spot research environment uses a 3% activation threshold and 2% trailing distance from the highest high observed in a completed candle. The stop is updated only after that candle is completed and can trigger only on a subsequent candle. A gap through an already-active stop fills at next open and suppresses the stale policy action for that candle without charging an invalid-action penalty; an intrabar low crossing the prior stop fills at the stop price. Fee/slippage accounting remains active. Trailing stop does not inspect future candles and does not authorize live orders.

## Acceptance

The report schema is `alix.rl-trading-agent-report.v15`. Unit tests must cover HOLD collapse, empty trade metrics, small positive exits, losing exits, stop activation/trigger, and the stop-exit penalty exemption. Exact-SHA CI must be green before the `[train-rl]` workflow gate permits training. Research outcomes never promote a Champion or authorize execution.
