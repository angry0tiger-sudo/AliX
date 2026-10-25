# AliX RL Trading Agent: Technical Roadmap

## Architecture decision

The RL research branch uses a Gymnasium market environment and Stable-Baselines3-Contrib Recurrent PPO. It remains isolated from the DEAP path. Portfolio accounting stays inside the simulator and is never exposed as a policy observation.

## Environment contract

- **Market input:** verified contiguous BTCUSDT Spot 1-minute candles: timestamp, open, high, low, close, volume.
- **Observation:** normalized OHLC prices, normalized volume, and cyclical UTC time-of-day/day-of-week. Scales are fitted on training rows only. No account balance, inventory, unrealized PnL, technical indicator, future label, or execution state is passed to the policy.
- **Action:** discrete `MultiDiscrete([3, 4])`. Direction index maps to Hold / Buy / Sell; size index maps to 25%, 50%, 75%, or 100%. No zero-size bucket exists. BUY spends that fraction of currently available cash including costs; SELL liquidates that fraction of inventory; HOLD ignores the size index. If BUY is legal and spendable cash exists, the purchase must create a positive position change. The execution-boundary legal mask still blocks SELL without inventory and BUY when available cash is exhausted; distribution logits are not claimed to be masked.
- **Execution:** an action selected after a completed candle fills at the next candle's open. Every fill pays configured per-side fees and slippage. Any residual position is liquidated with costs at an evaluation-slice boundary.
- **Reward:** net log-equity growth minus worsening/level drawdown penalties; a 0.001 inactivity penalty after three grace steps; a 0.01 penalty for genuinely masked BUY/SELL requests; a 0.0001 penalty for HOLD while an open position's expected liquidation value after exit fee/slippage is below its cost basis; and a 0.0005 opportunity-cost penalty only for flat HOLD above the causal 20-candle moving average. A policy-selected SELL that fully closes a profitable trade earns an additional reward equal to `0.5 * max(0, closed_trade_net_profit / initial_equity)` plus the existing 0.0001 profitable-close bonus. Partial sells and terminal forced liquidation do not earn the proportional realized-profit reward. The moving average is internal reward shaping only and never part of the observation. Reward shaping is diagnostic, not economic evidence. RecurrentPPO entropy coefficient for this experiment is 0.02.
- **Training cost curriculum:** training starts with zero fee/slippage. Only 100 executed policy-closed round trips can start a linear 100,000-step ramp to configured costs, applied in 1,000-step increments. Validation, final holdout and Buy & Hold baselines always use configured nonzero costs. An incomplete curriculum forces economic NO-GO.
- **Safety:** the model can emit simulated decisions only. It cannot reach exchange clients or grant order permission.

## Data and split protocol

1. Reuse the immutable verified two-year public minute-bar artifact when available; otherwise fetch the public endpoint.
2. Verify manifest schema, source, symbol, interval, row count, SHA-256, coverage and exact one-minute continuity. Invalid or missing data fails closed.
3. For this architecture, training and validation use only rows strictly before 2026-06-01 04:59 UTC. Split that pre-cutoff period chronologically, without shuffling: 70% training, 10% validation, and leave the previous candidate's held-out interval between validation and the cutoff unused by fitting and scoring.
4. Use the latest tail from 2026-06-01 04:59 UTC onward as the current model's final holdout. It must not enter training, scaler fit, or validation, and it is evaluated exactly once.
5. The latest tail was inspected in earlier research. It is fresh for this model's fit, but it is not a market sample unseen by the wider research history. The report must state this explicitly.
6. Fit scaling constants on training rows only. Train for at least 1,000,000 steps on CPU, then evaluate Validation once for diagnostics and the latest-tail Holdout once. No validation checkpoint selection or tuning.
7. Report exact split timestamps, excluded gap rows, selected discrete size buckets, requested/masked actions, position changes, completed trades, opportunity-cost penalties, fees, slippage, net return and drawdown.

## GitHub Actions execution budget

- Python 3.12, CPU-only PyTorch, pinned Gymnasium/Stable-Baselines3/SB3-Contrib, single on-policy environment, bounded LSTM, pinned dependency resolver.
- CI must pass on the exact training commit before training starts. The training workflow checks for that exact-commit success and fails closed on timeout or failed checks.
- The 55-minute research-workflow ceiling remains unchanged. No dependency audit, security, architecture, complexity or coverage gate is relaxed.
- Upload the data and manifest, dependency freeze, model checkpoint and machine-readable report with commit and SHA-256 provenance.

## Acceptance criteria

- Gymnasium checker passes and RecurrentPPO accepts the MultiDiscrete action space.
- Tests prove each BUY bucket creates a positive purchase amount from available cash, no size bucket is zero, sells liquidate the selected fraction, and invalid action indices fail closed.
- Tests prove opportunity-cost penalty is causal, applied only to flat HOLD above the moving average, and does not alter policy observations or penalize HOLD while invested.
- At least 1,000,000 steps are trained, model save/reload passes, all selected sizes and fills are reported, and final-tail evaluation executes exactly once.
- Holdout is disjoint from this model's training and validation; prior research exposure is disclosed rather than hidden.
- Compare Holdout with cash and Buy & Hold after fees/slippage. Economic acceptance is evidence only and never auto-promotes.
- `Champion = null`, `order_permission = false`, `trading_authorization = NO-GO`; zero private API calls and zero real orders.

## Phased plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0. Branch and contracts | Owner directive, pinned dependencies and architecture contract | DEAP unchanged; safety invariants explicit |
| 1. Environment | Discrete actions, next-bar execution, costs and reward accounting | Gymnasium checker and unit tests pass |
| 2. Training | Recurrent PPO with discrete direction/size outputs | At least 1,000,000 steps and successful reload |
| 3. Evaluation | Validation and one latest-tail Holdout pass | Hash, split, metrics and safety assertions pass |
| 4. Robustness research | Multi-seed, cost sensitivity and market regimes | Preregistered comparisons, never auto-promotion |
| 5. Shadow-only review | Non-executing report integration | Owner review; trading authorization remains NO-GO |

A successful Actions run proves that the experiment executed and its claims are reproducible. It does not prove the policy is profitable or grant live-trading authority.
