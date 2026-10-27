# AliX RL Trading Agent: Technical Roadmap

## Architecture decision

The RL research branch uses a Gymnasium market environment and Stable-Baselines3-Contrib Recurrent PPO with a training-only auxiliary guidance-classification loss. It remains isolated from the DEAP path. Portfolio accounting stays inside the simulator; only two state features required for policy awareness are exposed.

## Environment contract

- **Market input:** verified contiguous BTCUSDT Spot 1-minute candles: timestamp, open, high, low, close, volume.
- **Observation:** normalized OHLC prices, normalized volume, cyclical UTC time-of-day/day-of-week, `current_position_fraction`, `unrealized_pnl_pct`, and `position_holding_period_fraction`. Scales for market inputs are fitted on training rows only. The holding-period feature is the elapsed fraction of the 15-candle lock, clamped to [0, 1]. The policy sees only the minimum required state, not cash/equity amounts, technical indicators, future labels, or raw execution data.
- **Action:** discrete `Discrete(7)` with combined choices: `0 HOLD`, `1 BUY 25%`, `2 BUY 50%`, `3 BUY 100%`, `4 SELL 25%`, `5 SELL 50%`, `6 SELL 100%`. BUY spends that fraction of currently available cash including costs; SELL liquidates that fraction of inventory. The recurrent policy and environment both mask BUY and SELL while a position is within its 15-candle minimum holding period, BUY at full exposure, and SELL while flat. Additional buys are blocked during the lock, preventing the policy from repeatedly resetting the timer.
- **Execution:** an action selected after a completed candle fills at the next candle's open. Every fill pays configured per-side fees and slippage. Any residual position is liquidated with costs at an evaluation-slice boundary.
- **Reward and auxiliary loss:** net log-equity growth minus worsening/level drawdown penalties; a 0.001 inactivity penalty after three grace steps; no reward for merely holding cash; an explicit reward penalty equal to one times per-step fees plus slippage divided by prior equity; symmetric 0.0001 shaping for holding an open position in profit (after estimated exit cost) and a 0.0001 penalty for holding it below net cost basis; and a 0.01 opportunity-cost penalty only for flat HOLD above the causal 20-candle moving average. A policy-selected SELL that fully closes a profitable trade receives a realized-profit reward scaled to 0.05 of initial equity. The existing training-only causal reward guidance adds at most 0.0002 for a legal executed BUY while the 20-candle SMA slope is positive or SELL while it is negative, then linearly anneals to its 0.00002 floor in 300,000 steps with 1,000-step updates. In addition, a CrossEntropy auxiliary loss teaches direction targets from that causal SMA slope and the pre-action legal position mask. Its coefficient starts at 0.1 and linearly decreases to a 0.01 floor over 500,000 training steps. The target is BUY on a rising slope only while buying is legal, SELL on a falling slope only while selling is legal, and HOLD when the trend is neutral or the suggested direction is illegal. This loss is training-only; Validation and final Holdout have both the guidance reward and auxiliary loss disabled. No technical indicators or guidance labels are passed to the observation. For the 2026-10-27 HOLD-collapse remediation experiment, the drawdown-increment coefficient is 1.5, the transaction-cost reward multiplier is 1.0, and RecurrentPPO entropy coefficient is constant at 0.2 from the first training step. Reward shaping and auxiliary imitation are diagnostic, not economic evidence.
- **Transaction costs:** configured fees of 25 bps per side and slippage of 20 bps per side apply from the first training action through the end of training. No zero-cost stage or cost ramp exists. The report verifies all trained steps used the configured costs.
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
- The research workflow has a 180-minute execution ceiling to accommodate the recurrent 1,000,000-step training requirement. This is a runtime budget, not a quality-gate exception. No dependency audit, security, architecture, complexity or coverage gate is relaxed.
- Upload the data and manifest, dependency freeze, model checkpoint and machine-readable report with commit and SHA-256 provenance.

## Acceptance criteria

- Gymnasium checker passes, the default observation has 11 finite features including portfolio exposure and unrealized return, and the legacy nine-feature mode exists only to reproduce the archived Run #47 diagnostic.
- Tests prove each BUY bucket creates a positive purchase amount from available cash, no size bucket is zero, sells liquidate the selected fraction, and invalid action indices fail closed.
- Tests prove opportunity-cost penalty is causal, applied only to flat HOLD above the moving average, and does not alter policy observations or penalize HOLD while invested.
- At least 1,000,000 steps are trained, the auxiliary loss starts at 0.1 and reaches zero by step 500,000, model save/reload passes, all selected sizes and fills are reported, and final-tail evaluation executes exactly once.
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


## Policy-level action masking and combined action contract (Owner Directive, 2026-10-26; report schema v10)

The recurrent PPO policy masks the raw logits of a single seven-class categorical distribution before the distribution is constructed. It is not enough to create an unmasked distribution and mutate a component after construction.

- Action 0 is always HOLD. When `current_position_fraction >= 1 - 1e-9`, BUY actions 1, 2, and 3 are assigned logit `-1e8`. When `current_position_fraction <= 1e-9`, SELL actions 4, 5, and 6 are assigned logit `-1e8`. When a position has not completed 15 candles, both BUY and SELL logits are masked, with the environment enforcing the same rule at execution.
- Direction and size are a single categorical selection: BUY 25/50/100% of spendable cash, SELL 25/50/100% of inventory. The environment independently rejects illegal actions as defense in depth.
- The auxiliary cross-entropy uses all seven combined action classes, with causal guidance toward HOLD, BUY 25%, or SELL 25% when legally available. The early entropy schedule is a training-only exploration aid and does not randomize deterministic Validation/Holdout inference.
- Flat-cash HOLD receives no shaping bonus. A 0.01 opportunity-cost penalty applies when flat HOLD occurs above the causal 20-candle moving average. A separate explicit cost-shaping term subtracts five times this step's fees plus slippage divided by prior equity. Net log-equity return already includes actual transaction costs, so the additional penalty intentionally makes the learner more cost-sensitive.
- Missing or misaligned action-mask context fails closed. Tests assert all BUY probabilities are exactly zero at full exposure, all SELL probabilities exactly zero when flat, and the distribution is preserved through checkpoint save/reload.
- Report schema v10 records the combined action labels, masks, reward parameters, and per-action counts. It never promotes the model or authorizes live trading.

These constraints prove legal action enforcement for the tested states, not profitability or correct trade timing. buy_noop_action_count, net_return, and max_drawdown remain independent acceptance evidence. The regression test `test_masked_combined_actions_record_blocked_execution_not_side` verifies masked BUY/SELL requests remain `MASKED_*` in `executed_action`, leave inventory unchanged, and create no fill.
