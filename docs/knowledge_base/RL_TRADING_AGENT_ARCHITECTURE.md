# AliX RL Trading Agent: Technical Roadmap

## Architecture decision

The RL research branch uses a Gymnasium market environment and Stable-Baselines3-Contrib Recurrent PPO with a training-only auxiliary guidance-classification loss. It remains isolated from the DEAP path. Portfolio accounting stays inside the simulator; only two state features required for policy awareness are exposed.

## Environment contract

- **Market input:** verified contiguous BTCUSDT Spot 1-minute candles: timestamp, open, high, low, close, volume.
- **Observation:** normalized OHLC prices, normalized volume, cyclical UTC time-of-day/day-of-week, `current_position_fraction` and `unrealized_pnl_pct`. Scales for market inputs are fitted on training rows only. `current_position_fraction` is marked BTC notional divided by total marked equity, bounded to [0, 1]. `unrealized_pnl_pct` is the decimal return on remaining position cost basis including entry costs (0.05 means +5%); it is zero when flat. No cash/equity amount, technical indicator, future label, or execution state is passed to the policy.
- **Action:** discrete `MultiDiscrete([3, 4])`. Direction index maps to Hold / Buy / Sell; size index maps to 25%, 50%, 75%, or 100%. No zero-size bucket exists. BUY spends that fraction of currently available cash including costs; SELL liquidates that fraction of inventory; HOLD ignores the size index. If BUY is legal and spendable cash exists, the purchase must create a positive position change. The execution-boundary legal mask still blocks SELL without inventory and BUY when available cash is exhausted; distribution logits are not claimed to be masked.
- **Execution:** an action selected after a completed candle fills at the next candle's open. Every fill pays configured per-side fees and slippage. Any residual position is liquidated with costs at an evaluation-slice boundary.
- **Reward and auxiliary loss:** net log-equity growth minus worsening/level drawdown penalties; a 0.001 inactivity penalty after three grace steps; a reduced 0.001 penalty for genuinely masked BUY/SELL requests; symmetric 0.0001 shaping for holding an open position in profit (after estimated exit cost) and a 0.0001 penalty for holding it below net cost basis; and a 0.0005 opportunity-cost penalty only for flat HOLD above the causal 20-candle moving average. A policy-selected SELL that fully closes a profitable trade receives the realized-profit reward defined in this document's experiment history. The existing training-only causal reward guidance adds at most 0.0002 for a legal executed BUY while the 20-candle SMA slope is positive or SELL while it is negative, then linearly anneals to zero in 300,000 steps with 1,000-step updates. In addition, a CrossEntropy auxiliary loss teaches direction targets from that causal SMA slope and the pre-action legal position mask. Its coefficient starts at 0.1 and linearly decreases to zero over 500,000 training steps. The target is BUY on a rising slope only while buying is legal, SELL on a falling slope only while selling is legal, and HOLD when the trend is neutral or the suggested direction is illegal. This loss is training-only; Validation and final Holdout have both the guidance reward and auxiliary loss disabled. No technical indicators or guidance labels are passed to the observation. RecurrentPPO entropy coefficient is 0.05. Reward shaping and auxiliary imitation are diagnostic, not economic evidence.
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


## Policy-level action masking (Owner Directive, 2026-10-26)

The policy direction distribution is masked before sampling, deterministic argmax, PPO log-probability evaluation, and the training-only auxiliary cross-entropy calculation. The mask reads the explicit `current_position_fraction` feature from the same observation used by the recurrent policy; missing or misaligned observation context fails closed.

- At a position fraction of at least `1 - 1e-6`, BUY direction logits are set to negative infinity.
- At a position fraction of at most `1e-6`, SELL direction logits are set to negative infinity.
- HOLD remains legal in every state. The environment keeps its independent execution-boundary mask as defense in depth.
- If an invalid BUY reaches the environment despite policy masking, the step reward receives a `-1.0` fallback penalty; other masked invalid actions retain the default `-0.001` penalty.
- Tests assert the masked direction has exactly zero categorical probability in both flat and fully allocated states. Validation and final Holdout use the same policy mask but no Guidance reward or auxiliary optimization.
- The report schema v8 makes policy masking status, thresholds, invalid-BUY fallback penalty counts, and immutable authorization locks explicit.

A passing test proves the mask is enforced, not that the learned policy is profitable or that it will choose SELL at an economically useful time.
