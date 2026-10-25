# AliX RL Trading Agent: Technical Roadmap

## Architecture decision

The new research branch adds a Gymnasium environment and a Stable-Baselines3-compatible recurrent PPO policy without modifying the DEAP path. Recurrent PPO is used because the policy is intentionally restricted to chart-like OHLCV/time observations; portfolio accounting remains simulator-internal rather than becoming an extra market feature.

## Environment contract

- **Market input:** verified contiguous BTCUSDT Spot 1-minute candles: timestamp, open, high, low, close, volume.
- **Observation:** normalized OHLC prices, normalized volume, and cyclical UTC time-of-day/day-of-week. Scales are fitted on training rows only. No account balance, inventory, unrealized PnL, indicator, future label, or execution state is passed to the policy.
- **Action:** continuous two-value vector. The first component maps to Sell, Hold, or Buy; the second chooses a size between 0% and 100%. Because RecurrentPPO's continuous policy does not mask distribution logits, the environment applies a deterministic hard mask before simulated execution and reports the legal mask/reason in `info`. SELL is blocked without inventory; BUY is blocked when cash is exhausted/full allocation and whenever the target would reduce rather than increase the position. Short selling is disabled.
- **Execution:** an action selected after bar close fills at the next bar open. Each fill pays configured per-side fees and slippage. Any residual position is liquidated with costs at an evaluation-slice boundary.
- **Reward:** net log equity growth minus drawdown penalties, a 0.001 penalty on each step with no actual position-quantity change after three grace steps, a separate 0.001 penalty when the pre-action legal mask blocks a requested BUY/SELL, and a 0.0001 bonus only when a complete round trip closes with cumulative realized PnL positive after fees and slippage. Reward components and invalid-action penalty counts are reported separately. The bonus is not paid on partial sales or unrealized profit. This shaping can encourage churn, so activity alone is not success; net return, costs, drawdown, fills, and win/loss counts remain the evidence. Equity already includes fees and slippage; intrabar high/low marks contribute to drawdown. There is no fixed timeframe, take-profit, stop-loss, or maximum holding-period rule.
- **Training cost curriculum:** training starts with zero fee/slippage. Only 100 policy-closed simulated round trips, counted from executed position accounting rather than action requests or forced terminal liquidation, can start a linear 100,000-environment-step ramp to configured costs. Cost updates are applied in deterministic 1,000-step increments. Validation, final holdout, and Buy & Hold baselines always use configured nonzero costs. The report records stage timesteps, stage trade counts, start/end steps, and whether the ramp was actually completed; an incomplete ramp forces economic NO-GO.
- **Safety:** the model can emit a simulated decision only. It cannot reach exchange clients or grant order permission.

## Data and split protocol

1. Reuse the immutable public two-year minute-bar artifact when available; otherwise fetch up to 24 calendar months from the public endpoint.
2. Verify manifest schema, source, symbol, interval, row count, SHA-256, full coverage and exact one-minute continuity. Any missing/invalid row fails closed.
3. Exclude the previously reported June 1, 2026 04:59 UTC to October 25, 2026 holdout window from this experiment. The new analysis window ends at the last available candle before that boundary.
4. Split that earlier window chronologically, without shuffling: 70% training, 10% validation, 20% final holdout. The new holdout therefore does not reuse the previously reported final-holdout dates.
5. Fit observation scaling on training rows only. The CSV/hash still refer to the full verified public dataset; the report records the excluded tail rows and exact evaluation dates.
6. Train only on the training split, in chronological order, for at least 1,000,000 environment steps. Because training is shorter than the requested budget, it replays the same chronological training split from its beginning. Validation and final holdout rows are never used in training.
7. Evaluate validation once for diagnostics without early stopping, checkpoint selection, or parameter tuning.
8. Evaluate the new final holdout exactly once after training. Compare against cash and buy-and-hold under the same simulated costs. This window is disjoint from the previous holdout but is drawn from historical data used by earlier research runs; it is fresh for this model's fit, not a never-before-downloaded market sample.

## GitHub Actions execution budget

- Python 3.12, CPU-only PyTorch, pinned Gymnasium, Stable-Baselines3 and SB3-Contrib versions.
- A 55-minute job ceiling, a single on-policy environment, bounded LSTM size, rollout/batch sizes and pip cache keyed to the research requirements. The explicit 1,000,000-step minimum replays the training split as needed within the existing ceiling; the CI/time budget is not increased and no gate is relaxed.
- Source data is reused by artifact ID when available; a public-data download is the fallback if the retained artifact has expired.
- Upload the source CSV/manifest, dependency freeze, model checkpoint and machine-readable report with the source commit and safety invariants.
- Standard CI continues to compile, lint, format-check, type-check, security-scan, audit dependencies, enforce architecture/complexity contracts and run the existing unit-coverage gates. No quality gate is removed or relaxed.

## Acceptance criteria

- The Gymnasium checker passes; invalid data, actions, timestamps, costs and accounting states fail closed.
- Tests prove next-bar execution, position-size semantics, fee/slippage accounting, terminal liquidation, intrabar drawdown penalties, and absence of portfolio fields from observations.
- The historical workflow verifies model reload, at least 1,000,000 training steps, full validation coverage, action/trade statistics and exactly one full final-holdout evaluation.
- Tests prove SELL masking without inventory, BUY masking at full allocation, rejection of BUY actions that would reduce the position, and inactivity penalties based on consecutive unchanged positions rather than action labels.
- Reward tests prove that the profitable-trade bonus is issued only after a completed round trip has positive realized PnL after costs.
- The new holdout end timestamp is strictly earlier than the previously reported holdout start, with explicit excluded-row accounting.
- The report provides final-holdout net profit/return, max drawdown, fees, slippage, fill count, masked requests and separate reward penalties, plus comparison to cash/buy-and-hold.
- The entropy coefficient is explicitly recorded (initial iteration: `ent_coef=0.01`); the report proves at least 1,000,000 actual training steps and records whether 100 policy-closed trades and the entire cost ramp were reached. Incomplete curriculum is reported honestly and forces economic NO-GO.
- \`Champion=null\`, \`order_permission=false\`, and \`trading_authorization=NO-GO\` remain invariant, even if the report is profitable.

## Phased plan

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| 0. Branch and contracts | Isolated branch, owner directive, dependency pins, typed simulator | DEAP unchanged; all safety fields explicit |
| 1. Environment | Gymnasium contract, causal execution, cost ledger, drawdown reward | Unit tests and Gymnasium checker pass |
| 2. Training | Recurrent PPO, fixed seeds, CPU budget, saved checkpoint | Model trains across the complete train split and reloads |
| 3. Evaluation | One validation pass, one untouched holdout pass, cash/buy-and-hold baselines | Hashes/splits/metrics are machine-verified |
| 4. Robustness research | Separate runs for multiple seeds, fee/slippage shocks and market regimes | Consistent evidence across preregistered tests |
| 5. Shadow-only review | Non-executing prediction/report integration | Owner review; no trading authorization by default |

A successful Actions run proves that the experiment executed and its claims are reproducible. It does not prove the policy is profitable or grant live-trading authority.
