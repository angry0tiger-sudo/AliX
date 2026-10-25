# AliX RL Trading Agent: Technical Roadmap

## Architecture decision

The new research branch adds a Gymnasium environment and a Stable-Baselines3-compatible recurrent PPO policy without modifying the DEAP path. Recurrent PPO is used because the policy is intentionally restricted to chart-like OHLCV/time observations; portfolio accounting remains simulator-internal rather than becoming an extra market feature.

## Environment contract

- **Market input:** verified contiguous BTCUSDT Spot 1-minute candles: timestamp, open, high, low, close, volume.
- **Observation:** normalized OHLC prices, normalized volume, and cyclical UTC time-of-day/day-of-week. Scales are fitted on training rows only. No account balance, inventory, unrealized PnL, indicator, future label, or execution state is passed to the policy.
- **Action:** continuous two-value vector. The first component maps to Sell, Hold, or Buy; the second chooses a size between 0% and 100%. Buy targets a long allocation, Sell reduces the existing holding by the selected fraction, and Hold performs no rebalance. Short selling is disabled.
- **Execution:** an action selected after bar close fills at the next bar open. Each fill pays configured per-side fees and slippage. Any residual position is liquidated with costs at an evaluation-slice boundary.
- **Reward:** net log equity growth minus a strong penalty for worsening drawdown and a smaller persistent drawdown penalty. Equity already includes fees and slippage; intrabar high/low marks contribute to drawdown. There is no fixed timeframe, take-profit, stop-loss, or maximum holding-period rule.
- **Safety:** the model can emit a simulated decision only. It cannot reach exchange clients or grant order permission.

## Data and split protocol

1. Reuse the immutable public two-year minute-bar artifact when available; otherwise fetch up to 24 calendar months from the public endpoint.
2. Verify manifest schema, source, symbol, interval, row count, SHA-256, full coverage and exact one-minute continuity. Any missing/invalid row fails closed.
3. Split chronologically, without shuffling: 70% training, 10% validation, 20% final holdout.
4. Fit observation scaling on the training split only.
5. Train one chronological pass through the training split. The policy sees all train rows in sequence; there are no randomly synthesized bars or forced short training windows.
6. Evaluate validation once for diagnostic reporting, without early stopping, checkpoint choice, or parameter tuning.
7. Evaluate the final holdout exactly once after training. Compare against cash and buy-and-hold under the same simulated costs.

## GitHub Actions execution budget

- Python 3.12, CPU-only PyTorch, pinned Gymnasium, Stable-Baselines3 and SB3-Contrib versions.
- A 55-minute job ceiling, a single on-policy environment, bounded LSTM size, capped rollout/batch sizes and pip cache keyed to the research requirements.
- Source data is reused by artifact ID when available; a public-data download is the fallback if the retained artifact has expired.
- Upload the source CSV/manifest, dependency freeze, model checkpoint and machine-readable report with the source commit and safety invariants.
- Standard CI continues to compile, lint, format-check, type-check, security-scan, audit dependencies, enforce architecture/complexity contracts and run the existing unit-coverage gates. No quality gate is removed or relaxed.

## Acceptance criteria

- The Gymnasium checker passes; invalid data, actions, timestamps, costs and accounting states fail closed.
- Tests prove next-bar execution, position-size semantics, fee/slippage accounting, terminal liquidation, intrabar drawdown penalties, and absence of portfolio fields from observations.
- The historical workflow verifies model reload, full validation coverage and exactly one full final-holdout evaluation.
- The report provides final-holdout net profit/return, max drawdown, fees, slippage, fill count and comparison to cash/buy-and-hold.
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
