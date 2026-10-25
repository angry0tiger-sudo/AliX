# Owner Directive: RL Trading Agent Research

**Date:** 2026-10-25  
**Branch:** \`research/rl-trading-agent\`  
**Status:** Approved direction; offline research only

## Owner direction

This directive establishes Reinforcement Learning (RL) as a new research path for AliX. Keep all existing DEAP implementation, reports, tests, and workflows intact. RL is an additional implementation path on this branch, not a destructive migration.

## Required implementation

1. Use a Gymnasium environment that steps through verified public BTCUSDT 1-minute Spot candles.
2. The policy receives OHLC price, volume, and UTC time features only. Do not expose account balance, open position, PnL, technical indicators, or labels as observations.
3. Provide Buy, Sell, and Hold decisions with continuous position size in the closed interval [0, 1].
4. Execute actions after a candle is complete, at the next candle's open. No same-bar fills, future data, synthesized missing candles, private market-data calls, or exchange execution APIs.
5. Apply fees and slippage to every simulated fill. Reward net equity growth and penalize worsening and persistent drawdown. Drawdown is measured with intrabar high/low marks.
6. Do not hardcode a candidate timeframe, holding-period limit, stop-loss, or take-profit rule into the policy or environment.
7. Train with Stable-Baselines3-compatible Recurrent PPO so policy memory can learn temporal context while market observations remain restricted.
8. Split the verified dataset chronologically into 70% training, 10% validation, and 20% final holdout. Fit scaling constants on training data only. Validation must not select checkpoints or tune parameters. Evaluate the final holdout once after training and report its full-period result.
9. Keep all training, evaluation, artifact creation, and result verification in GitHub Actions. Use fixed dependency versions, CPU-only execution, a bounded timeout, and artifact provenance/hash checks.
10. The economic report is evidence, not authorization. Never auto-promote the model.

## Immutable safety invariants

- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`
- Zero private API calls.
- Zero real order attempts.
- No live exchange connection, real orders, or execution API use.
- Existing CI quality gates remain enabled and unchanged in strength.
- The final holdout never participates in training, reward calculation during training, or model selection.

## Evidence required

The GitHub Actions artifact must include the source manifest and SHA-256, dependency inventory, trained recurrent PPO checkpoint, and a JSON report. The report must distinguish training, validation and final-holdout metrics; include fees, slippage, net profit, net return, drawdown and fills; record the model/research code commit; and prove the safety invariants.

A profitable single holdout is not proof of durable edge. It is a candidate for subsequent regime, fee/slippage sensitivity, repeated-seed, and forward shadow evaluation only. Trading authorization remains NO-GO.
