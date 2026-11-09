# RL Research — Risk Allocation Only

The risk allocator is separate from the legacy RL trading environment because the legacy environment has directional BUY/SELL actions. This implementation exposes only five target-exposure actions: 0%, 25%, 50%, 75% and 100%.

- Observation: ATR fraction, drawdown, current Spot exposure and an independent Primary BUY signal.
- Without a Primary BUY signal, PPO may maintain or reduce exposure but cannot increase it.
- Decisions are made after bar close and fills are simulated at the next bar open.
- Cost contract: 10 bps fee plus 5 bps slippage per side, or 30 bps round trip.
- Long Spot only; no shorting, leverage, exchange connection or order generation.

The training script reuses the existing causal ATR implementation and V4-AP Donchian-20 Primary signal. Only 15m, 30m, 1h and 4h are accepted.

Training requires immutable five-year BTCUSDT/ETHUSDT CSVs and matching SHA-256 manifests. The final 15% chronological holdout is not sent into feature generation, training or validation. The currently inspected V4-AP run #26 artifact contains a report, trade ledger and summary, but not the OHLCV CSVs. Do not use that artifact as training data. The manual workflow therefore requires an artifact run ID and name supplied by the Owner. This phase never downloads market data.

Nobitex mock code is in integrations/nobitex/liquidity.py. It uses an in-memory response transport and the documented GET path shape, but no HTTP client or order-write method. The liquidity gate uses contra-side quote depth within 0.5% of mid-price. Passive maker-intent slices use a default 10% participation cap on the same-side book. Every slice is marked post_only_required=true and exchange_enforced_post_only_confirmed=false.

Tests cover environment input contracts, signal gating, cost accounting, drawdown, terminal liquidation, order-book schema/freshness, the 0.5% liquidity gate, passive slicing and no-execution invariants. Standard CI retains all existing quality, complexity and 90% coverage gates.

PPO dependencies are isolated in requirements-rl-research.txt and are installed only by the manual research workflow. Core runtime requirements and requirements.lock remain unchanged.
