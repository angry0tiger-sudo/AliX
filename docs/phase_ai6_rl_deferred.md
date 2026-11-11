# Phase AI-6 RL — RESUMED BY OWNER DIRECTIVE

Decision: RESUMED by Owner Directive dated 2026-11-11.

The 2026-11-09 DEFERRED decision is superseded for the research workstream only. The owner has aborted the External Data Pivot, mandated OHLCV-only research, and explicitly authorized one final AI-6 PPO risk-filter experiment.

Active research branch: research/alternative-data-pivot

Scope:
- Nobitex BTC/USDT Spot OHLCV, 1-hour, fixed pre-hack dataset only.
- Frozen LightGBM + XGBoost + causal HMM + Logistic Regression primary teacher.
- PPO has exactly two actions: SKIP_SIGNAL (0) and TAKE_SIGNAL (1).
- PPO filters existing long-only baseline BUY signal episodes; it cannot create signals, short, leverage, size exposure continuously, or execute orders.
- At least 200,000 PPO training timesteps.
- Validation and final holdout remain chronological, untouched, and evaluation-only.
- Execution-cost assumption remains 25 bps maker fee per side (50 bps round trip), with 0 bps slippage.

The External Data Pivot is closed. Funding Rate, Open Interest and other external features are excluded. No interpolation, synthetic backfill or future data is permitted.

Safety invariants remain:
- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- private_api_calls = 0
- real_order_attempts = 0
- nobitex_api_calls_during_rl_run = 0

A passing research gate is only eligible for separate owner review. It does not authorize Paper Trading or live orders.
