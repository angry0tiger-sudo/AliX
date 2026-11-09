# Phase AI-6 (Funding / External Data) — Owner-Aborted

- Decision: ABORTED by Owner Directive dated 2026-11-09.
- Scope: external Funding Rate ingestion and Funding-only 36-cell sweep.
- Replacement track: risk-only PPO research and Nobitex mock execution preparation.
- Branch for replacement work: research/ml-core-ai6-rl-nobitex.
- Base commit: ef14d9ecbf06a0a984977b869fcdc75c60c38115.

## Decision and interpretation

The Owner has ended the Funding Rate workstream because the Bybit/CoinAPI/runner investigation became an infrastructure and operational distraction from the final Spot execution target, Nobitex. Prior attempts encountered provider/geographic access restrictions and a self-hosted European x64 runner dependency. CoinAPI's exact historical metric coverage, plan entitlement and cost were not established.

This is a project-priority decision. It is **not** a statistical or economic finding about Funding Rates. No complete, valid 36-cell Funding experiment was completed, so the project must not report “NO STATISTICAL EDGE FOUND WITH FUNDING RATES.” The result is ABORTED / NOT EVALUATED.

## Evidence and outstanding operational cleanup

- The AI-6 Funding workflow run #57 was created on commit 00f0a0fc10278de005f29ddfa6ec2fe090ffc5ba, not on the later branch HEAD ef14d9ecbf06a0a984977b869fcdc75c60c38115.
- At the last verified GitHub inspection, run #57 remained queued and its Job had no assigned runner (runner_id=0, empty runner name). No Funding dataset or economic report was produced by that queued job.
- The available GitHub integration in this session exposed read-only workflow-run inspection but no workflow-run cancellation operation. The owner should cancel run #57 manually in GitHub Actions if it is still queued. This record is not a cancellation confirmation.
- Standard CI #2102 succeeded on the base SHA ef14d9ecbf06a0a984977b869fcdc75c60c38115.
- V4-AP workflow run #26 (42493418590) completed its 36-cell Primary-only research, but its artifact contains the report/trade ledger/summary, not the underlying five-year OHLCV CSV and manifests. It is therefore not a valid input artifact for PPO training.

## New approved research scope

1. PPO may allocate long Spot exposure only after an independent Primary entry signal is present.
2. The policy action is a target exposure fraction in {0%, 25%, 50%, 75%, 100%}. It cannot create, reverse or suppress the Primary signal. In the absence of a Primary BUY signal, the policy may maintain or reduce exposure but may not increase it.
3. The first implementation uses causal ATR(14), drawdown, current exposure and the independent Donchian-20 Primary signal. PPO is evaluated as a risk allocator, not as a signal generator.
4. The only allowed entry timeframes remain 15m, 30m, 1h and 4h. Daily and weekly execution/selection timeframes remain prohibited under MRD v0.7.
5. The backtest cost contract remains 10 bps fee plus 5 bps slippage per side, equal to 30 bps round trip.
6. Training consumes only an already-produced, immutable five-year BTCUSDT/ETHUSDT Spot OHLCV artifact with matching manifests and hashes. This phase does not download market data from any exchange. If the artifact is absent or invalid, PPO training must fail closed.
7. The final 15% chronological holdout is reserved and must not be used for training, tuning, model selection or reward calibration.

## Nobitex mock execution scope

- Implement only in-memory mock order-book access using the documented GET path shape /v3/orderbook/{SYMBOL}.
- Enforce a liquidity gate against cumulative executable quote depth within 0.5% of mid-price.
- Create passive maker-intent child slices using a conservative, explicit participation cap.
- Every slice is marked post_only_required=true, but exchange_enforced_post_only_confirmed=false. A limit price alone is not proof of maker-only execution.
- There is no HTTP client, API-key loading, order submission, cancel, account, wallet or private endpoint in this phase.
- Every interaction is synthetic/mock-only. No real Nobitex API calls are permitted.

## Key handling

The Owner-provided public key must be stored by the Owner as a GitHub Actions repository secret named NOBITEX_PUBLIC_KEY for a future, separately authorized phase. It is not needed by the mock implementation and is not read by the current code or workflow. It must not appear in source, tests, artifacts, commit messages or logs. The current phase does not attempt to use the key.

## Safety state

Champion = null  
order_permission = false  
trading_authorization = NO-GO  
private_api_calls = 0  
real_order_attempts = 0  
orders_or_execution_api_used = false

No model training or mock execution plan can grant live trading permission. Nobitex production integration, exchange-enforced post-only support, real order lifecycle, reconciliation and a separate Release Authorization remain future blockers.
