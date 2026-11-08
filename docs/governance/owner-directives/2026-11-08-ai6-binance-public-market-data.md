# Owner Directive: AI-6 Funding-Only 36-Cell Economic Evaluation

- directive_id: ALIX-OD-2026-11-08-AI6-FUNDING-ONLY
- owner: angry0tiger-sudo
- timestamp: 2026-11-08
- approval_status: Authorized for bounded internal Funding-only research implementation and economic diagnosis.
- branch: research/ml-core-v4aq-ai6
- parent_base: V4-AP / AI-6 implementation branch
- supersedes: the Funding+Open-Interest scope in the initial AI-6 directive, for this phase only.

## 1. Owner decision

Open Interest is removed entirely from the active AI-6 implementation. Binance's public Open Interest Statistics endpoint exposes only a limited recent history and cannot support the required 1825-day training window. Do not fetch, store, feature-engineer, join, or model Open Interest in this phase.

Funding Rate History remains the sole external source. Use only the unauthenticated public HTTPS GET endpoint:

- Host: fapi.binance.com
- Path: /fapi/v1/fundingRate
- Symbols: BTCUSDT and ETHUSDT
- Window: the exact 1825-day window proven by the corresponding Spot OHLCV manifests
- Pagination: provider timestamps, bounded page size/retries, and no synthetic records

No alternate provider, archive, endpoint, credential, private API, account endpoint, proxy/VPN/tunnel change, or order endpoint is authorized by this directive.

## 2. Funding data semantics and features

1. Preserve provider funding settlement timestamp, local retrieval timestamp, conservative availability timestamp, record content hash, CSV hash, and manifest provenance.
2. Funding is considered available no earlier than fundingTime + 5 minutes. This is a conservative research proxy, not a provider-certified historical publication timestamp.
3. Build exactly two causal external features:
   - funding_rate_zscore
   - funding_rate_ma
4. Use a 30-settlement rolling window with at least 20 observations; reset warm-up after a settlement gap exceeding 48 hours.
5. Use backward as-of joins only when available_at <= decision_time; maximum observation age is 48 hours.
6. Require exact 1825-day source boundaries, first/last settlement within one day of the requested boundary, maximum observed settlement gap no greater than 48 hours, and estimated settlement coverage fraction >= 90%.
7. Missing, stale, malformed, duplicated, out-of-order, or incomplete Funding data must fail closed. Never interpolate, backfill, forward-fill beyond the maximum age, or synthesize a Funding observation.
8. The active connector must expose only the Funding endpoint. No OI class, OI endpoint, OI fetch option, oi_change feature, or OI join may remain in the active data pipeline.

## 3. Economic sweep

Use the existing V4-AP 36-cell grid and economic execution implementation, without altering its primary signal definitions, exit geometries, gap treatment, or transaction-cost assumptions:

- Assets: BTCUSDT and ETHUSDT.
- Timeframes: 15m, 1h, 4h.
- Primary strategies: the existing RSI/EMA mean-reversion rule and Donchian20 close breakout.
- Exit geometries: TIGHT, BALANCED, TRAILING.
- Total: 2 x 3 x 2 x 3 = 36 combinations.
- Source: public Binance Spot 1m OHLCV, exact common 1825-day window for both assets.
- Costs: 10 bps fee + 5 bps slippage per side = exactly 30 bps round trip.
- Primary remains the sole source of entry events. Funding features may filter Primary BUY events only.

Fit one Funding-only LightGBM Meta filter for each asset/timeframe/primary-strategy group (12 groups). Reuse each group's frozen, out-of-sample filtered event stream to evaluate the three exit geometries. Do not fit a separate Meta model per exit.

Use the existing V4-X purged OOF and fit/calibration split. Training events are confined to the first 70% of the chronological dataset. Select the acceptance threshold only from purged in-training calibration probabilities; never use forward-validation labels to select it. Evaluate the 70%-85% forward interval. Preserve the final 15% as an untouched holdout, excluded from fitting, threshold selection, and the reported economic decision.

The model target is the fixed 24-hour forward close outcome net of the 30 bps execution costs. Report forward-validation ROC-AUC and whether it exceeds the diagnostic reference 0.51. ROC-AUC is diagnostic, not a standalone promotion rule.

## 4. Preregistered candidate acceptance

Report, for all 36 combinations:

- ROC-AUC and comparison to 0.51.
- Net Profit Factor.
- Win Rate.
- DSR Z-score.
- Trade count, signals, costs, drawdown, and candidate decision.

A cell is accepted only when every condition passes:

- Trades >= 50.
- Win Rate >= 40%.
- Net PF >= 1.10.
- DSR Z > 0.

Use the full 36-cell cohort as the nominal DSR trial count. Clearly disclose that statistical independence among candidates has not been verified. Net PF >= 1.10 alone is not a promotion decision.

If no cell reaches Net PF >= 1.10, close the External Data phase with a negative Funding-only diagnostic and proceed to the separately isolated RL Research track. RL may be investigated only for risk management and position sizing; it must never create, replace, or modify Primary entry signals. If any cell reaches the PF threshold but fails the remaining gates, do not label it a winner.

## 5. Workflow failure correction

AI-6 Workflow #15 failed at the Funding fetch command with ModuleNotFoundError: No module named 'external_data'. The OHLCV fetch had completed; this was a Python module invocation/path failure, not insufficient Funding/OI history or a pagination failure.

All repository scripts in the workflow must be invoked as modules from the repository root (for example, python -m scripts.fetch_binance_external_data). The workflow must check out the exact commit SHA and require successful Standard CI for that same SHA before any data fetch.

## 6. Safety, engineering and governance invariants

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- No live, dry-run, or paper trading session may be started.
- No interaction with exchange accounts, balances, positions, or orders.
- No changes to Freqtrade order lifecycle, Risk Gate, Kill Switch, or live configuration.
- Standard CI, security, dependency, architecture, complexity, type, and coverage gates remain unchanged.
- Coverage must remain >= 90%.
- All code, tests, public-data fetches, and experiments execute in GitHub Actions. No local execution.
- main remains at its owner-pinned SHA; no merge without explicit owner approval.

## 7. Completion evidence

A successful workflow must publish a Step Summary with all 36 rows, exact run SHA, dataset window, Funding coverage, ROC-AUC, Net PF, Win Rate, DSR Z, trade count, acceptance outcome, and closed safety state.

If Funding coverage fails its preflight, publish a successful fail-closed blocked report with 36 explicit NOT-RUN rows, no model fit, and no economic results. Do not mislabel a data coverage block as evidence of no statistical edge.
