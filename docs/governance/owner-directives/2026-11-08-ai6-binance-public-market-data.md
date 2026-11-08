# Owner Directive: Phase AI-6 Public Funding and Open-Interest Research

- directive_id: ALIX-OD-2026-11-08-AI6-BINANCE-PUBLIC
- owner: angry0tiger-sudo
- timestamp: 2026-11-08
- approval_status: Authorized for bounded internal research implementation and public-data diagnostic only.
- branch: research/ml-core-v4aq-ai6
- base_sha: 798ef2d49bcbc715d35b16862a2cf83aa0865486
- parent_verdict: V4-AP, 36/36 combinations failed the preregistered statistical acceptance rule.

## Purpose

V4-AP concluded:

`NO STATISTICAL EDGE FOUND IN CURRENT FEATURE SET.`

Phase AI-6 investigates whether public derivatives market context improves the existing Primary + Meta-Model research pipeline. The initial source is Binance USD-M public Funding Rate and Open Interest History for BTCUSDT and ETHUSDT. The selected economic follow-up is BTCUSDT 4h Donchian20 with the existing BALANCED exit geometry. Funding/OI are Meta-Model features, not a replacement for the Primary signal.

## Narrow supersession of the previous Phase 6 directive

This directive supersedes only the prior blanket prohibition on real provider calls for the following exact internal-research scope:

- HTTPS GET to `fapi.binance.com/fapi/v1/fundingRate`.
- HTTPS GET to `fapi.binance.com/futures/data/openInterestHist`.
- Symbols: BTCUSDT and ETHUSDT.
- OI period: 4h.
- Unauthenticated public market-data access only.
- Retention: versioned research artifact and manifest within this GitHub Actions run, subject to the repository's artifact retention control.
- No redistribution or claim of commercial-use rights is authorized by this directive.

All other restrictions in the earlier Phase 6 directive remain in force. No other provider, unofficial endpoint, credential, private API, account endpoint, notification, proxy/VPN/tunnel modification, exchange connectivity for trading, or order endpoint is authorized. No live, dry-run, or paper trading session may be started.

## Required data semantics

1. Preserve exact provider timestamps, local retrieval timestamps, and conservative `available_at` research proxies separately.
2. Funding is treated as available no earlier than `fundingTime + 5 minutes`.
3. Four-hour OI aggregates are treated as available no earlier than `timestamp + 4 hours + 5 minutes`.
4. These availability offsets are conservative implementation assumptions, not provider-certified historical publication timestamps. Record the assumption in manifests and reports.
5. Funding z-score and OI change are computed causally. Historical feature rows may use only observations whose `available_at` is no later than the decision timestamp.
6. Missing, stale, malformed, duplicated, out-of-order, or unapproved observations fail closed or remain missing. No interpolation, backfill, synthetic row, or fabricated historical value is permitted.
7. Preserve raw provider observation identity through a content hash and the resulting CSV hash.

## Historical coverage limitation

Binance USD-M Open Interest History is limited to the latest 30 days. It cannot provide the requested 1825-day OI history through this endpoint. Therefore:

- Funding may be collected over the requested five-year interval, subject to returned coverage.
- OI must report its actual first/last timestamps, row count, observed span, missing-period count and coverage fraction.
- The 1825-day LightGBM fit and economic replay must be skipped unless the OI manifest proves complete requested-window coverage with every expected four-hour period observed.
- A 30-day OI slice must not be joined to five years of OHLCV as if it were a five-year feature history.
- No substitute archive or external provider may be activated without a separate owner directive and provider-rights review.

The correct result when coverage is insufficient is a successful diagnostic run with status `BLOCKED_INSUFFICIENT_EXTERNAL_HISTORY`, an explicit reason, and no model fit. It is not an economic failure or evidence that Funding/OI lack value.

## Modeling scope

- Primary remains the causal Donchian20 close breakout.
- `funding_rate_zscore` and `oi_change` are added to a fixed V4-AQ Meta-Model feature schema alongside causal Primary context.
- Reuse the existing LightGBM OOF, purging, calibration and Meta-Labeling contracts.
- Use the V4-AP BALANCED geometry and exactly 30 bps round-trip cost assumptions.
- The Meta-Model filters Primary BUY events only. It cannot issue independent entry signals, alter position sizing, approve risk, or change policy.
- The forward validation is diagnostic only. DSR is not inferred from a single selected cell. No Promotion/Champion is permitted from Net PF alone.

## Safety and governance

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- No change to Freqtrade order lifecycle, Risk Gate, Kill Switch, live configuration, or account secrets.
- All implementation, tests, fetches, and evaluations execute in GitHub Actions. No local execution.
- Standard CI, security, type, complexity, architecture and dependency checks remain unchanged.
- Coverage must remain >= 90%.
- Do not merge without explicit owner approval.

## Acceptance evidence

- Green Standard CI for the exact AI-6 SHA.
- Public-only endpoint manifest and exact data hashes for both assets.
- Actual Funding/OI coverage reported independently.
- If full five-year OI is unavailable, the model fit and economic replay must be explicitly NOT RUN.
- If complete data later becomes available through separately approved provenance, the report must include control vs Meta-filtered forward results, all costs, missing feature counts, untouched final holdout status, and closed safety invariants.
