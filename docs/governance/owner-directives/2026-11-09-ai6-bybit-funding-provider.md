# Owner Directive: AI-6 Funding Provider Switch to Bybit

- directive_id: ALIX-OD-2026-11-09-AI6-BYBIT-PROVIDER
- owner: angry0tiger-sudo
- timestamp: 2026-11-09
- approval_status: Authorized for bounded internal Funding-only research implementation and economic diagnosis.
- branch: research/ml-core-v4aq-ai6
- supersedes: Binance USD-M Funding provider selection in ALIX-OD-2026-11-08-AI6-FUNDING-ONLY only as to Funding source and adapter.

## Root cause and provider decision

The AI-6 workflow completed both 1825-day Spot baselines, then the public Binance USD-M Funding endpoint returned HTTP 451 from the GitHub-hosted runner. This is an infrastructure/provider access restriction, not an economic result.

Use only Bybit V5 public Linear Funding Rate History:
- Host: api.bybit.com
- Path: /v5/market/funding/history
- Unauthenticated HTTPS GET only; category=linear
- Symbols BTCUSDT and ETHUSDT
- Exact 1825-day Spot baseline window
- Maximum page size 200
- Backward pagination using endTime only. startTime alone is invalid; next endTime is oldest prior-page timestamp minus one millisecond.
- Response: result.list[].symbol, fundingRate, fundingRateTimestamp
- No account, wallet, order, position, private, or authenticated endpoints.

This supersedes the earlier Binance-only provider restriction. No proxies, VPNs, tunnels, credentials, endpoint rotation, or geo-restriction circumvention are authorized.

## Data and safety contracts

Map fundingRateTimestamp to canonical funding_time and fundingRate to funding_rate. Preserve the canonical CSV schema; mark_price is null because Bybit does not supply it. Preserve raw record hash, retrieval timestamp, and conservative available_at = fundingRateTimestamp + 5 minutes.

Reject nonzero retCode, malformed envelopes, wrong symbols, invalid timestamps/rates, timestamps after the requested window/cursor, malformed descending page order, duplicate/conflicting observations, stalled pagination, oversized responses, exhausted retries, and page-limit exhaustion. Validate provider descending order before sorting canonical observations ascending. Never synthesize or interpolate observations.

Manifest provenance identifies Bybit, api.bybit.com, /v5/market/funding/history, category=linear, page limit 200. Keep the exact 1825-day common Spot window and existing boundary, maximum-gap, and >=90% expected-settlement coverage gates. Insufficient history blocks model fitting and economic replay.

Unchanged: Funding features, 5-minute availability proxy, 48-hour as-of age, 30-settlement rolling window, Primary entries, V4-X purge/calibration, threshold policy, 36-cell grid, 30 bps cost, and acceptance gates (Trades >=50, Win Rate >=40%, Net PF >=1.10, DSR Z >0). Final 15% remains untouched. ROC-AUC remains diagnostic/N/A under the current AI-6 implementation.

Safety invariants: Champion=null; order_permission=false; trading_authorization=NO-GO; private_api_calls=0; real_order_attempts=0; credentials_used=false. No live, dry-run, paper, account, or order activity. Standard CI and >=90% coverage gates remain unchanged. All tests and research run in GitHub Actions only. main remains untouched at 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb.

## Completion evidence

Publish all 36 rows, exact SHA and data window, Bybit coverage/provenance, economic metrics, acceptance outcomes, and closed safety state. A coverage block is not a negative economic result. Only a completed 36-cell report establishes whether Funding produced an accepted edge.
