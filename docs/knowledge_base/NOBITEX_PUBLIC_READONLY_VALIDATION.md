# Nobitex Public Read-Only Validation

Status: implementation candidate / manual-only / no account credentials / no orders.

Workflow: \`.github/workflows/nobitex-public-readonly-validation.yml\`
Tracking PR: https://github.com/angry0tiger-sudo/AliX/pull/22

## How it runs

The workflow has only the \`workflow_dispatch\` trigger. It must be started manually from GitHub Actions; it is not attached to push or pull-request events. The standard CI only runs the synthetic contract tests. An actual artifact containing responses from public Nobitex endpoints is evidence only after a manual workflow run succeeds.

## Public requests

The script uses a fixed TLS endpoint host and an explicit path allowlist. It sends GET requests only:

- \`/market/stats?srcCurrency=btc&dstCurrency=usdt\`
- \`/v3/orderbook/BTCUSDT\`

The implementation has a timeout and maximum response size, validates JSON/status and positive finite prices/amounts, rejects crossed books and reports top-of-book spread plus sampled quote depth. The run writes a timestamped JSON artifact.

## Evidence boundary

The artifact identifies itself as \`PUBLIC_READ_ONLY_MARKET_SMOKE\`, records \`credentials_used=false\`, \`orders_submitted=0\`, \`order_permission=false\`, and preserves the production release flags as NO-GO.

Public market statistics/orderbook do **not** prove an account's fee schedule, authenticated balance, market-specific minimum notional or precision filters, or status of a particular order. Those fields are intentionally recorded as \`NOT_VERIFIED\` or \`NOT_QUERIED\`; they must never be inferred from mocked responses.

No private API key, POST/PUT/DELETE request, order placement, cancellation or account read is part of this workflow. Tests of malformed API payloads are synthetic contract tests, not live endpoint evidence.

## Release impact

A passing read-only public smoke improves market-data observability only. It does not satisfy the economic acceptance gate, approve a strategy, grant order permission, verify real order execution or authorize live trading. The flags remain:

- Champion = null
- order_permission = false
- promotion_status = BLOCKED_NO_ECONOMIC_GATE
- economic_acceptance = NO-GO
- trading_authorization = NO-GO
