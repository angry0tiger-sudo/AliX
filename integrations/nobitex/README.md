# Nobitex Integration Boundary

AliX references the Novaryn Nobitex integration contract at immutable commit
`54e66ad40b5a7e422002619f7dad9378ecdf09f9`. This repository does not vendor or
wholesale-copy the Novaryn source. Confirm authoritative source licensing and
provenance before any further literal code reuse.

## Phase 2 contract work

- `integrations/nobitex/adapter.py` retains the Spot/USDT market and payload
  validation boundary. Unknown order states fail closed; a positive matched
  amount below the requested amount is represented as `PARTIALLY_FILLED`.
- `integrations/nobitex/contracts.py` defines immutable balance/order snapshots,
  response normalization, market precision/minimum/maximum checks, bounded
  pagination inputs, and a retry policy scoped to idempotent read requests.
- The response helpers consume mappings supplied by callers. They perform no
  network calls and do not prove that Nobitex's current API response schema
  matches these contracts.
- The read transport remains injected and read-only. No default HTTP client,
  order-write transport, order cancellation method, or live order submission is
  enabled. Payload construction is not permission to send an order.
- Read retries must never be reused for order mutations. An order submission
  timeout is an unknown outcome and must be reconciled by client order ID and
  exchange state before any later retry decision.

## Safety and outstanding validation

CI uses only fake transports and sanitized fixtures. It does not contact
Nobitex, create/cancel orders, or use API keys. Freqtrade remains configured
with dry-run enabled, stopped initial state, zero open trades, blank credentials,
and an empty market whitelist. Stock Freqtrade is not claimed to support Nobitex
natively.

Still blocked pending a separately authorized validation environment:
- Confirm response schemas, currently supported USDT markets, precision and
  min/max values against current official API responses.
- Validate read-only candles, balances, open orders, fills/fees, pagination and
  rate-limit behavior using permitted credentials.
- Prove restart reconciliation and unknown-submission recovery without duplicate
  orders.
- Prove the complete Freqtrade-to-adapter integration boundary without patching
  Freqtrade core or creating a second order lifecycle.
