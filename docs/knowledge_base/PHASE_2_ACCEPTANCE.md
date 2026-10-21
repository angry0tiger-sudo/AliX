# Phase 2 Acceptance: Nobitex Spot Adapter Integration

Status: **IN PROGRESS — contract-only implementation; deployment validation remains blocked**.

Owner approval: Phase 1 accepted; Phase 2 authorized.

## Contract and CI gates

- [ ] A clean CI run passes on the final Phase 2 candidate SHA.
- [ ] Balance and order payload normalization rejects malformed or ambiguous
  response data and preserves decimal precision.
- [ ] Partial fill, full fill, cancellation and unknown order status are
  represented and tested.
- [ ] Market validation enforces Spot/USDT pair identity, supported side/type,
  precision, minimums and maximums.
- [ ] Retry policy is explicitly limited to idempotent reads; unknown order-write
  outcomes never trigger blind retries.
- [ ] Pagination inputs are validated and bounded.
- [ ] CI tests remain fixture/mock-only. No real Nobitex request, order
  submission/cancellation, API credential, Bale send, VPN or tunnel is used.
- [ ] Freqtrade core/site-packages are not patched, and a second order lifecycle
  is not introduced.

## External deployment gates (not met by CI)

- [ ] Verify current official Nobitex response schemas and actual USDT market
  symbols, precision, minimums/maximums and trading availability.
- [ ] Validate authorized read-only candles, balances, open orders, fills/fees,
  pagination and rate-limit responses.
- [ ] Verify cancel/status behavior in an approved non-capital test environment;
  no order cancellation is performed by this CI suite.
- [ ] Prove timeout recovery and restart reconciliation by client order ID,
  including the case where submission outcome remains unknown.
- [ ] Demonstrate the single Freqtrade -> adapter -> Nobitex path and version-
  pinned runtime callbacks without patching Freqtrade core.
- [ ] Confirm Novaryn source license/provenance before further literal source reuse.

## Decision rule

Phase 2 must remain **NO-GO for live trading** until the required runtime,
schema, reconciliation and integration evidence is recorded. Successful mocks
and imports do not satisfy the external deployment gates.
