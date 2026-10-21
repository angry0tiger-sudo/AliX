# Phase 2 Acceptance: Nobitex Spot Adapter Integration

Status: **CONTRACT/CI GATES PASS ON THE IMPLEMENTATION CANDIDATE; EXTERNAL DEPLOYMENT GATES OPEN**.

Owner approval: Phase 1 accepted; Phase 2 authorized.

## Contract and CI gates

- [x] Clean GitHub Actions execution passed for implementation SHA `8743469f5f072ea482e5d016859ebd7e8d32a92c` (push run and pull-request run succeeded). The documentation/template update in this PR is revalidated by the subsequent commit's checks.
- [x] Balance and order payload normalization rejects malformed or ambiguous
  response data and preserves decimal precision.
- [x] Partial fill, full fill, cancellation and unknown order status are
  represented and tested.
- [x] Market validation enforces Spot/USDT pair identity, supported side/type,
  precision, minimums and maximums.
- [x] Retry policy is explicitly limited to idempotent reads; unknown order-write
  outcomes never trigger blind retries.
- [x] Pagination inputs are validated and bounded.
- [x] CI tests remain fixture/mock-only. No real Nobitex request, order
  submission/cancellation, API credential, Bale send, VPN or tunnel is used.
- [x] Freqtrade core/site-packages are not patched, and a second order lifecycle
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
