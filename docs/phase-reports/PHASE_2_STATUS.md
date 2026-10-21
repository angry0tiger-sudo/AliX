# AliX Phase 2 Status Report

Status: **IN PROGRESS / CONTRACT-ONLY IMPLEMENTATION / LIVE EXECUTION BLOCKED**

Owner approval: Phase 1 accepted; Phase 2 authorized.

## Work scope

Phase 2 follows the roadmap goal of reusing the pinned Novaryn Nobitex Spot
adapter contract with the smallest compatible boundary. This work does not
rewrite the exchange client, copy full Novaryn modules, patch Freqtrade core, or
create a second order lifecycle.

## Contract implementation in this candidate

- Order status classification now distinguishes partial fills from open orders
  and rejects invalid matched/total amount relationships.
- Pure response contracts normalize read-only Spot balance and order snapshots,
  including amount, matched amount, price, average price, total price, fee fields,
  pair identity and client order ID.
- Candidate order validation checks Spot/USDT identity, supported order types
  and sides, decimal precision, and market minimum/maximum limits.
- Pagination inputs are bounded; retry policy is explicitly scoped to
  idempotent read requests only.
- All validation functions are pure; CI fixtures and mocks do not call Nobitex.

## Safety state

- No real API request, order submission or cancellation has been performed.
- The adapter still has no default network transport or order-write method.
- Freqtrade remains dry-run, stopped, with `max_open_trades=0`, empty exchange
  credentials and an empty whitelist.
- No real API token or secret is stored in GitHub.
- Contract test results must be read from the PR's GitHub Actions checks for
  the exact head SHA; imports and locally constructed fixtures are not proof of
  external schema compatibility.

## Open acceptance gates

1. Successful CI on the final Phase 2 candidate SHA.
2. Validate current official Nobitex response schemas and actual USDT symbols,
   precision and limits using an authorized read-only environment.
3. Exercise candle, balance, open-order, status, fills/fees, pagination and
   rate-limit read paths using permitted credentials.
4. Prove restart reconciliation and recovery from an unknown submission
   outcome without blind resubmission or duplicate orders.
5. Demonstrate the single Freqtrade -> adapter -> Nobitex path and runtime
   callback compatibility without patching Freqtrade core.
6. Verify Novaryn source license/provenance before any further literal reuse.

## Decision

Phase 2 is **NOT ACCEPTED YET**. The mock-only contract foundation is the first
implementation slice; no live trading or phase promotion is authorized by this
report. All external and execution gates remain open until independently
evidenced.
