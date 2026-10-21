# Nobitex Integration Boundary

AliX Phase 1 now includes a Novaryn-derived contract adapter in `integrations/nobitex/adapter.py`.

- Reference repository: https://github.com/angry0tiger-sudo/Novaryn
- Reference commit: `54e66ad40b5a7e422002619f7dad9378ecdf09f9`
- Contract: Spot/USDT only; normalized market metadata; positive finite decimal payload values; client order ID limited to 32 characters; known order state map; unknown state fails closed.
- Transport: injected read-only protocol only. No default HTTP transport is configured and no Nobitex API is called by tests or CI.
- Writes: payload construction is only a validation aid. `submit_order` unconditionally raises `LiveExecutionDisabled`.
- Freqtrade: retain its stopped/dry-run config. Upstream CCXT does not natively support Nobitex, so this scaffold does not claim that the stock Freqtrade resolver can place Nobitex orders.

Read [Novaryn reference contract](../../docs/architecture/NOVARYN_REFERENCE.md). Live read-only verification, runtime shim/resolver validation, private credential checks and order reconciliation remain deployment gates.
