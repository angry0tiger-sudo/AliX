# Novaryn Reference for AliX

**Reference repository:** https://github.com/angry0tiger-sudo/Novaryn  
**Pinned inspected commit:** `54e66ad40b5a7e422002619f7dad9378ecdf09f9`

AliX Phase 1 uses the Novaryn integration contract as its reference, especially:

- `docs/NOBITEX.md` for read-only-first exchange validation, runtime-only credentials and reconciliation requirements.
- `docs/FREQTRADE.md` and `docs/architecture/NOBITEX_FREQTRADE_INTEGRATION_DECISION.yaml` for the explicit Freqtrade boundary and version baseline.
- `src/novaryn/infrastructure/exchange/nobitex/market_mapping.py` for strict Spot market metadata normalization.
- `src/novaryn/infrastructure/exchange/nobitex/status.py` for explicit order-state mapping and fail-closed handling of unknown states.
- `src/novaryn/infrastructure/exchange/nobitex/adapter.py` for amount semantics, the BASE/USDT restriction, client order IDs and conservative write-result handling.

See [Novaryn reference contract](../../docs/architecture/NOVARYN_REFERENCE.md) for source commit and blob SHA evidence.

AliX Phase 1 applies these deterministic contracts in `integrations/nobitex/adapter.py`; it does not wholesale copy Novaryn source files. There is no default HTTP transport, no real API call in CI, and all order writes raise `LiveExecutionDisabled`.

**Still gated:** runtime integration tests, authorized read-only Nobitex verification, reconciliation/recovery proof, and authoritative Novaryn license/provenance review before any further literal code reuse.
