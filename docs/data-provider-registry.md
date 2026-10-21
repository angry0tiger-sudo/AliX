# External Data Provider Registry

Status: Phase 1 reference and test boundaries registered. Enabling external use still requires separate terms, license, limits, retention and runtime validation.

| Provider / dataset | Purpose | Access / terms | License review | Rate limit | Retention | Fallback | Status |
|---|---|---|---|---|---|---|---|
| Exchange OHLCV | Market candles | Nobitex terms and current response schema must be verified | Pending | Pending | Pending | Fail closed until approved | Read-only contract scaffold |
| Novaryn / Nobitex adapter | Exchange boundary | https://github.com/angry0tiger-sudo/Novaryn at 54e66ad40b5a7e422002619f7dad9378ecdf09f9 | Source provenance to verify before literal reuse | Provider-specific | N/A | Injected read-only transport | Contract tests only |
| Freqtrade | Bot framework | 2026.7 pinned to Novaryn documented integration baseline | Package licensing and compatibility reviewed separately | N/A | Local runtime | Remain stopped | Dry-run bootstrap |
| Bale Bot API | Report delivery | BaleIO pinned to 66be3000af6e62332e1764f227aed64847a0d9ba; endpoint not contacted in CI | BaleIO declares MIT; verify package metadata during resolved install | Pending | Do not retain secrets in logs | Local report artifact | Mock only |

An empty or pending registry cell is not approval. No provider is considered runtime-validated just because its package resolves or a mock transport test passes.
