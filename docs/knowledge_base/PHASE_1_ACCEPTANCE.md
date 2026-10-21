# Phase 1 Acceptance: Freqtrade Base and Connectivity Manager

A check is accepted only with CI evidence on the candidate commit. Live exchange connections, live orders, real VPN/proxy/tunnel changes, and real Bale sends are out of scope for this phase.

- [ ] Direct dependencies resolve on Python 3.12 and the transitive lock matches a clean pip-tools regeneration.
- [ ] Freqtrade 2026.7 is installed and its CLI/version smoke check passes without connecting to an exchange.
- [ ] Freqtrade configuration remains Spot + dry-run + stopped; no credentials, enabled markets, live order capability or public API server are introduced.
- [ ] VPN, proxy and tunnel route descriptors use injected activation callbacks and perform no OS/network mutation by default.
- [ ] Connectivity failover sorts enabled routes by priority, checks the allowlist before probing, continues after health/activation failure, and fails closed if none succeeds.
- [ ] Nobitex mapping follows the pinned Novaryn contract: Spot-only metadata, canonical BASE/USDT checks, Decimal limits, client order IDs limited to 32 characters, explicit known-status mapping, and unknown statuses fail closed.
- [ ] No actual order-write transport is available in this adapter. Payload construction tests are not authorization to submit orders.
- [ ] Bale uses the pinned immutable BaleIO commit, loads token/chat ID from environment, defaults chat ID to 832112407, allows override, enforces an explicit allowlist, and defaults reporting to disabled.
- [ ] Bale and Nobitex tests use fake transports; the suite makes no external API requests.
- [ ] Ruff, strict Mypy, Bandit, pip-audit, import-linter, Xenon, all tests, and at least 90% coverage pass on the same commit.
- [ ] Pre-commit config validation and Starter Kit integrity checks pass.
- [ ] Current status, source pins and unverified deployment gates are documented.

## Deployment gates beyond Phase 1

- Authorized read-only Nobitex API validation with real, non-writing credentials and verified market metadata.
- Runtime validation of the Novaryn-compatible Freqtrade integration boundary, including candle, balance/open-order reads and reconciliation/recovery.
- Authorized Bale delivery test with a real token and the configured chat allowlist.
- Verify Novaryn source license/provenance before any further literal code reuse.
- Verify operational VPN/proxy/tunnel clients and recovery behavior on an authorized target environment.

Do not mark these deployment gates passed based on mocks or successful Python imports.
