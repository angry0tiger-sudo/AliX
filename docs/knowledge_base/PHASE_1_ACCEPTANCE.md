# Phase 1 Acceptance: Freqtrade Base and Connectivity Manager

Status of implementation gates: **PASS on CI candidate**. Live exchange connections, live orders, real VPN/proxy/tunnel changes, and real Bale sends are outside this phase's scope and remain deployment gates.

- [x] Direct dependencies resolve on Python 3.12 and the transitive lock matches a clean pip-tools regeneration constrained by the committed pins.
- [x] Freqtrade 2026.7 is installed and its CLI/version smoke check passes without connecting to an exchange.
- [x] Freqtrade configuration remains Spot + dry-run + stopped; no credentials, enabled markets, live order capability or public API server are introduced.
- [x] VPN, proxy and tunnel route descriptors use injected activation callbacks and perform no OS/network mutation by default.
- [x] Connectivity failover sorts enabled routes by priority, checks the allowlist before probing, continues after health/activation failure, and fails closed if none succeeds.
- [x] Nobitex mapping follows the pinned Novaryn contract: Spot-only metadata, canonical BASE/USDT checks, Decimal limits, client order IDs limited to 32 characters, explicit known-status mapping, USDT symbol discovery with ambiguity checks, and unknown states fail closed.
- [x] No actual order-write transport is available in the adapter. Payload construction tests are not authorization to submit orders.
- [x] Bale uses the pinned immutable BaleIO commit, loads token/chat ID from environment, defaults chat ID to 832112407, allows override, enforces an explicit numeric recipient allowlist, and defaults reporting to disabled.
- [x] Bale and Nobitex tests use fake transports; the suite makes no external exchange API requests or real Bale sends.
- [x] Ruff, strict Mypy, Bandit, pip-audit, import-linter, Xenon, all tests, and at least 90% coverage pass on the same workflow candidate.
- [x] Pre-commit config validation and Starter Kit integrity checks pass.
- [x] Current status, Novaryn source pin, dependency pins, implementation scope and unverified deployment gates are documented.

## CI evidence

- Quality test suite: **45 passed**.
- Coverage: **95.49%**, above the enforced 90% minimum.
- Phase 1 safety/dependency contract tests: **4 passed**.
- Freqtrade runtime smoke: install/version/strategy-load/config checks passed; no exchange connection was requested.
- Ruff lint/format, strict Mypy, Bandit, resolver dry run, locked dependency diff, pip-audit, import-linter, Xenon, and Starter Kit validation: passed.

Use the [AliX GitHub Actions history](https://github.com/angry0tiger-sudo/AliX/actions) and match the most recent run's head SHA to the current report commit.

## Deployment gates beyond Phase 1

These are intentionally not represented as completed by mock tests:

- Authorized read-only Nobitex API validation with real, non-writing credentials and verified current market metadata.
- Runtime validation of the custom Nobitex/Freqtrade integration boundary, including candle/balance/open-order reads and reconciliation/recovery. Upstream stock Freqtrade is not claimed to support Nobitex natively.
- Authorized Bale delivery test with a real token and the configured chat allowlist.
- Verify Novaryn source license/provenance before any additional literal source reuse.
- Verify operational VPN/proxy/tunnel clients and recovery behavior on an authorized target environment.

These deployment gates must not be marked as passed based on mocks or successful Python imports.
