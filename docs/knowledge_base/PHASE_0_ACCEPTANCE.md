# Phase 0 Acceptance Checklist

A checked item requires evidence on the candidate commit. External integrations that cannot be verified from GitHub are accepted for Phase 0 only when explicitly recorded as blocked, with no compatibility or live-use claim.

- [x] Starter Kit archive integrity and embedded validation requirements reviewed by CI.
- [x] Canonical docs and project rules are available in-repository; original Starter Kit archive remains preserved.
- [x] Standard AliX directory structure exists, with safe placeholders where no implementation is approved.
- [x] Runtime/direct dependencies are pinned, transitive versions are locked in `requirements.lock`, clean resolver dry run passes, and CI verifies lock reproducibility.
- [x] `.env.example` contains placeholders only; real credentials are absent.
- [x] Trading configuration defaults to `dry_run: true`, disables live trading, and uses no real exchange keys.
- [x] `AliXStrategy.py` is a conservative scaffold; it emits no entry or exit signals in Phase 0 tests.
- [x] Data Quality Gate covers missing columns, NaN/inf, stale data, OHLCV relationships and timestamp ordering, with unit tests.
- [x] Connectivity Manager has explicit route policy, fail-closed behavior and mock tests.
- [x] Reporting Engine produces deterministic mock reports; delivery is disabled by default and uses injected sender/allowlist gates.
- [x] GitHub Actions runs on push and pull request with no secrets and no live network/trading.
- [x] Ruff lint and formatter checks pass in CI.
- [x] Mypy strict type checking passes in CI.
- [x] Bandit security scan and pip-audit against the transitive lock pass in CI.
- [x] pytest-cov passes the 90% minimum coverage threshold; latest verified coverage was 93.71%.
- [x] import-linter architecture contracts and Xenon complexity thresholds pass in CI.
- [x] Pre-commit hooks cover quality, typing, security, dependency audit, architecture, complexity, tests and coverage; `pre-commit validate-config` passes.
- [x] Starter Kit validator verifies ZIP integrity, required embedded docs, pinned Git blob SHA-1, pinned SHA-256, and the market whitelist sample against its JSON Schema.
- [x] All unit tests and repository documentation/integrity checks pass in CI.
- [x] Novaryn adapter is explicitly marked OPEN / BLOCKED; missing official URL, immutable commit, license and API contract are recorded, with no compatibility claim.
- [x] Market whitelist sample is explicitly NEEDS_LOCAL_REVIEW and contains no enabled markets. Provider/model/data licensing and Bale endpoint/timezone/recipient allowlist gaps are recorded as open tasks.
- [x] Phase 0 status report links the implementation candidate and successful CI evidence. Phase 1 remains unstarted until explicit owner approval.

## Evidence boundaries

The Novaryn repository URL/commit and authorized live Nobitex market metadata are not available in this GitHub-only environment. Bale delivery and licensing evidence also require source/owner verification. These are registered blockers for later integration/deployment gates, not fabricated as completed integrations.
