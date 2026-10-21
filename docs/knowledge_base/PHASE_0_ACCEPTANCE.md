# Phase 0 Acceptance Checklist

A checked item requires evidence on the exact candidate commit. A file existing is not proof of acceptance.

- [ ] Starter Kit archive integrity and embedded validation requirements reviewed.
- [ ] Canonical docs and project rules are available in-repository; original Starter Kit preserved.
- [ ] Standard AliX directory structure exists, with safe placeholders where no implementation is approved.
- [ ] Runtime and direct dependencies are pinned; clean-environment resolution and compatibility are verified.
- [ ] `.env.example` contains placeholders only; real credentials are absent.
- [ ] Trading configuration defaults to `dry_run: true`, disables live trading, and uses no real exchange keys.
- [ ] `AliXStrategy.py` is a conservative scaffold; missing/invalid data cannot create an entry signal.
- [ ] Data Quality Gate covers missing columns, NaN/inf, stale data and timestamp/order checks, with unit tests.
- [ ] Connectivity Manager has explicit route policy, health checks, fail-closed behavior and mock tests.
- [ ] Reporting Engine produces deterministic mock reports; Bale transport is mocked unless a future separately approved integration test is configured.
- [ ] GitHub Actions runs on push and pull request with no secrets and no live network/trading.
- [ ] Ruff lint and formatter checks pass in CI.
- [ ] Mypy strict type checking passes in CI.
- [ ] Bandit security scan and pip-audit dependency scan pass in CI.
- [ ] pytest-cov reaches the configured minimum coverage threshold in CI.
- [ ] import-linter architecture contracts and Xenon complexity thresholds pass in CI.
- [ ] Pre-commit hooks are configured for all quality/security/architecture gates.
- [ ] Starter Kit validator verifies archive integrity, pinned Git blob checksum, and required embedded documentation in CI.
- [ ] All unit tests and repository documentation checks pass in CI.
- [ ] Novaryn adapter URL, immutable commit, license and contract-test evidence are recorded; otherwise compatibility remains a blocker. The adapter task is currently marked OPEN / BLOCKED in `integrations/novaryn/README.md`.
- [ ] Data/model/provider licensing, market whitelist evidence, Bale timezone and recipient allowlist are recorded or explicitly marked blocked. The sample market whitelist is explicitly `NEEDS_LOCAL_REVIEW`, with all markets disabled.
- [ ] Phase report links the exact commit and successful CI run. No phase acceptance before all mandatory items pass.

## Known evidence gaps at bootstrap
The Starter Kit archive is present at repository root. The adapter repository URL/commit and actual market metadata are not present in the root listing inspected during bootstrap. These must not be fabricated.
