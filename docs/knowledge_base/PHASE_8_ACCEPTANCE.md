# Phase 8 Acceptance: Local-First Multi-User / Multi-Exchange Foundation

Status: IMPLEMENTATION SOURCE CI PASSED; REVIEW-ONLY; NO TRADING AUTHORIZATION.

Implementation evidence SHA: b59123c3f86ed048c58fa6668eb345aa2c3e953e.
Push CI (implementation SHA): https://github.com/angry0tiger-sudo/AliX/actions/runs/39853084305
PR CI (implementation SHA): https://github.com/angry0tiger-sudo/AliX/actions/runs/39853089921
The documentation update creates a new commit and must be checked on that exact SHA before Phase 8 is declared ready.

Base SHA: 900eeef90b4e7c1d0e5652fad22f932a2aad9978.
Owner Directive: docs/governance/owner-directives/2026-10-22-phase-8.md.
ADR: docs/architecture/decisions/ADR-0008-phase-8-local-single-user-control-plane.md.
Canonical MRD: docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md, section 2.3.
Canonical Roadmap: docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md, Phase 8.

## Required acceptance checklist

- [x] A non-empty random owner token is required at start; no default or anonymous admin path exists (unit tests passed).
- [x] Missing/wrong token fails closed; comparison is constant-time (unit tests passed).
- [x] Validation errors do not echo untrusted field values; credentials are never accepted as raw exchange API fields (secret-reflection tests passed).
- [x] Tenant ownership and membership checks are covered by the passing tenant authorization tests.
- [x] Tenant A cannot enumerate/read Tenant B resources when IDs are forged (cross-tenant denial tests passed).
- [x] Revoked membership cannot continue to access the Tenant (revocation test passed).
- [x] Disabling a Tenant blocks subsequent scoped access and queues disable commands for its Workers (unit test passed).
- [x] Disabling an account blocks new Worker creation, revokes Worker credentials and queues stop/disable (unit test passed).
- [x] A delayed START acknowledgement cannot change a disabled Worker back to STARTING or RUNNING (regression test passed).
- [x] Account-to-Worker binding checks tenant, account, exchange and environment (binding-denial tests passed).
- [ ] PostgreSQL Control Plane schema and per-Worker SQLite data contracts are present and version-checked.
- [ ] Concurrent Worker creation for one slot is rejected; start/stop commands are idempotent within current state.
- [x] Worker command claims have leases; expiry/reclaim and stale-dispatcher acknowledgement denial tests passed.
- [x] Worker credential hashes, IDs and generations are scoped; wrong token or stale generation cannot heartbeat (unit tests passed).
- [x] Stop/disable credential revocation and fail-closed runtime behaviour are covered by passing tests.
- [ ] Worker cannot read the Control Plane DB or another Worker's SQLite volume; no Docker socket is mounted into the API or Worker.
- [x] Docker Compose static security contracts pass: loopback API, no published PostgreSQL port, non-root/read-only/capability/resource controls.
- [x] Runtime dispatcher fixed-service and scoped-token contracts pass; it does not receive exchange credentials.
- [x] CI passes with coverage >=90%, strict Mypy, Ruff, Bandit, dependency audit, architecture contracts and complexity limits; no quality gate was relaxed.
- [x] Phase report, Owner Directive, ADR, project index and document validator checks pass in CI.
- [ ] Exact-head Push CI and PR CI are green with mock-only tests and no exchange calls.
- [x] order_permission remains false, Champion remains null, economic acceptance NO-GO and trading authorization NO-GO (safety/runtime checks passed).

## Explicit non-goals

No actual Freqtrade Worker start, paper/dry-run session, live API key, exchange read/write, real account, OIDC server, external provider, model download, host Docker/network mutation or trading is performed by CI. The worker profile is a heartbeat-only isolation proof scaffold, not a trading runtime.

## Required tests

Run functional API tests, cross-tenant negative authorization tests, secret-reflection tests, lease/replay tests, generation-scoped heartbeat tests, runtime failure tests, SQLite identity isolation checks and Compose security contract tests. Failed controls must remain failed; do not lower an existing quality threshold.


## CI evidence: implementation source SHA

Verified source commit: `b59123c3f86ed048c58fa6668eb345aa2c3e953e`

- Push CI: [run 39853084305](https://github.com/angry0tiger-sudo/AliX/actions/runs/39853084305) — `quality` and `freqtrade-runtime-smoke` both completed successfully.
- PR CI: [run 39853089921](https://github.com/angry0tiger-sudo/AliX/actions/runs/39853089921) — `quality` and `freqtrade-runtime-smoke` both completed successfully.
- Ruff lint/format, strict Mypy, Bandit, dependency resolution/lock and audit, architecture contracts, and Xenon complexity gate passed.
- Unit suite: **195 passed**, 13 warnings; statement coverage **90.77%** (required minimum 90%).
- Phase 1 safety/dependency contracts: **4 passed**. Starter Kit/documentation validation passed.
- Runtime smoke verified Freqtrade loads with closed configuration, with synthetic Phase 4/5 mechanics checks. No exchange connection or real order was requested.

## Acceptance boundaries and residual evidence

- Actual concurrent multi-transaction worker-slot contention has not been demonstrated by the named sequential duplicate-slot test.
- PostgreSQL production/runtime integration and actual cross-container/kernel/filesystem isolation were not run. Docker Compose was validated as configuration only; tests cover static security contracts and per-worker SQLite identity behaviour.
- The Worker remains heartbeat-only and does not import/start Freqtrade or submit/cancel orders.
- `order_permission = false`; `Champion = null`; `promotion_status = BLOCKED_NO_ECONOMIC_GATE`; economic acceptance and trading authorization remain `NO-GO`.
- These CI links are evidence for the implementation source SHA above. The documentation commit has its own SHA, which must be validated by its own exact-head Push and PR runs. A commit cannot contain its own full SHA and CI run ID without changing that commit; the final documentation-commit SHA and exact-head CI links are therefore recorded in PR #10 after those runs finish.
