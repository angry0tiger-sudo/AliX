# Phase 8 Acceptance: Local-First Multi-User / Multi-Exchange Foundation

Status at implementation start: BLOCKED pending exact-head CI and runtime evidence.

Base SHA: 900eeef90b4e7c1d0e5652fad22f932a2aad9978.
Owner Directive: docs/governance/owner-directives/2026-10-22-phase-8.md.
ADR: docs/architecture/decisions/ADR-0008-phase-8-local-single-user-control-plane.md.
Canonical MRD: docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md, section 2.3.
Canonical Roadmap: docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md, Phase 8.

## Required acceptance checklist

- [ ] A non-empty random owner token is required at start; no default or anonymous admin path exists.
- [ ] Missing/wrong token fails closed; comparison is constant-time.
- [ ] Validation errors do not echo untrusted field values; credentials are never accepted as raw exchange API fields.
- [ ] Tenant ownership and membership are checked on every tenant-scoped operation.
- [ ] Tenant A cannot enumerate, read or mutate Tenant B resources when IDs are forged.
- [ ] Revoked membership cannot continue to access the Tenant.
- [ ] Disabling a Tenant blocks subsequent scoped access and queues disable commands for every Worker under that Tenant.
- [ ] Disabling an account blocks new Worker creation, immediately revokes Worker credentials and queues stop/disable.
- [ ] A delayed START acknowledgement cannot change a disabled Worker back to STARTING or RUNNING.
- [ ] Account-to-Worker binding checks tenant, account, exchange and environment.
- [ ] PostgreSQL Control Plane schema and per-Worker SQLite data contracts are present and version-checked.
- [ ] Concurrent Worker creation for one slot is rejected; start/stop commands are idempotent within current state.
- [ ] Worker command claims have leases; an expired claim can be reclaimed and the previous dispatcher cannot acknowledge it.
- [ ] Worker credential hashes, IDs and generations are scoped; wrong token or stale generation cannot heartbeat.
- [ ] Stop, disable or failed runtime acknowledgement revokes the Worker credential.
- [ ] Worker cannot read the Control Plane DB or another Worker's SQLite volume; no Docker socket is mounted into the API or Worker.
- [ ] Docker Compose pins service configuration, publishes the API only on loopback, hides PostgreSQL port and applies non-root/read-only/capability/resource controls.
- [ ] Runtime dispatcher uses only a fixed Compose Worker service and scoped token; it never receives an exchange credential.
- [ ] Regression CI retains coverage >=90%, Strict Mypy, Ruff, Bandit, dependency audit, architecture contracts and complexity limits.
- [ ] Phase report, Owner Directive, ADR, project index and document validator include the new phase.
- [ ] Exact-head Push CI and PR CI are green with mock-only tests and no exchange calls.
- [ ] order_permission remains false, Champion remains null, economic acceptance NO-GO and trading authorization NO-GO.

## Explicit non-goals

No actual Freqtrade Worker start, paper/dry-run session, live API key, exchange read/write, real account, OIDC server, external provider, model download, host Docker/network mutation or trading is performed by CI. The worker profile is a heartbeat-only isolation proof scaffold, not a trading runtime.

## Required tests

Run functional API tests, cross-tenant negative authorization tests, secret-reflection tests, lease/replay tests, generation-scoped heartbeat tests, runtime failure tests, SQLite identity isolation checks and Compose security contract tests. Failed controls must remain failed; do not lower an existing quality threshold.
