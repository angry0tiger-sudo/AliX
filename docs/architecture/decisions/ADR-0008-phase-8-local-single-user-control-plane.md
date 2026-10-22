# ADR-0008: Local-First Control Plane, Single-User Auth and Isolated Worker Containers

- Status: Accepted for implementation on the Phase 8 branch; release/runtime security review remains separate.
- Date: 2026-10-22.
- Owner directive: docs/governance/owner-directives/2026-10-22-phase-8.md.
- Base commit: 900eeef90b4e7c1d0e5652fad22f932a2aad9978.

## Context

The project owner is currently the only operator and intends to run AliX locally in the future. A separate OIDC identity server would add avoidable setup to the first deployment, but the security model must remain extensible to more users. PostgreSQL is required for shared Control Plane metadata, while existing Freqtrade state must remain isolated per Worker. The project is not authorized for trading.

## Decisions

1. Use a PostgreSQL database for tenants, memberships, account bindings, Worker lifecycle, versioned generations, runtime outbox commands and audit events. The Control Plane uses SQLAlchemy Core with declared composite foreign keys and uniqueness constraints. SQLite is used by deterministic integration tests and by each Worker only for its own private runtime/heartbeat state.
2. Bootstrap the local owner with a single high-entropy Bearer token stored in the owner's ignored .env file. The token must have at least 32 characters, has no default, is compared using a constant-time comparison and is never returned by an endpoint. The server accepts only auth mode single_user. OIDC requires a separate ADR before activation.
3. Publish the HTTP API at loopback only through Compose. The PostgreSQL port is not published. API schemas reject unknown fields and validation responses omit rejected input values to avoid reflecting accidental secrets.
4. Enforce authorization in repository transactions on every tenant-scoped operation. Membership is looked up on each request; revoked memberships therefore do not rely on cache invalidation. Account and Worker bindings must match tenant, exchange and non-trading environment. Only local/test environments are accepted.
5. Use one non-root Docker container per Worker, with a dedicated named SQLite volume, read-only root, dropped capabilities, no-new-privileges, internal networking and CPU/memory/PID limits. The Worker code is inert: it records its own identity and heartbeat and contains no trading or exchange implementation.
6. Keep container management outside the Control Plane container. A trusted host-side dispatcher polls an authenticated outbox and uses the Docker CLI with a fixed Compose worker service, deterministic container names, generation labels and scoped per-Worker tokens. The API and Worker containers do not receive Docker socket access. The host dispatcher is a privileged local component and must be reviewed as part of the deployment threat model.
7. Store only a SHA-256 digest of each Worker token. A credential is accepted only for the exact Worker ID and current generation and is revoked on stop/disable or failed runtime acknowledgement. Unknown or stale generations fail closed.
8. Record management and runtime acknowledgements in the audit table. Audit entries are append-only at the application API boundary; direct database administrators remain trusted and production database-role hardening is a deployment gate.
9. Do not introduce a broker or shared writable cache in Phase 8. A transactional runtime command outbox and lease-based claims prevent two dispatchers from owning the same unexpired command. Any future shared queue/cache must have a separate threat analysis and ADR.
10. The Risk Gate, Kill Switch, Risk Supervisor and order lifecycle remain outside this phase's execution path. order_permission must remain false, and Worker lifecycle success is not a trading permission.

## Trust boundaries

- Owner local machine: trusted for the initial single-user token and host Docker CLI; compromise of the owner account or Docker daemon is outside the container-only threat boundary.
- Control Plane API: trusted authority for user/account ownership, policy metadata and lifecycle requests; it has no exchange API key and no Docker socket.
- PostgreSQL: durable shared Control Plane metadata, isolated on an internal network; production DB user separation and backup/restore review remain required.
- Worker container: untrusted for authorization decisions. Its token grants only heartbeat for its own ID and generation; it receives no admin token or database password.
- Worker SQLite volume: writable only by its Worker container identity under Compose configuration; file ACLs and actual host isolation require target-runtime verification.
- Exchange integrations: inert metadata bindings only; no exchange transport or credentials are active in this phase.

## Alternatives considered

- A shared SQLite control database: rejected because concurrency and central ownership constraints are more complex for lifecycle/lease coordination.
- OIDC as a mandatory first dependency: rejected for this single-owner bootstrap, retained as a separately approved future auth backend.
- Mounting /var/run/docker.sock into the API container: rejected because it gives the API a high-impact host control capability and couples business authorization to container-admin authority.
- Shared PostgreSQL for Worker trade state: rejected because it expands the blast radius and makes application tenant filters a security boundary for Freqtrade state.
- One multi-account Worker: rejected because credentials, process memory, logs, caches and failure containment would be shared.
- Shared Redis in the first iteration: rejected until a concrete async requirement justifies another cross-tenant service.

## Consequences and blockers

- A local .env token is a practical single-user guard, not a substitute for TLS/SSO on an exposed network. The API is bound to loopback and is not intended for remote deployment.
- The host-side dispatcher can control Docker containers and is privileged relative to the containers. It remains a host trust boundary, not a sandboxed unprivileged service.
- The Phase 8 Worker is heartbeat-only. It does not launch Freqtrade and does not establish real exchange compatibility, risk runtime wiring or production isolation.
- Schema initialization is bootstrap-only. A migration policy, production database roles, backup/restore drills, Windows/WSL2 filesystem-permission verification, actual Compose build/stop/restart tests and security review remain release blockers.
- No live or paper session, exchange request or order was enabled. Economic acceptance and trading authorization remain NO-GO.
