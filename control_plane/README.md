# Control Plane — Phase 8

The Control Plane owns tenants, memberships, account bindings, Worker desired/observed lifecycle, runtime command leases and audit events. It uses PostgreSQL for shared management state. The API is local-first and currently supports a single-owner Bearer token only; OIDC is not enabled.

## Safety boundary

- The API does not accept raw exchange API credentials and does not connect to any exchange.
- Worker containers receive only a generation-scoped heartbeat token, stored server-side as a SHA-256 digest.
- Each Worker has its own SQLite file on a dedicated named volume.
- The API and Worker do not mount the Docker socket. A trusted host-side dispatcher executes only the fixed Compose Worker service.
- The Worker implementation is heartbeat-only. It does not load Freqtrade, place orders, cancel orders or request external data.
- order_permission remains false; this package does not authorize any trading session.

See ADR-0008, the Phase 8 acceptance checklist and the two Phase 8 runbooks before any local deployment.
