# Phase 8 Local Compose Runbook

This runbook describes future owner-controlled local deployment of the non-trading Control Plane scaffold. Do not use it to start a Freqtrade trading session.

## Local secret setup

1. Copy .env.example to .env. The .env file is ignored by Git.
2. Set POSTGRES_PASSWORD to a long random URL-safe value and ALIx_ADMIN_TOKEN to a separate random value containing at least 32 characters. Do not reuse exchange or Bale credentials.
3. Keep POSTGRES_USER, POSTGRES_DB, and authentication mode as documented. Never paste .env contents into issues, logs or a chat.

## Start only the base services

From a controlled machine with Docker Compose available, run:

    docker compose up -d --build postgres control-plane

The API is published only on 127.0.0.1:8080. PostgreSQL has no published host port and is reachable only from the internal Compose network. Check GET /healthz and use the owner Bearer token for management endpoints.

## Worker lifecycle dispatcher

The host-side dispatcher runs outside the API container because it needs Docker CLI authority. Never mount the Docker socket into the API or Worker.

    python -m control_plane.runtime_cli --env-file .env

This dispatcher polls the authenticated runtime outbox and can create or stop only the declared heartbeat-only Compose Worker service. It has host-level Docker authority, so run it as the local owner on the trusted machine. Stop it with the normal interrupt signal.

The Worker is a security/isolation scaffold that writes heartbeat metadata to its own SQLite volume. It is not Freqtrade, does not access exchange credentials, and does not trade. Do not add an exchange secret or replace the command with a trading process without a separately approved Owner Directive, risk review and release gate.

## Stop and preserve state

Use the Control Plane lifecycle endpoint to request Worker stop/disable; the dispatcher will process the command and acknowledge the result. Stop the base services with:

    docker compose stop control-plane postgres

Do not use docker compose down -v unless intentionally deleting the PostgreSQL and Worker state volumes. Backups, migrations and restore procedures are not yet production-approved.

## Recovery and troubleshooting

- If the Control Plane health check fails, do not start new Workers.
- If a Worker token is rejected or its generation is stale, its process exits and the record must be investigated before retry.
- A failed stop/start command leads to quarantine; inspect the actual container and audit record before recovery.
- A missing Docker CLI, failed image build or absent local .env blocks runtime dispatch. Never work around these errors by mounting the Docker socket into the API.
- The local token is not suitable for remote or untrusted-network exposure. The Compose port must remain loopback-only.
