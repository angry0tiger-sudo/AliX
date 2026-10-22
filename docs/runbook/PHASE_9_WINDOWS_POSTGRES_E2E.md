# Phase 9 Windows and PostgreSQL Validation Runbook

Status: PREPARED FOR FUTURE OWNER-RUN VALIDATION; not executed by the assistant.

## Purpose and safety boundary

This runbook validates the current Phase 9 synthetic end-to-end harness and a disposable real PostgreSQL Control Plane database from a Windows machine controlled by the owner. It does not run the production `compose.yaml`, start the Freqtrade trading lifecycle, access an exchange, or submit orders. GitHub Actions also runs PostgreSQL integration against an ephemeral PostgreSQL service using synthetic data.

Non-negotiable configuration:
- `order_permission = false`; `Champion = null`; promotion and trading authorization remain `NO-GO`.
- No real exchange credentials are needed. Keep `NOBITEX_API_KEY`, `NOBITEX_API_SECRET`, `BALE_BOT_TOKEN` empty and reporting disabled.
- Never copy a real `.env`, database dump, account response or credential into GitHub, issues, CI artifacts or test fixtures.
- The heartbeat-only Phase 8 Worker remains inert. Do not start `control_plane.runtime_cli`, the Worker profile, Freqtrade trade commands or production services as part of this runbook.

## Prerequisites

Owner-managed Windows 11 with a supported Docker Desktop/WSL2 setup, PowerShell, Python 3.12, Git, and permission to run a disposable local PostgreSQL test container. Follow the machine's approved change controls; do not weaken machine-wide PowerShell execution policy just for this test.

## 1. Check out the reviewed candidate

Use GitHub to identify the reviewed Phase 9 candidate. In the repository root, record the full SHA and make sure it matches the commit approved for local verification. Do not use an unreviewed moving branch head as the test identifier.

```powershell
git status --short
git rev-parse HEAD
py -3.12 --version
docker compose version
```

This runbook is documentation; the assistant has not run these commands on the owner's system.

## 2. Prepare the locked Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.2
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pytest tests/unit/test_phase9_simulation_ledger.py -q
.\.venv\Scripts\python.exe scripts/run_phase9_e2e.py --output artifacts/phase9-e2e.json --commit-sha (git rev-parse HEAD)
```

Expected behaviour: Phase 9 emits a JSON report labelled `SYNTHETIC_E2E_ONLY`, records zero exchange calls and zero real-order attempts, leaves `order_permission=false`, and sets `Champion=null`. PnL in that artifact is synthetic only and must remain `NOT_RECONCILED`.

## 3. Start only the dedicated test PostgreSQL service

The separate `compose.phase9-test.yaml` file is for disposable test PostgreSQL only. Its host port binds to `127.0.0.1:55432`; it does not alter the protected production `compose.yaml`, which keeps PostgreSQL on an internal network without a published port.

```powershell
docker compose --file compose.phase9-test.yaml --profile phase9-test up -d --wait postgres-phase9-test
$env:ALIX_TEST_POSTGRES_URL = "postgresql+psycopg://phase9_test:phase9-only-synthetic-password@127.0.0.1:55432/alix_phase9_test"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_phase9_postgres.py -q
```

The username/password/database above are disposable test fixtures, not project, user or exchange secrets. The test creates fake tenant/account/Worker identifiers and verifies PostgreSQL constraints and transactions only. It does not create exchange credentials, call a provider, or start the Worker/Freqtrade.

When finished:

```powershell
Remove-Item Env:ALIX_TEST_POSTGRES_URL
docker compose --file compose.phase9-test.yaml --profile phase9-test down -v
```

The `down -v` command here applies only to the test-only file/profile. Do not use `down -v` against the production `compose.yaml`, which has persistent state volumes.

## 4. Optional quality validation for the reviewed commit

```powershell
.\.venv\Scripts\python.exe -m compileall -q control_plane risk connectivity external_data services integrations user_data/strategies user_data/freqaimodels tests scripts
.\.venv\Scripts\ruff.exe check control_plane risk connectivity external_data services integrations user_data/strategies user_data/freqaimodels tests scripts
.\.venv\Scripts\ruff.exe format --check --diff control_plane risk connectivity external_data services integrations user_data/strategies user_data/freqaimodels tests scripts
.\.venv\Scripts\python.exe -m mypy --config-file mypy.ini
.\.venv\Scripts\bandit.exe -r control_plane risk connectivity external_data services integrations user_data/strategies user_data/freqaimodels scripts -ll
.\.venv\Scripts\python.exe scripts/validate_project_docs.py
```

Use the project's existing locked environment and preserve the CI coverage floor of 90%. Do not regenerate the Starter Kit archive as a shortcut; the integrity validator is expected to verify its committed checksums.

## 5. What success does and does not establish

A successful PostgreSQL test demonstrates the Control Plane's tested schema/repository paths against a real PostgreSQL server in this disposable environment. It does not demonstrate production roles, real migration/backup/restore, Windows ACLs, Docker daemon isolation, provider compatibility, Freqtrade database integration, exchange reconciliation or model profitability.

A green synthetic test is not economic acceptance, permission to trade, or proof of live-market behaviour. Keep any failed/unmeasured acceptance gate explicitly open. No actual order timeout, partial fill, cancel or exchange restart is invoked by this runbook.
