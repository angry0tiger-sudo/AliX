# Local System Test Runbook (Phase 3)

Status: **Prepared for owner-run local validation; not executed by GitHub CI.**

## Purpose and boundary

This runbook verifies the repository in a clean local Python environment. It
separates deterministic unit/contract tests from tests that require a local
machine, explicitly authorized credentials, real provider access, or network
changes. GitHub-hosted CI remains mock-only.

No trade, exchange write, Bale message, VPN launch, proxy activation, SSH tunnel,
DNS change, route-table change, or credential-dependent request is required by
the commands in Sections 1–4.

## 1. Safe baseline

Before running commands:

- Start from a clean checkout of the reviewed branch and record `git rev-parse HEAD`.
- Use Python 3.12 and the committed `requirements.lock`; do not install floating dependencies.
- Keep `config.json` at its committed safe defaults: dry-run true, initial state stopped, `max_open_trades=0`, empty exchange credentials and empty pair whitelist.
- Keep `ALIx_ALLOW_LIVE_TRADING=false` and `ALIx_REPORTING_ENABLED=false`.
- Do not place real `.env`, API keys, private keys, proxy credentials, or tokens in Git, fixtures, captured terminal output, screenshots, or CI artifacts.
- Do not run live route callbacks. Tests must use injected fake probes and activators.

## 2. Windows PowerShell: clean environment and quality tests

Run from the repository root in a new terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==25.2
python -m pip install -r requirements.lock
python -m compileall -q connectivity services integrations user_data/strategies tests scripts
ruff check connectivity services integrations user_data/strategies tests scripts
ruff format --check connectivity services integrations user_data/strategies tests scripts
mypy --config-file mypy.ini
bandit -r connectivity services integrations user_data/strategies scripts -ll
lint-imports
xenon --max-absolute B --max-modules B --max-average A connectivity services integrations user_data/strategies scripts
pytest --cov=connectivity --cov=services --cov=integrations.nobitex --cov=user_data/strategies --cov-report=term-missing --cov-fail-under=90 -q
pytest tests/unit/test_phase1_contracts.py -q
python scripts/validate_project_docs.py
```

Expected result: every command exits with status 0. Coverage must remain at or
above the repository's 90% threshold. The Starter Kit validator must pass without
regenerating or modifying the archive.

If PowerShell blocks activation, use the project's approved execution policy or
call `.venv\Scripts\python.exe` and tool executables directly. Do not weaken
the machine-wide policy merely to run this test.

## 3. Focused Phase 3 tests

```powershell
pytest tests/unit/test_phase0_scaffolds.py -q
pytest tests/unit/test_data_quality_gate.py -q
pytest tests/unit/test_strategy_data_quality_gate.py -q
pytest tests/unit/test_connectivity_failover.py -q
```

Verify that results include all quality outcomes:

- `VALID` for well-formed, fresh, correctly ordered fixtures.
- `DEGRADED_ALLOWED` only for an explicitly configured tolerated gap or a
  provider record marked rate-limited but not blocked.
- `REJECTED` for missing/ambiguous fields, NaN/Inf, invalid OHLC, negative
  volume, stale/future timestamps, data gaps under strict policy, feature
  schema mismatch, future data availability, unapproved licence, or blocked
  rate-limit state.
- Each injected fault yields its stable reason code; the gate never repairs or
  interpolates OHLCV silently.
- Connectivity failover uses only fake probes/activation callbacks, reports
  route reason codes/latency, never probes an unallowlisted route, and fails
  closed when no route succeeds.

## 4. Optional local Freqtrade smoke (still no exchange connection)

Only if the pinned runtime is already installed in this isolated environment:

```powershell
freqtrade --version
python -c "import json; from pathlib import Path; from freqtrade.strategy import IStrategy; from user_data.strategies.AliXStrategy import AliXStrategy; c=json.loads(Path('config.json').read_text(encoding='utf-8')); assert issubclass(AliXStrategy, IStrategy); assert c['dry_run'] is True and c['initial_state'] == 'stopped' and c['max_open_trades'] == 0; assert c['exchange']['key'] == '' and c['exchange']['secret'] == '' and c['exchange']['pair_whitelist'] == []; print('Freqtrade smoke passed; no exchange connection requested.')"
```

Do not use `trade`, `create-order`, `cancel-order`, `download-data`, or any
command that contacts the exchange during this Phase 3 smoke run.

## 5. Separately authorized local connectivity tests

The automated checks do not prove DNS/TLS, provider authentication, egress IP,
VPN, proxy or SSH tunnel compatibility. Those checks require an owner-approved
test plan and a target environment.

Before any such test, record the approved endpoint/provider, route ID and type,
credential reference (never the value), expected egress IP/allowlist status,
timeout, rollback, test window and owner authorization. Begin with read-only
health checks. Do not activate or change the host networking stack until route
policy, secrets, and rollback have been reviewed. A failed or unknown order
result must be reconciled before any route switch or retry.

## 6. Evidence to record

For each local run, store only:

- UTC start/end time and operator/environment identifier.
- Branch and full Git commit SHA.
- Python/tool/runtime versions.
- Exact command, exit code and sanitized test summary.
- Coverage percentage and quality-gate reason-code counts.
- Connectivity route IDs, outcome reason codes and latency summaries, without
  raw exception strings, URLs containing credentials, tokens or private IPs
  unless explicitly classified for the internal incident record.
- Any failed assertion, issue reference, rollback result and go/no-go decision.

## 7. Stop conditions

Stop and record **NO-GO** if a secret appears in output/artifacts; a test attempts
a real network write; configuration is not safely stopped/dry-run; a schema or
quality failure lacks a reason code; an unapproved provider/licence is allowed;
or a route failure results in fail-open behavior. Do not silence the relevant
test, relax the coverage threshold, interpolate gaps, or infer real-market
compatibility from fixture success.
