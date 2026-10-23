# Phase 10 Runbook: GitHub Actions and Later Windows Handover

## Safety boundary
Prepared only. This runbook has not been executed on Windows/WSL2. The public workflows are manual-only and contain no secrets. They do not invoke order endpoints, start Freqtrade, deliver Bale messages, or change the trading authorization state.

## Public historical research in GitHub Actions
After the owner-authorized candidate is available on a ref from which the workflow can be manually dispatched:
1. Open **Actions → AliX Phase 10 Public Historical Market Research → Run workflow**.
2. Select 365 days unless an explicit shorter research window is intended.
3. Inspect the artifact named `alix-phase10-public-market-research-<run_id>`.
4. Verify manifest commit SHA, row count, time range, CSV SHA-256, OHLCV gap policy, model split configuration, fee/slippage assumptions, sample count, win rate, net expectancy and maximum drawdown.
5. The initial research cost assumptions are 25 bps per side fees and 5 bps per side slippage. Public candles do not disclose the owner's actual fee tier. These assumptions are not an assertion that the actual charge is exactly 0.25%.
6. Threshold evaluation is research-only: at least 100 trades, win rate >=60%, max drawdown <=15% in absolute value, and positive net expectancy. Insufficient sample or any failed threshold is NO-GO.
7. The workflow must not be auto-run on push/PR; it uses only `workflow_dispatch`. Downloaded data remains an Actions artifact, not a repository commit.

## Optional public-price observer
Manual Actions dispatch only. Choose 5, 15 or 60 minutes. Every record must contain `decision=ABSTAIN`, `orders_submitted=0`, `credentials_used=false`. This is public price telemetry, not Paper trading.

## Later owner-run Windows validation (not executed)
1. Check out the separately owner-approved release SHA; compare it with the SHA in its release record before any test.
2. Verify Python 3.12 and install the committed locked dependencies using the reviewed project install procedure. Do not copy a real .env file into test paths.
3. Run tests only against synthetic fixtures and disposable storage; never mount a personal account or live Freqtrade user_data database.
4. Run Unit/Integration tests and `python -m scripts.run_phase9_e2e --output <disposable-path> --commit-sha <exact-approved-sha>`; capture exact exit status and report hashes.
5. Run a one-shot report with an aware timestamp, dedicated test DB and temporary output directory, e.g. `python scripts/run_phase10_reporting_once.py --once --scheduled-for 2026-10-23T07:00:00+03:30 --database-url sqlite:///runtime/phase10-test/scheduler.sqlite3 --output-dir runtime/phase10-test/reports --commit-sha <exact-approved-sha>`. Replace the sample date only as needed for testing; do not pass production DB paths.
6. Inspect JSON/text/PNG and confirm the report says DEGRADED and NOT_RECONCILED when account/decision sources are missing. Certified PnL/equity/drawdown should remain null.
7. For PostgreSQL integration, use only `compose.phase9-test.yaml` with the dedicated `phase9-test` profile. Do not alter production `compose.yaml` network topology or remove its volumes.
8. Inspect logs/artifacts for secrets. No Bale token or recipient is required for reporting unit tests.
9. Stop the test process normally and remove only the dedicated disposable test directory after verifying its exact absolute path. Never use broad Docker volume pruning or production volume cleanup commands.

## Evidence classification
- Base main exact-head CI: verified successful on the Phase 9 merge SHA (not Phase 10).
- Phase 10 candidate CI: pending until exact candidate run completes.
- Scheduler/report source logic: must be proven through candidate CI.
- Actual public historical dataset and economic metrics: pending manual Actions dispatch; not yet fetched.
- Real account/order reconciliation, production PostgreSQL roles/backup, owner host ACLs and Windows/WSL2: owner-run later, not tested.
- Orders, trading sessions and Bale delivery: prohibited in this phase.
