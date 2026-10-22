# Local System Test Matrix (Phase 3)

This matrix is a prepared checklist, not proof that the owner machine has been tested.

| Area | Safe automated check | Owner-local check | Evidence required |
|---|---|---|---|
| Clean install | CI lock/resolver checks | New Python 3.12 venv, pinned lock | Python/tool versions and commit SHA |
| Source quality | Compile, Ruff, Mypy, Bandit, import-linter, Xenon | Same commands on target OS | Exit code per command |
| Data Quality Gate | Synthetic OHLCV, feature, external-record and model-output fixtures | Run focused and full test suites | Test summary, coverage, reason-code assertions |
| Time alignment | Epoch and timezone-aware fixtures | Compare local pipeline's timestamp unit/timezone | UTC examples and schema version |
| Data gaps | Strict reject and explicit degraded-mode tests | Confirm ingestion/backtest/runtime use same policy | Gap interval, policy, status and reason code |
| External provider governance | Unapproved licence and rate-limited fixtures | Review official terms/schema/limits in approved environment | Provider ID, policy status and review time |
| Connectivity | Fake probes/activation callbacks, duplicate route-ID fail-closed test | DNS/TLS/egress/read-only health only after authorization | Route ID, safe reason, latency, sanitized result |
| Freqtrade | CLI/strategy safe-config smoke without exchange | Optional local version/config smoke only | Version and safe-config assertions |
| Network side effects | CI checks use injected callbacks | Separate approved plan required for any real route activation | Authorization, rollback and explicit result |
| Recovery | Unit-level fail-closed behavior | Local restart/timeout test without order writes | State transitions, reconciliation checkpoint, no duplicate retry |
| Secrets | Static and test safeguards | Confirm secret manager/local env is not captured | Redacted checklist; never include secret values |

## Explicit NO-GO conditions

Unexpected real network activity, an unapproved provider/licence, future-data
leakage, an unreasoned rejection, an unallowlisted route probe, raw secret text
in logs, or fail-open behavior invalidates the run. Preserve evidence and stop.
