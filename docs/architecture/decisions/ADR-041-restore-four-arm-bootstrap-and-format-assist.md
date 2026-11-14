# ADR-041: Restore the Proven Four-Arm Bootstrap and Separate Formatting Failures

- **Status:** Accepted
- **Date:** 2026-11-14 UTC
- **Branch:** research/nobitex-hf-strategy
- **Supersedes:** The incorrect stop-loss interpretation recorded in ADR-040; ADR-040 remains immutable historical evidence.
- **Owner directive:** OWNER-2026-11-14-FOURARM

## Context and verified evidence

Attempt 3 was commit `3c9c7562ee90129e9b847a33683b54a6f03ba458`. Its Preflight #14 artifact (`14115004641`, run `43598104335`) contains seven tool results:

| Tool | Exit code | Result |
|---|---:|---|
| Ruff check | 0 | PASS |
| Ruff format | 1 | FORMAT-ONLY |
| mypy | 0 | PASS |
| Bandit | 0 | PASS |
| Xenon | 0 | PASS |
| pip-audit | 0 | PASS |
| pytest | 0 | PASS |

Pytest reported 981 passed and 5 skipped. The only non-zero tool exit was Ruff format, with one formatting-only diff in `research/four_arm_core/lock_validator.py`.

Standard CI #2407 (run `43598104306`) independently reported `four-arm-lock-bare-env=success`; the bare-environment lock-only validation was therefore already passing on attempt 3. The run's quality job failed on the formatting gate, not on the lock validator, imports, or bare-environment path.

The previous STOP-LOSS implementation incorrectly counted this formatting-only outcome as a substantive failed corrective attempt and reverted the proven architecture. That classification is corrected by this ADR.

## Decision

1. Restore the exact source tree from attempt 3 as a new ordinary commit whose parent is the previous rollback commit. Do not force-push.
2. A Ruff-format-only failure is not a substantive corrective attempt and does not consume the maximum of three attempts.
3. Before any subsequent Python commit, run the artifact-only `format-assist` workflow, download its `ruff-formatted-<run_id>` artifact, and commit the returned file bytes without hand-editing them.
4. The formatter workflow is pinned to the project's locked `ruff==0.14.0`, runs `ruff format .`, has read-only repository permissions, and uploads only Python files changed by Ruff. It never commits or pushes.
5. The maximum of three corrective attempts applies only to substantive failures: mypy, Bandit, Xenon, pytest, runtime/bootstrap logic, or other behavioral defects. After three such failures, perform a surgical rollback and stop.
6. Formatting assistance does not relax or bypass any gate. Standard CI, the complete seven-tool Preflight, and `four-arm-lock-bare-env` must pass on the exact commit before execution may be authorized.
7. A runtime failure in the real experiment is an immediate stop without automatic retry.

## Safety invariants

- `main` remains frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
- The public fetcher, Spec v3, `HOLDOUT_COMMITMENT_V2.md`, and `data/nobitex/prehack/*` remain unchanged.
- The execution lock remains `NOT_AUTHORIZED` until the final exact-SHA preflight and CI gates pass and the execution workflow can be safely dispatched.
- Any eventual experiment is restricted to the immutable pre-hack dataset and the twelve registered cells. No post-hack holdout access, prices, or OHLCV values may enter logs.
- Dry-run remains true; live trading remains false; no private API or order path is permitted.

## Consequences

Attempt 3's stdlib bootstrap architecture and bare-env CI protection are restored. The format-assist artifact creates a reproducible handoff between formatting and commits, while all substantive quality and safety gates remain mandatory. No economic outcome is implied by a passing bootstrap or quality gate.
