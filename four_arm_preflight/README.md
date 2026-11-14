# Four-Arm Quality Preflight

This directory anchors the `four_arm*/**` push path filter for
`.github/workflows/quality-preflight.yml`. The workflow is diagnostic only:
it runs alongside, and never replaces or weakens, Standard CI.

The preflight retains the complete stdout/stderr of Ruff lint, Ruff format,
strict mypy, Bandit, Xenon, pip-audit, and pytest in a single JSON report and
individual log files. Each diagnostic is allowed to fail without preventing
the remaining diagnostics from running.

Pytest uses `--maxfail=0` to collect all failures. The requested spelling
`-x=false` is not a valid pytest option: `-x` is a boolean stop-after-first-failure
switch, so `--maxfail=0` is the correct explicit no-early-abort equivalent.
