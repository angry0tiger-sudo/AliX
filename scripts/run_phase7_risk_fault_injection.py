#!/usr/bin/env python3
"""Run isolated Phase 7 fault tests and write a per-scenario audit artifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


_STATUS = re.compile(r"\b(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")


def _scenario_results(output: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for line in output.splitlines():
        if "::test_" not in line:
            continue
        match = _STATUS.search(line)
        if match is None:
            continue
        results.append(
            {
                "scenario": line.split("::")[-1].split()[0],
                "status": match.group(1),
            }
        )
    return results


def _report(output: str, return_code: int, commit_sha: str | None) -> dict[str, object]:
    scenarios = _scenario_results(output)
    return {
        "schema_version": "phase7-fault-injection-report.v1",
        "execution_mode": "synthetic_unit_tests_only",
        "network_calls": 0,
        "orders_submitted": 0,
        "trading_authorization": "NO_GO",
        "commit_sha": commit_sha or "UNAVAILABLE",
        "suite_return_code": return_code,
        "suite_status": "PASS" if return_code == 0 else "FAIL",
        "scenario_count": len(scenarios),
        "scenario_pass_count": sum(item["status"] == "PASSED" for item in scenarios),
        "scenarios": scenarios,
        "raw_pytest_output": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--commit-sha", default=None)
    args = parser.parse_args()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "--color=no",
            "tests/unit/test_risk_engine.py",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    output = completed.stdout + "\n" + completed.stderr
    result = _report(output, completed.returncode, args.commit_sha)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "raw_pytest_output"},
            indent=2,
        )
    )
    if completed.returncode != 0 or result["scenario_count"] == 0:
        return completed.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
