"""CLI entry point for the offline V4-E hierarchical target audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.ml_core.hierarchical_target_audit import (
    run_hierarchical_target_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()

    report = run_hierarchical_target_audit(
        args.input_dir, days=args.days, commit_sha=args.commit_sha
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "commit_sha": report["commit_sha"],
                "experiment_count": report["experiment_count"],
                "late_hit_transition_count_13_to_24": report[
                    "late_hit_transition_count_13_to_24"
                ],
                "model_training_performed": False,
                "Champion": None,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
