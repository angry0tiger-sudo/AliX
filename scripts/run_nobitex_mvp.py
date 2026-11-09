from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.nobitex.mvp import MvpConfig, load_env_file, run_paper_simulation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AliX Nobitex MVP against deterministic simulated data."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--simulate-hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    values = load_env_file(args.env_file)
    config = MvpConfig.from_environment(values)
    report = run_paper_simulation(config, hours=args.simulate_hours, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
