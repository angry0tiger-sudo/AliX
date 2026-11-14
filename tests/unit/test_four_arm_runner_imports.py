"""Regression tests for the standard-library-only lock bootstrap."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_runner_and_lock_validator_import_without_third_party_modules() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    probe = """
import sys
import scripts.run_four_arm_experiment
from research.four_arm_core import lock_validator
blocked = {
    "accelerate", "chronos", "einops", "huggingface_hub", "model",
    "numpy", "pandas", "safetensors", "torch", "tqdm", "transformers",
}
loaded = sorted(name for name in sys.modules if name.split(".", 1)[0] in blocked)
if loaded:
    raise SystemExit("third_party_imported_during_bootstrap:" + ",".join(loaded))
print("FOUR_ARM_BOOTSTRAP_IMPORTS=STDLIB_ONLY")
"""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(repository_root),
    }
    result = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FOUR_ARM_BOOTSTRAP_IMPORTS=STDLIB_ONLY" in result.stdout
