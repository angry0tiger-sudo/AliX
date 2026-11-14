"""Capture immutable SHA-256 fingerprints for the four-arm execution contracts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

CONTRACTS: dict[str, str] = {
    "kronos_adapter_sha256": "docs/research/preregistration/contracts/kronos_adapter.md",
    "chronos_adapter_sha256": "docs/research/preregistration/contracts/chronos_adapter.md",
    "combined_rule_sha256": "docs/research/preregistration/contracts/combined_rule.md",
    "fold_plan_sha256": "docs/research/preregistration/contracts/fold_plan.md",
    "triple_barrier_sha256": "docs/research/preregistration/contracts/triple_barrier.md",
    "cost_fill_simulator_sha256": "docs/research/preregistration/contracts/cost_fill_simulator.md",
    "dsr_method_sha256": "docs/research/preregistration/contracts/dsr_method.md",
    "license_and_corpus_disclosure_sha256": "docs/research/preregistration/contracts/model_provenance.md",
    "runtime_lock_sha256": "docs/research/preregistration/contracts/runtime_lock.md",
}


def capture_contract_hashes(repository_root: Path) -> dict[str, Any]:
    fingerprints: dict[str, str] = {}
    for key, relative_path in CONTRACTS.items():
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"execution_contract_file_missing:{relative_path}")
        fingerprints[key] = sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "alix.four-arm.execution-contract-fingerprints.v1",
        "contract_count": len(fingerprints),
        "contracts": fingerprints,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(__import__("os").environ["RUNNER_TEMP"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = capture_contract_hashes(root)
    output_path = output_dir / "four-arm-contract-fingerprints.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    print("FOUR_ARM_CONTRACT_FINGERPRINT_CAPTURE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
