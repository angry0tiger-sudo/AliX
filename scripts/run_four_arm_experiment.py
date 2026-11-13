"""Fail-closed four-arm Kronos/Chronos experiment orchestrator.

Mock mode is deterministic and synthetic-only. Research mode requires an immutable
execution lock and pinned executor adapter; no experimental choices are guessed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, cast

PREHACK_SHA256 = "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
EXPECTED_ROWS = 31_801
ARM_IDS = ("A", "B", "C", "D")
REQUIRED_LOCK_PATHS = (
    ("models", "kronos", "repository"),
    ("models", "kronos", "revision"),
    ("models", "kronos", "weights_sha256"),
    ("models", "chronos", "repository"),
    ("models", "chronos", "revision"),
    ("models", "chronos", "weights_sha256"),
    ("forecast_mapping", "arm_b"),
    ("forecast_mapping", "arm_c"),
    ("forecast_mapping", "arm_d"),
    ("folds", "boundaries"),
    ("labels", "price_anchor"),
    ("labels", "barrier_fill_convention"),
    ("labels", "collision_rule"),
    ("cost_model", "revision"),
    ("dsr", "method"),
    ("dsr", "implementation_revision"),
    ("dsr", "global_n_sensitivity"),
    ("stability_rule",),
    ("executor", "module"),
    ("executor", "callable"),
    ("executor", "source_sha256"),
)


class ExperimentError(RuntimeError):
    """An experiment invariant or precondition was not satisfied."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"json_read_failed:{path}:{type(exc).__name__}") from None
    if not isinstance(payload, dict):
        raise ExperimentError(f"json_root_must_be_object:{path}")
    return payload


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ExperimentError(f"file_read_failed:{path}:{type(exc).__name__}") from None
    return digest.hexdigest()


def _append_log(output_dir: Path, event: str, **fields: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"event": event, **fields}
    with (output_dir / "run-log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def load_contract(config_path: Path, preregistration_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    try:
        preregistration = preregistration_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentError(
            f"preregistration_read_failed:{type(exc).__name__}"
        ) from None
    match = re.search(r"CSV SHA-256[^\x60]*\x60([0-9a-f]{64})\x60", preregistration)
    configured_sha = config.get("pre_hack_sha")
    if (
        match is None
        or configured_sha != match.group(1)
        or configured_sha != PREHACK_SHA256
    ):
        raise ExperimentError("pre_hack_sha_mismatch_against_preregistration")
    if config.get("arms") != list(ARM_IDS):
        raise ExperimentError("four_arm_configuration_must_be_exactly_A_B_C_D")
    if config.get("cost_scenarios_bps") != [50, 90]:
        raise ExperimentError("cost_scenarios_must_be_50_and_90_bps")
    if config.get("gates") != {
        "netPF_min": 1.10,
        "trades_min": 100,
        "maxDD_max": 0.15,
        "dsr_min": 0,
        "min_stable_folds": 3,
    }:
        raise ExperimentError("economic_gates_differ_from_preregistration")
    return config


def _fixture_returns(arm_index: int, fold_index: int, count: int = 40) -> list[float]:
    """Deterministic synthetic trade returns for runner mechanics only."""
    bias_bps = (arm_index - 1) * 1.5
    values: list[float] = []
    for trade_index in range(count):
        cycle = ((trade_index * 7 + fold_index * 5 + arm_index * 3) % 19) - 9
        values.append(((cycle * 9.0) + bias_bps) / 10_000.0)
    return values


def _profit_factor(returns: list[float]) -> float | None:
    gains = sum(value for value in returns if value > 0.0)
    losses = -sum(value for value in returns if value < 0.0)
    return gains / losses if losses > 0.0 else None


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0.0:
            maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _metrics(gross_returns: list[float]) -> dict[str, float | None]:
    net_50 = [value - 0.005 for value in gross_returns]
    net_90 = [value - 0.009 for value in gross_returns]
    return {
        "netPF_50bps": _profit_factor(net_50),
        "netPF_90bps": _profit_factor(net_90),
        "maxDD": _max_drawdown(net_90),
    }


def _write_synthetic_ohlcv(output_dir: Path, row_count: int = 1200) -> dict[str, Any]:
    """Write a deterministic synthetic OHLCV fixture, never downloaded market data."""
    path = output_dir / "synthetic-nobitex-btcusdt-1h.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("date", "open", "high", "low", "close", "volume")
        )
        writer.writeheader()
        previous_close = 100.0
        for index in range(row_count):
            timestamp = 1_600_000_000 + index * 3600
            close = 100.0 + (0.005 * index) + (0.75 * math.sin(index / 17.0))
            opened = previous_close
            writer.writerow(
                {
                    "date": datetime.fromtimestamp(timestamp, timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "open": f"{opened:.8f}",
                    "high": f"{max(opened, close) + 0.2:.8f}",
                    "low": f"{min(opened, close) - 0.2:.8f}",
                    "close": f"{close:.8f}",
                    "volume": f"{1.0 + (index % 10) / 10.0:.8f}",
                }
            )
            previous_close = close
    return {
        "kind": "synthetic",
        "row_count": row_count,
        "sha256": _sha256_file(path),
        "file": path.name,
        "market_data_read": False,
    }


def run_mock(
    config: dict[str, Any], output_dir: Path, commit_sha: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dataset = _write_synthetic_ohlcv(output_dir)
    arm_manifests: list[dict[str, Any]] = []
    for arm_index, arm_id in enumerate(ARM_IDS):
        folds: list[dict[str, Any]] = []
        pooled: list[float] = []
        for fold_index in range(3):
            gross = _fixture_returns(arm_index, fold_index)
            pooled.extend(gross)
            folds.append(
                {
                    "fold": fold_index + 1,
                    "trades": len(gross),
                    **_metrics(gross),
                    "DSR": None,
                    "evidence_type": "SYNTHETIC_FIXTURE_ONLY",
                }
            )
        manifest = {
            "schema_version": "alix.four-arm-experiment.manifest.v1",
            "mode": "MOCK_ONLY",
            "evaluation_status": "SYNTHETIC_MECHANICS_ONLY",
            "commit_sha": commit_sha,
            "arm_id": arm_id,
            "trial_id": f"NOBITEX-KC26-{arm_id}",
            "pre_hack_sha": config["pre_hack_sha"],
            "dataset": synthetic_dataset,
            **_metrics(pooled),
            "trades": len(pooled),
            "DSR": None,
            "fold_results": folds,
            "winner": False,
            "gates_evaluated": False,
            "economic_evaluation_performed": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "safety": {
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "private_api_calls": 0,
                "real_order_attempts": 0,
            },
        }
        path = output_dir / f"arm-{arm_id}.manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        arm_manifests.append(manifest)
        _append_log(
            output_dir,
            "mock_arm_manifest_written",
            arm_id=arm_id,
            manifest=path.name,
            market_data_read=False,
        )
    summary = {
        "schema_version": "alix.four-arm-experiment.run.v1",
        "mode": "MOCK_ONLY",
        "evaluation_status": "SYNTHETIC_MECHANICS_ONLY",
        "commit_sha": commit_sha,
        "arms": [item["arm_id"] for item in arm_manifests],
        "winner": None,
        "economic_verdict": "NOT_EVALUATED",
        "economic_evaluation_performed": False,
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }
    (output_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _append_log(output_dir, "mock_run_completed", arms=list(ARM_IDS), winner=None)
    return summary


def _validate_lock_fields(lock: dict[str, Any]) -> None:
    missing = [
        ".".join(path)
        for path in REQUIRED_LOCK_PATHS
        if _nested_value(lock, path) in (None, "", [], {})
    ]
    if missing:
        raise ExperimentError("execution_lock_incomplete:" + ",".join(missing))
    if lock.get("status") != "LOCKED":
        raise ExperimentError("execution_lock_status_must_be_LOCKED")
    if _nested_value(lock, ("stability_rule",)) != (
        "all_three_folds_pass_both_cost_scenarios"
    ):
        raise ExperimentError("unsupported_fold_stability_rule")


def _validate_locked_executor(lock: dict[str, Any]) -> tuple[Path, str]:
    module_name = cast(str, _nested_value(lock, ("executor", "module")))
    callable_name = cast(str, _nested_value(lock, ("executor", "callable")))
    if (
        not module_name.startswith("research.")
        or not callable_name.isidentifier()
        or ".." in module_name
    ):
        raise ExperimentError("executor_reference_out_of_policy")
    module_path = Path(*module_name.split(".")).with_suffix(".py")
    expected_sha = cast(str, _nested_value(lock, ("executor", "source_sha256")))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ExperimentError("executor_source_sha256_invalid")
    if not module_path.is_file() or _sha256_file(module_path) != expected_sha:
        raise ExperimentError("executor_source_missing_or_sha256_mismatch")
    return module_path, module_name


def validate_execution_lock(lock_path: Path) -> tuple[dict[str, Any], Path, str]:
    if not lock_path.is_file():
        raise ExperimentError("execution_lock_missing:research_execution_blocked")
    lock = _read_json(lock_path)
    _validate_lock_fields(lock)
    module_path, module_name = _validate_locked_executor(lock)
    return lock, module_path, module_name


def _require_manifest_value(
    manifest: dict[str, Any], key: str, expected: Any, error_code: str
) -> None:
    if manifest.get(key) != expected:
        raise ExperimentError(error_code)


def _verify_prehack_manifest(
    manifest: dict[str, Any], digest: str, config: dict[str, Any]
) -> None:
    if digest != config["pre_hack_sha"] or digest != PREHACK_SHA256:
        raise ExperimentError("pre_hack_dataset_sha256_mismatch")
    _require_manifest_value(
        manifest, "csv_sha256", digest, "pre_hack_manifest_sha256_mismatch"
    )
    _require_manifest_value(
        manifest,
        "row_count",
        EXPECTED_ROWS,
        "pre_hack_manifest_row_count_mismatch",
    )
    _require_manifest_value(
        manifest,
        "dataset_window",
        "PRE_HACK_NOBITEX_ONLY",
        "dataset_window_must_be_PRE_HACK_NOBITEX_ONLY",
    )
    _require_manifest_value(
        manifest,
        "source_policy",
        "NOBITEX_ONLY_NO_BACKFILL",
        "pre_hack_source_policy_invalid",
    )


def _count_prehack_csv_rows(ohlcv_path: Path) -> int:
    required = {"date", "open", "high", "low", "close", "volume"}
    with ohlcv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ExperimentError("pre_hack_csv_schema_invalid")
        return sum(1 for _ in reader)


def _verify_prehack_csv(ohlcv_path: Path) -> int:
    row_count = _count_prehack_csv_rows(ohlcv_path)
    if row_count != EXPECTED_ROWS:
        raise ExperimentError("pre_hack_csv_row_count_mismatch")
    return row_count


def _verify_prehack_dataset(
    ohlcv_path: Path, manifest_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    digest = _sha256_file(ohlcv_path)
    _verify_prehack_manifest(manifest, digest, config)
    row_count = _verify_prehack_csv(ohlcv_path)
    return {
        "kind": "PRE_HACK_NOBITEX_ONLY",
        "sha256": digest,
        "row_count": row_count,
        "manifest_commit_sha": manifest.get("commit_sha"),
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _arm_result_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_arms = result.get("arms")
    if not isinstance(raw_arms, list):
        raise ExperimentError("executor_result_arms_missing")
    by_id: dict[str, dict[str, Any]] = {}
    for raw_item in raw_arms:
        if not isinstance(raw_item, dict):
            continue
        arm_id = raw_item.get("arm_id")
        if isinstance(arm_id, str):
            by_id[arm_id] = raw_item
    if set(by_id) != set(ARM_IDS) or len(raw_arms) != 4:
        raise ExperimentError("executor_must_return_exactly_four_unique_arms")
    return by_id


def _validate_arm_identity_and_metrics(
    arm_id: str, item: dict[str, Any]
) -> None:
    if item.get("trial_id") != f"NOBITEX-KC26-{arm_id}":
        raise ExperimentError(f"trial_id_mismatch:{arm_id}")
    for key in ("netPF_50bps", "netPF_90bps", "maxDD", "DSR"):
        if not _finite_number(item.get(key)):
            raise ExperimentError(f"arm_metric_missing_or_nonfinite:{arm_id}:{key}")


def _stable_fold_count(folds: list[Any]) -> int:
    stable_count = 0
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        net_50 = fold.get("netPF_50bps")
        net_90 = fold.get("netPF_90bps")
        if (
            _finite_number(net_50)
            and _finite_number(net_90)
            and net_50 >= 1.0
            and net_90 >= 1.0
        ):
            stable_count += 1
    return stable_count


def _dsr_sensitivity_passes(item: dict[str, Any], dsr_min: float) -> bool:
    evidence = item.get("dsr_evidence")
    sensitivity = (
        evidence.get("unresolved_remainder_worst_case_dsr")
        if isinstance(evidence, dict)
        else None
    )
    return (
        item["DSR"] > dsr_min
        and sensitivity is not None
        and _finite_number(sensitivity)
        and sensitivity > 0
    )


def _arm_gate_results(
    item: dict[str, Any], config: dict[str, Any], stable_folds: int
) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "netPF_50bps": item["netPF_50bps"] >= gates["netPF_min"],
        "netPF_90bps": item["netPF_90bps"] >= gates["netPF_min"],
        "trades": item.get("trades", 0) >= gates["trades_min"],
        "maxDD": item["maxDD"] <= gates["maxDD_max"],
        "dsr": _dsr_sensitivity_passes(item, gates["dsr_min"]),
        "stability": stable_folds >= gates["min_stable_folds"],
    }


def _evaluate_arm(
    arm_id: str,
    raw_item: dict[str, Any],
    config: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    item = dict(raw_item)
    _validate_arm_identity_and_metrics(arm_id, item)
    folds = item.get("fold_results")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ExperimentError(f"exactly_three_fold_results_required:{arm_id}")
    stable_folds = _stable_fold_count(folds)
    gates = _arm_gate_results(item, config, stable_folds)
    item["gate_results"] = gates
    item["stable_folds"] = stable_folds
    item["winner"] = all(gates.values())
    item["pre_hack_sha"] = dataset["sha256"]
    item["dataset"] = dataset
    return item


def _write_arm_manifest(
    output_dir: Path, item: dict[str, Any]
) -> None:
    arm_id = item["arm_id"]
    path = output_dir / f"arm-{arm_id}.manifest.json"
    path.write_text(
        json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _append_log(
        output_dir,
        "research_arm_evaluated",
        arm_id=arm_id,
        winner=item["winner"],
        gate_results=item["gate_results"],
    )


def _select_winner(evaluated: list[dict[str, Any]]) -> str | None:
    passing = [item for item in evaluated if item["winner"]]
    if not passing:
        return None
    return max(passing, key=lambda item: item["DSR"])["arm_id"]


def _write_research_summary(
    output_dir: Path, evaluated: list[dict[str, Any]], winner: str | None
) -> dict[str, Any]:
    summary = {
        "schema_version": "alix.four-arm-experiment.run.v1",
        "mode": "RESEARCH",
        "evaluation_status": "COMPLETED",
        "arms": evaluated,
        "winner": winner,
        "economic_verdict": "PASS" if winner is not None else "NO-GO",
        "economic_evaluation_performed": True,
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }
    (output_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _append_log(output_dir, "research_run_completed", winner=winner)
    return summary


def validate_research_result(
    result: dict[str, Any],
    config: dict[str, Any],
    dataset: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    by_id = _arm_result_map(result)
    evaluated = [
        _evaluate_arm(arm_id, by_id[arm_id], config, dataset)
        for arm_id in ARM_IDS
    ]
    for item in evaluated:
        _write_arm_manifest(output_dir, item)
    winner = _select_winner(evaluated)
    return _write_research_summary(output_dir, evaluated, winner)


def run_research(
    *,
    config: dict[str, Any],
    execution_lock: Path,
    ohlcv_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    # Lock and executor are checked before opening the market-data files.
    lock, _module_path, module_name = validate_execution_lock(execution_lock)
    dataset = _verify_prehack_dataset(ohlcv_path, manifest_path, config)
    module = importlib.import_module(module_name)
    executor_name = cast(str, _nested_value(lock, ("executor", "callable")))
    executor = getattr(module, executor_name, None)
    if not callable(executor):
        raise ExperimentError("locked_executor_callable_missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    _append_log(
        output_dir,
        "research_execution_started",
        pre_hack_sha=dataset["sha256"],
        row_count=dataset["row_count"],
    )
    result = executor(
        {
            "config": config,
            "execution_lock": lock,
            "ohlcv_path": str(ohlcv_path),
            "dataset_manifest_path": str(manifest_path),
            "dataset": dataset,
            "output_dir": str(output_dir),
        }
    )
    if not isinstance(result, dict):
        raise ExperimentError("executor_result_must_be_object")
    return validate_research_result(result, config, dataset, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("mock", "research", "preflight"), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment/four_arm_kronos_chronos.json"),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("docs/research/preregistration/FOUR_ARM_KRONOS_CHRONOS.md"),
    )
    parser.add_argument(
        "--execution-lock",
        type=Path,
        default=Path("config/experiment/four_arm_execution_lock.json"),
    )
    parser.add_argument("--ohlcv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/four-arm-experiment")
    )
    parser.add_argument("--commit-sha", default="UNKNOWN")
    args = parser.parse_args()
    try:
        config = load_contract(args.config, args.preregistration)
        if args.mode == "mock":
            summary = run_mock(config, args.output_dir, args.commit_sha)
        elif args.mode == "preflight":
            lock, module_path, module_name = validate_execution_lock(
                args.execution_lock
            )
            summary = {
                "mode": "PREFLIGHT",
                "status": "READY",
                "executor_module": module_name,
                "executor_path": str(module_path),
                "lock_status": lock["status"],
                "dataset_bytes_read": False,
                "economic_evaluation_performed": False,
            }
            _append_log(args.output_dir, "preflight_ready", executor_module=module_name)
        else:
            if args.ohlcv is None or args.manifest is None:
                raise ExperimentError("research_mode_requires_ohlcv_and_manifest")
            summary = run_research(
                config=config,
                execution_lock=args.execution_lock,
                ohlcv_path=args.ohlcv,
                manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except (ExperimentError, OSError, ImportError, AttributeError) as exc:
        message = str(exc) or type(exc).__name__
        print(json.dumps({"status": "BLOCKED", "reason": message}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
