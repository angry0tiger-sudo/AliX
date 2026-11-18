"""Owner-authorized G4.6 pre-hack volume-feature ablation with offline ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from research.regime_r.ngboost_cqr.adapters import HGBQCQRAdapter
from scripts.run_regime_r_trials import (
    COSTS_BPS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_DATA_SHA256,
    FEATURE_COLUMNS,
    _validate_requested_commit_sha,
    aggregate_fold_ledgers,
    build_features,
    build_fold_samples,
    build_folds,
    load_and_verify_inputs,
    require_owner_authorization,
    simulate_costed_ledger,
    summarize_forecast_diagnostics,
    trial_identity,
)
from research.regime_r.ngboost_cqr.contracts import TRIAL_IDS

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = ROOT / "config/regime_r/hgbq_cqr_preregistration_ablation.json"
EXPECTED_PREREGISTRATION_SHA256 = (
    "83d9dffc0e3332eaddd5ee35bf846b13b049039eda6df087b9ccb734ce9f9de1"
)
EXPECTED_FEATURE_CONTRACT_SHA256 = (
    "6bcd91dcfdd3e48cfbe64adb714f432ae8dc610f136ed2731967c9aed5282250"
)
VOLUME_FEATURE_COLUMNS = ("log_volume_change_1", "volume_zscore_24")
RETAINED_FEATURE_COLUMNS = (
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "realized_volatility_6",
    "realized_volatility_24",
    "range_fraction_1",
    "atr_fraction_14",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
OUTPUT_SCHEMA = "alix.regime-r.g4.6-volume-ablation.v1"


def _validate_g46_registration_identity(payload: dict[str, Any]) -> None:
    prereg = cast(dict[str, Any], payload.get("preregistration", {}))
    dataset = cast(dict[str, Any], payload.get("dataset", {}))
    frozen_g3 = cast(dict[str, Any], payload.get("frozen_g3", {}))
    if prereg.get("experiment_id") != "G4.6":
        raise RuntimeError("regime_r_g46_preregistration_identity_invalid")
    if prereg.get("status") != "PREREGISTERED_LOCKED":
        raise RuntimeError("regime_r_g46_preregistration_not_locked")
    if dataset.get("sha256") != EXPECTED_DATA_SHA256:
        raise RuntimeError("regime_r_g46_dataset_commitment_mismatch")
    if dataset.get("confirmatory_holdout") is not False:
        raise RuntimeError("regime_r_g46_must_be_exploratory_prehack_only")
    if frozen_g3.get("sha256") != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("regime_r_g46_frozen_g3_commitment_mismatch")
    if frozen_g3.get("must_remain_byte_identical") is not True:
        raise RuntimeError("regime_r_g46_g3_must_remain_locked")


def _validate_g46_feature_registration(feature: dict[str, Any]) -> None:
    if feature.get("profile") != "VOLUME_DERIVED_FEATURES_EXCLUDED":
        raise RuntimeError("regime_r_g46_feature_profile_invalid")
    if tuple(feature.get("removed_predictors", ())) != VOLUME_FEATURE_COLUMNS:
        raise RuntimeError("regime_r_g46_removed_feature_set_invalid")
    if tuple(feature.get("retained_predictors", ())) != RETAINED_FEATURE_COLUMNS:
        raise RuntimeError("regime_r_g46_retained_feature_set_invalid")
    contract = feature.get("feature_contract")
    if not isinstance(contract, list) or not all(
        isinstance(line, str) for line in contract
    ):
        raise RuntimeError("regime_r_g46_feature_contract_invalid")
    contract_sha = hashlib.sha256("\n".join(contract).encode("utf-8")).hexdigest()
    committed_sha = feature.get("feature_contract_sha256")
    if contract_sha != EXPECTED_FEATURE_CONTRACT_SHA256:
        raise RuntimeError("regime_r_g46_feature_contract_sha256_mismatch")
    if committed_sha != EXPECTED_FEATURE_CONTRACT_SHA256:
        raise RuntimeError("regime_r_g46_feature_contract_sha256_mismatch")


def _validate_g46_acceptance_registration(acceptance: dict[str, Any]) -> None:
    gates = cast(dict[str, Any], acceptance.get("per_cost_gates", {}))
    expected_gates = (
        gates.get("net_profit_factor_min") == 1.10
        and gates.get("trade_count_min") == 100
        and gates.get("max_drawdown_max") == 0.15
        and gates.get("qualifying_lower_bound_folds_min") == 3
        and acceptance.get("cost_scenarios_bps") == [50, 90]
    )
    if not expected_gates:
        raise RuntimeError("regime_r_g46_acceptance_contract_invalid")


def load_g46_preregistration(
    path: Path = PREREGISTRATION_PATH,
) -> dict[str, Any]:
    """Load the byte-locked G4.6 registration and fail closed on drift."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("regime_r_g46_preregistration_sha256_mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("regime_r_g46_preregistration_invalid")
    _validate_g46_registration_identity(payload)
    feature = cast(dict[str, Any], payload.get("feature_ablation", {}))
    acceptance = cast(dict[str, Any], payload.get("acceptance", {}))
    _validate_g46_feature_registration(feature)
    _validate_g46_acceptance_registration(acceptance)
    return payload

def build_volume_ablation_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen baseline feature matrix, then exclude exactly two volume predictors."""
    baseline = build_features(frame)
    if tuple(baseline.columns) != FEATURE_COLUMNS:
        raise RuntimeError("regime_r_g46_baseline_feature_schema_drift")
    if any(column not in baseline.columns for column in VOLUME_FEATURE_COLUMNS):
        raise RuntimeError("regime_r_g46_volume_feature_missing_from_baseline")
    result = baseline.drop(columns=list(VOLUME_FEATURE_COLUMNS)).copy()
    if tuple(result.columns) != RETAINED_FEATURE_COLUMNS:
        raise RuntimeError("regime_r_g46_ablation_feature_schema_invalid")
    if len(result.columns) != 11:
        raise RuntimeError("regime_r_g46_expected_eleven_predictors")
    return result


def evaluate_cost_acceptance(
    pooled_metrics: dict[str, Any],
    fold_records: list[dict[str, Any]],
    cost_bps: int,
) -> dict[str, Any]:
    """Apply the four owner-registered G4.6 gates for one horizon and one cost."""
    if cost_bps not in COSTS_BPS:
        raise ValueError("regime_r_g46_cost_scenario_not_preregistered")
    if len(fold_records) != 3:
        raise RuntimeError("regime_r_g46_requires_exactly_three_folds")
    net_pf = pooled_metrics.get("net_profit_factor")
    max_dd = pooled_metrics.get("max_drawdown")
    trades = int(pooled_metrics.get("trade_count", 0))
    qualifying_folds = sum(
        int(int(item["metrics"]["signal_count_above_lower_bound"]) > 0)
        for item in fold_records
    )
    gates = {
        "net_pf_gte_1_10": net_pf is not None
        and math.isfinite(float(net_pf))
        and float(net_pf) >= 1.10,
        "trades_gte_100": trades >= 100,
        "max_dd_lte_15_percent": max_dd is not None
        and math.isfinite(float(max_dd))
        and float(max_dd) <= 0.15,
        "lower_bound_cost_passed_in_all_three_folds": qualifying_folds >= 3,
    }
    return {
        "cost_bps": cost_bps,
        "qualifying_lower_bound_folds": qualifying_folds,
        "required_qualifying_folds": 3,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def _metrics_without_paths(ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in ledger.items()
        if key not in ("trades", "equity_curve")
    }


def run_ablation_trial(
    trial_id: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    folds: tuple[Any, ...],
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit one registered slot once per fold and evaluate its two offline cost ledgers."""
    horizon, slot = trial_identity(trial_id)
    horizon_bars = int(horizon[:-1])
    target = (
        frame["close"].shift(-horizon_bars) / frame["open"].shift(-1) - 1.0
    ).to_numpy(dtype=np.float64)
    values = features.to_numpy(dtype=np.float64)
    fold_results: dict[str, list[dict[str, Any]]] = {"50": [], "90": []}
    fold_ledgers: dict[str, list[dict[str, Any]]] = {"50": [], "90": []}
    diagnostics: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for fold in folds:
        samples = build_fold_samples(features, target, horizon_bars, fold)
        adapter = HGBQCQRAdapter(horizon=horizon)
        adapter.fit(
            values[samples.train_indices],
            target[samples.train_indices],
            values[samples.calibration_indices],
            target[samples.calibration_indices],
        )
        forecasts, intervals = adapter.predict_batch_intervals(
            values[samples.test_indices]
        )
        lower_bounds = np.asarray(
            [item.lower_bound for item in intervals], dtype=np.float64
        )
        statistics = summarize_forecast_diagnostics(forecasts, intervals)
        diagnostic = {
            "trial_id": trial_id,
            "horizon": horizon,
            "candidate_slot": slot,
            "fold_id": fold.fold_id,
            "train_rows": int(samples.train_indices.size),
            "calibration_rows": int(samples.calibration_indices.size),
            "test_rows": int(samples.test_indices.size),
            "test_start_utc": fold.test_start_utc,
            "test_end_utc": fold.test_end_utc,
            "statistics": statistics,
        }
        diagnostics.append(diagnostic)

        for cost in COSTS_BPS:
            ledger = simulate_costed_ledger(
                frame,
                fold.test_start,
                fold.test_end,
                horizon_bars,
                samples.test_indices,
                lower_bounds,
                cost,
            )
            cost_key = str(cost)
            fold_results[cost_key].append(
                {
                    "fold_id": fold.fold_id,
                    "metrics": _metrics_without_paths(ledger),
                }
            )
            fold_ledgers[cost_key].append(ledger)
            trade_rows.extend(
                {
                    "trial_id": trial_id,
                    "horizon": horizon,
                    "candidate_slot": slot,
                    "fold_id": fold.fold_id,
                    "cost_bps": cost,
                    **trade,
                }
                for trade in ledger["trades"]
            )

    scenarios: dict[str, Any] = {}
    for cost in COSTS_BPS:
        cost_key = str(cost)
        pooled = aggregate_fold_ledgers(fold_ledgers[cost_key])
        pooled_metrics = _metrics_without_paths(pooled)
        acceptance = evaluate_cost_acceptance(
            pooled_metrics, fold_results[cost_key], cost
        )
        scenarios[cost_key] = {
            "pooled_metrics": pooled_metrics,
            "folds": fold_results[cost_key],
            "acceptance": acceptance,
        }

    duplicate_of = f"NOBITEX-R1-NGBOOST-{horizon}-01" if slot > 1 else None
    result = {
        "trial_id": trial_id,
        "horizon": horizon,
        "candidate_slot": slot,
        "duplicate_of": duplicate_of,
        "unique_candidate_configuration": slot == 1,
        "run_id": run_id,
        "fit_calls": len(folds),
        "diagnostic_records": len(diagnostics),
        "cost_scenarios": scenarios,
    }
    return result, diagnostics, trade_rows


def aggregate_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only unique slot-01 diagnostics, weighted by forecast count."""
    if not records:
        raise RuntimeError("regime_r_g46_diagnostics_empty")
    count = sum(int(item["statistics"]["prediction_count"]) for item in records)
    positive_count = sum(
        int(item["statistics"]["median_positive"]["count"]) for item in records
    )
    median_mean = (
        sum(
            float(item["statistics"]["median"]["mean"])
            * int(item["statistics"]["prediction_count"])
            for item in records
        )
        / count
    )
    raw_width = (
        sum(
            float(item["statistics"]["base_quantile_interval_width"]["mean"])
            * int(item["statistics"]["prediction_count"])
            for item in records
        )
        / count
    )
    conformal_width = (
        sum(
            float(item["statistics"]["conformal_interval_width"]["mean"])
            * int(item["statistics"]["prediction_count"])
            for item in records
        )
        / count
    )
    lower_cost: dict[str, Any] = {}
    for cost in COSTS_BPS:
        key = str(cost)
        mean_distance = (
            sum(
                float(
                    item["statistics"]["lower_bound_to_cost"][key]["mean_distance_bps"]
                )
                * int(item["statistics"]["prediction_count"])
                for item in records
            )
            / count
        )
        above = sum(
            int(item["statistics"]["lower_bound_to_cost"][key]["count_above_cost"])
            for item in records
        )
        lower_cost[key] = {
            "mean_distance_bps": mean_distance,
            "count_above_cost": above,
            "percent_above_cost": float(above / count * 100.0),
        }
    return {
        "prediction_count": count,
        "median_positive": {
            "count": positive_count,
            "percentage": float(positive_count / count * 100.0),
        },
        "mean_median_bps": median_mean * 10_000.0,
        "mean_raw_quantile_interval_width_fraction": raw_width,
        "mean_conformal_interval_width_fraction": conformal_width,
        "mean_conformal_interval_width_bps": conformal_width * 10_000.0,
        "lower_bound_minus_cost": lower_cost,
    }


def build_diagnostic_rollup(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_records = [item for item in records if int(item["candidate_slot"]) == 1]
    horizons = ("1h", "4h", "24h")
    by_horizon = {
        horizon: aggregate_diagnostics(
            [item for item in unique_records if item["horizon"] == horizon]
        )
        for horizon in horizons
    }
    return {
        "aggregation_policy": "unique_slot_01_only_weighted_by_prediction_count",
        "overall": aggregate_diagnostics(unique_records),
        "by_horizon": by_horizon,
    }


def _candidate_acceptance(
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique_trials = [item for item in trials if item["unique_candidate_configuration"]]
    candidates = []
    for trial in unique_trials:
        passed = all(
            bool(trial["cost_scenarios"][str(cost)]["acceptance"]["all_gates_passed"])
            for cost in COSTS_BPS
        )
        candidates.append(
            {
                "horizon": trial["horizon"],
                "trial_id": trial["trial_id"],
                "candidate_passed_both_costs": passed,
                "cost_scenarios": trial["cost_scenarios"],
            }
        )
    pass_count = sum(int(item["candidate_passed_both_costs"]) for item in candidates)
    return candidates, pass_count


def _g46_verdict(
    candidates: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> tuple[str, str]:
    if any(item["candidate_passed_both_costs"] for item in candidates):
        return (
            "PASS",
            "At least one unique horizon candidate cleared all four gates at both frozen costs.",
        )
    unique_records = [
        item for item in diagnostics if int(item["candidate_slot"]) == 1
    ]
    qualifying = sum(
        int(
            item["statistics"]["lower_bound_to_cost"][str(cost)][
                "count_above_cost"
            ]
        )
        for item in unique_records
        for cost in COSTS_BPS
    )
    if qualifying == 0:
        return (
            "COMPLETE_ABSTENTION_REPEATED",
            "No calibrated lower bound exceeded either frozen cost in any unique horizon/fold slice.",
        )
    return (
        "NO_GO_PARTIAL_SIGNALING",
        "Some lower bounds exceeded cost, but no unique horizon candidate cleared every registered gate at both costs.",
    )


def _g46_dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    volume = frame["volume"].to_numpy(dtype=np.float64)
    return {
        "path": "data/nobitex/prehack/nobitex-btcusdt-1h.csv",
        "csv_sha256": EXPECTED_DATA_SHA256,
        "row_count": len(frame),
        "volume_zero_rows": int(np.count_nonzero(volume == 0.0)),
        "volume_positive_rows": int(np.count_nonzero(volume > 0.0)),
        "start_utc": frame["timestamp"].iloc[0].isoformat(),
        "end_utc": frame["timestamp"].iloc[-1].isoformat(),
        "scope": "PRE_HACK_NOBITEX_ONLY_EXPLORATORY",
        "historically_exposed": True,
        "confirmatory_holdout": False,
    }


def _g46_baseline_comparison(
    rollup: dict[str, Any],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    baseline = cast(dict[str, Any], prereg["baseline"])
    overall = rollup["overall"]
    return {
        "baseline_run_id": baseline["g4_5_run_id"],
        "median_positive_rate_pct_delta": (
            overall["median_positive"]["percentage"]
            - float(baseline["median_positive_rate_pct"])
        ),
        "conformal_width_fraction_delta": (
            overall["mean_conformal_interval_width_fraction"]
            - float(baseline["mean_conformal_interval_width_fraction"])
        ),
        "lower_bound_minus_cost_bps_delta": {
            str(cost): (
                overall["lower_bound_minus_cost"][str(cost)]["mean_distance_bps"]
                - float(baseline["mean_lower_bound_minus_cost_bps"][str(cost)])
            )
            for cost in COSTS_BPS
        },
    }


def _g46_feature_summary(prereg: dict[str, Any]) -> dict[str, Any]:
    feature = cast(dict[str, Any], prereg["feature_ablation"])
    return {
        "profile": feature["profile"],
        "removed_predictors": list(VOLUME_FEATURE_COLUMNS),
        "retained_predictors": list(RETAINED_FEATURE_COLUMNS),
        "retained_feature_count": len(RETAINED_FEATURE_COLUMNS),
        "feature_contract_sha256": EXPECTED_FEATURE_CONTRACT_SHA256,
        "raw_volume_column_is_predictor": False,
    }


def build_report(
    commit_sha: str,
    run_id: str,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
    prereg: dict[str, Any],
    trials: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the auditable G4.6 report from completed trial evidence."""
    candidates, pass_count = _candidate_acceptance(trials)
    rollup = build_diagnostic_rollup(diagnostics)
    verdict, reason = _g46_verdict(candidates, diagnostics)
    trial_ids = tuple(config["trial_family"]["trial_ids"])
    acceptance = cast(dict[str, Any], prereg["acceptance"])
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "COMPLETED",
        "mode": "HGBQ_CQR_G4_6_VOLUME_FEATURE_ABLATION_OFFLINE_ECONOMIC_EVALUATION",
        "code_commit_sha": commit_sha,
        "workflow_run_id": run_id,
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "dataset": _g46_dataset_summary(frame),
        "g3_configuration_sha256": EXPECTED_CONFIG_SHA256,
        "target_contract": prereg["frozen_g3"]["target_contract"],
        "feature_ablation": _g46_feature_summary(prereg),
        "trial_accounting": {
            "registered_slots": len(trial_ids),
            "unique_candidate_configurations": len(candidates),
            "duplicate_slots": len(trial_ids) - len(candidates),
            "fit_calls": sum(int(item["fit_calls"]) for item in trials),
            "diagnostic_records": len(diagnostics),
        },
        "diagnostics": rollup,
        "baseline_comparison": _g46_baseline_comparison(rollup, prereg),
        "acceptance": {
            "criteria": acceptance["per_cost_gates"],
            "unique_candidates_passing_both_costs": pass_count,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "all_required_gates_passed": pass_count > 0,
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "dsr": {
            "global_n": "UNKNOWN",
            "numeric_dsr": None,
            "status": "NOT_COMPUTABLE",
            "required_for_g4_6_gate": False,
        },
        "trials": trials,
        "diagnostic_records": diagnostics,
        "trade_row_count": len(trade_rows),
        "safety": {
            "real_data_scope": "PRE_HACK_ONLY",
            "holdout_bytes_read": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "hypothetical_trades_simulated": True,
            "trade_ledger_execution": "OFFLINE_LOCAL_CODE_ONLY",
            "exchange_network_access": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".artifacts/regime-r-g4-g6",
    )
    args = parser.parse_args()

    if os.environ.get("ALIX_REGIME_R_G46_EXECUTION_AUTHORIZED") != "true":
        raise PermissionError("regime_r_g46_requires_exact_owner_authorization")
    require_owner_authorization()
    _validate_requested_commit_sha(args.commit_sha)
    prereg = load_g46_preregistration()
    frame, manifest, config = load_and_verify_inputs()
    features = build_volume_ablation_features(frame)
    folds = build_folds(frame, manifest, config)
    trial_ids = tuple(config["trial_family"]["trial_ids"])
    if len(trial_ids) != 30 or set(trial_ids) != set(TRIAL_IDS):
        raise RuntimeError("regime_r_g46_trial_registry_mismatch")

    run_id = os.environ.get("GITHUB_RUN_ID", "local-run")
    trials: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for trial_id in trial_ids:
        result, trial_diagnostics, trial_trades = run_ablation_trial(
            trial_id, frame, features, folds, run_id
        )
        trials.append(result)
        diagnostics.extend(trial_diagnostics)
        trade_rows.extend(trial_trades)

    fit_calls = sum(int(item["fit_calls"]) for item in trials)
    if fit_calls != 90 or len(diagnostics) != 90:
        raise RuntimeError("regime_r_g46_fit_or_diagnostic_accounting_mismatch")

    report = build_report(
        args.commit_sha,
        run_id,
        frame,
        manifest,
        config,
        prereg,
        trials,
        diagnostics,
        trade_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "regime-r-g4-g6-volume-ablation.json"
    csv_path = args.output_dir / "regime-r-g4-g6-hypothetical-trades.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(trade_rows).to_csv(csv_path, index=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "fit_calls": fit_calls,
                "diagnostic_records": len(diagnostics),
                "trade_rows": len(trade_rows),
                "json_artifact": str(json_path),
                "trade_artifact": str(csv_path),
                "Champion": None,
                "trading_authorization": "NO-GO",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
