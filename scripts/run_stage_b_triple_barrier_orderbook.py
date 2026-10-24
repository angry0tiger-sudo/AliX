"""Stage B Triple-Barrier and relative cross-venue Order Book research."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from scripts.run_stage_b_walk_forward import (
    A_HOLDOUT,
    HOUR,
    SOURCE_RUN_ID,
    utc_text,
    verify_source_artifact,
)
from services.meta_labeling.triple_barrier import (
    TripleBarrierConfig,
    apply_triple_barrier,
)

SCHEMA = "alix.stage-b.triple-barrier-orderbook.v1"
HORIZONS = (24, 48)
PT_MULTIPLIERS = (1.5, 2.0)
SL_MULTIPLIERS = (0.75, 1.0)
SLIPPAGE_BPS = (20.0, 35.0, 50.0)
FEE_BPS_SIDE = 25.0
PARTICIPATION_CAP = 0.01
BOOK_TIME = "snapshot_timestamp_utc"
BOOK_LEVELS = 20
BOOK_MAX_AGE_SECONDS = 60
INITIAL_TRAIN_ROWS = 3 * 365 * 24
OOS_WINDOW_ROWS = 90 * 24
PROBABILITY_THRESHOLD = 0.50
REFERENCE_ORDER_NOTIONAL_QUOTE = 100.0
ORDERBOOK_MANIFEST_SCHEMA = "alix.binance-orderbook-manifest.v1"
SAFETY: dict[str, object] = {
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
    "economic_acceptance": "NO-GO",
    "model_promoted": False,
    "production_model_updated": False,
    "strategy_registry_updated": False,
    "orders_or_execution_api_used": False,
    "private_api_calls": 0,
    "order_endpoint_calls": 0,
    "real_order_attempts": 0,
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _required_orderbook_columns() -> set[str]:
    required = {BOOK_TIME}
    for level in range(1, BOOK_LEVELS + 1):
        required.update(
            {
                f"bid_price_{level}",
                f"bid_qty_{level}",
                f"ask_price_{level}",
                f"ask_qty_{level}",
            }
        )
    return required


def _validate_book_timestamps(frame: DataFrame) -> None:
    frame[BOOK_TIME] = pd.to_datetime(frame[BOOK_TIME], utc=True, errors="raise")
    timestamps = frame[BOOK_TIME]
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("orderbook_timestamp_invalid_or_duplicate")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("orderbook_timestamps_not_monotonic")


def _validate_book_level(frame: DataFrame, side: str, level: int) -> None:
    price_column = f"{side}_price_{level}"
    quantity_column = f"{side}_qty_{level}"
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    frame[quantity_column] = pd.to_numeric(frame[quantity_column], errors="coerce")
    values = frame[[price_column, quantity_column]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"orderbook_non_finite:{side}:{level}")
    if (frame[price_column] <= 0).any() or (frame[quantity_column] < 0).any():
        raise ValueError(f"orderbook_value_out_of_range:{side}:{level}")


def _validate_book_price_order(bids: np.ndarray, asks: np.ndarray) -> None:
    if np.any(np.diff(bids, axis=1) > 0) or np.any(np.diff(asks, axis=1) < 0):
        raise ValueError("orderbook_prices_not_price_sorted")
    if np.any(bids[:, 0] >= asks[:, 0]):
        raise ValueError("orderbook_crossed_or_locked")


def _validate_book_side_depth(bid_quantities: np.ndarray, ask_quantities: np.ndarray) -> None:
    if np.any(bid_quantities.sum(axis=1) <= 0) or np.any(
        ask_quantities.sum(axis=1) <= 0
    ):
        raise ValueError("orderbook_side_depth_empty")


def _validate_book_geometry(frame: DataFrame) -> None:
    bids = frame[[f"bid_price_{i}" for i in range(1, BOOK_LEVELS + 1)]].to_numpy(float)
    asks = frame[[f"ask_price_{i}" for i in range(1, BOOK_LEVELS + 1)]].to_numpy(float)
    bid_quantities = frame[
        [f"bid_qty_{i}" for i in range(1, BOOK_LEVELS + 1)]
    ].to_numpy(float)
    ask_quantities = frame[
        [f"ask_qty_{i}" for i in range(1, BOOK_LEVELS + 1)]
    ].to_numpy(float)
    _validate_book_price_order(bids, asks)
    _validate_book_side_depth(bid_quantities, ask_quantities)


def _validate_orderbook(book: DataFrame) -> DataFrame:
    required = _required_orderbook_columns()
    missing = sorted(required - set(book.columns))
    if missing:
        raise ValueError(f"orderbook_schema_missing:{','.join(missing)}")
    if book.empty:
        raise ValueError("orderbook_dataset_empty")
    result = book.loc[:, sorted(required)].copy()
    _validate_book_timestamps(result)
    for level in range(1, BOOK_LEVELS + 1):
        for side in ("bid", "ask"):
            _validate_book_level(result, side, level)
    _validate_book_geometry(result)
    return result


def _validate_orderbook_manifest(
    book_path: Path, manifest_path: Path
) -> dict[str, Any]:
    if not book_path.is_file() or not manifest_path.is_file():
        raise ValueError("orderbook_data_or_manifest_missing")
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise ValueError("orderbook_manifest_not_object")
    required_fields = {
        "schema",
        "venue",
        "symbol",
        "source_url",
        "permission_status",
        "csv_sha256",
    }
    if not required_fields.issubset(manifest_value):
        raise ValueError("orderbook_manifest_schema_missing")
    if manifest_value["schema"] != ORDERBOOK_MANIFEST_SCHEMA:
        raise ValueError("orderbook_manifest_schema_unsupported")
    if manifest_value["venue"] != "binance_usdm_futures":
        raise ValueError("orderbook_manifest_venue_mismatch")
    if manifest_value["symbol"] != "BTCUSDT":
        raise ValueError("orderbook_manifest_symbol_mismatch")
    if manifest_value["permission_status"] != "approved":
        raise ValueError("orderbook_data_permission_not_approved")
    source = urlsplit(str(manifest_value["source_url"]))
    if source.scheme != "https" or "binance" not in (source.hostname or "").casefold():
        raise ValueError("orderbook_manifest_source_url_invalid")
    expected_hash = str(manifest_value["csv_sha256"])
    actual_hash = hashlib.sha256(book_path.read_bytes()).hexdigest()
    if len(expected_hash) != 64 or expected_hash != actual_hash:
        raise ValueError("orderbook_manifest_checksum_mismatch")
    return manifest_value


def _load_orderbook_features(
    candles: DataFrame,
    orderbook_path: Path | None,
    manifest_path: Path | None,
) -> tuple[str, DataFrame | None, int | None, str | None, list[str]]:
    if orderbook_path is None:
        return (
            "NOT_PROVIDED",
            None,
            None,
            None,
            ["historical_binance_orderbook_snapshots_not_provided"],
        )
    if manifest_path is None:
        return (
            "INVALID",
            None,
            None,
            None,
            ["orderbook_provenance_manifest_not_provided"],
        )
    try:
        manifest = _validate_orderbook_manifest(orderbook_path, manifest_path)
        book = pd.read_csv(orderbook_path)
        features = add_relative_orderbook_features(candles, book)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError, pd.errors.ParserError) as error:
        return (
            "INVALID",
            None,
            None,
            None,
            [f"orderbook_data_invalid:{error}"],
        )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return "VALIDATED_RELATIVE_FEATURES", features, len(book), manifest_hash, []


def _build_feature_groups(
    price_features: DataFrame, book_features: DataFrame | None
) -> dict[str, DataFrame]:
    groups: dict[str, DataFrame] = {"ohlcv_only": price_features}
    if book_features is not None:
        groups["ohlcv_plus_binance_relative_book"] = pd.concat(
            [
                price_features.reset_index(drop=True),
                book_features.drop(columns=["date"]),
            ],
            axis=1,
        )
    return groups


def _evaluate_grid(
    candles: DataFrame,
    outcomes_by_config: dict[tuple[int, float, float], DataFrame],
    feature_groups: dict[str, DataFrame],
    capacity: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_folds: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for (horizon, profit_multiplier, stop_multiplier), outcomes in outcomes_by_config.items():
        baseline_id = (
            f"ohlcv_only|h{horizon}|pt{profit_multiplier:g}|sl{stop_multiplier:g}"
        )
        for group_name, matrix in feature_groups.items():
            candidate_id = baseline_id.replace("ohlcv_only", group_name, 1)
            folds, predictions = _evaluate_candidate(
                candles, outcomes, matrix, capacity, candidate_id
            )
            all_folds.extend(folds)
            all_predictions.extend(predictions)
            for slippage in SLIPPAGE_BPS:
                stress_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "feature_group": group_name,
                        "horizon_hours": horizon,
                        "profit_taking_multiplier": profit_multiplier,
                        "stop_loss_multiplier": stop_multiplier,
                        **_stress_metrics(predictions, slippage),
                    }
                )
    return all_folds, all_predictions, stress_rows


def _robust_passing_candidates(stress_rows: list[dict[str, Any]]) -> set[str]:
    outcomes: dict[str, list[bool]] = {}
    for row in stress_rows:
        if row["feature_group"] != "ohlcv_plus_binance_relative_book":
            continue
        outcomes.setdefault(str(row["candidate_id"]), []).append(
            bool(row["strict_economic_gate_on_executable_trades"]["passed"])
        )
    return {
        candidate_id
        for candidate_id, passed_scenarios in outcomes.items()
        if len(passed_scenarios) == len(SLIPPAGE_BPS) and all(passed_scenarios)
    }


def _liquidity_blockers(liquidity: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if liquidity["status"] != "READY_FOR_CAP_CHECK":
        blockers.append(str(liquidity["status"]).lower())
    if int(liquidity["rows_with_reference_order_capacity"]) == 0:
        blockers.append("liquidity_filter_blocks_all_reference_orders")
    return blockers


def _build_report(
    commit_sha: str,
    candles: DataFrame,
    source_report: dict[str, Any],
    segment_hash: str,
    outcomes_by_config: dict[tuple[int, float, float], DataFrame],
    barrier_scenarios: list[dict[str, Any]],
    feature_groups: dict[str, DataFrame],
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    orderbook_status: str,
    orderbook_path: Path | None,
    orderbook_rows: int | None,
    orderbook_manifest_hash: str | None,
    liquidity: dict[str, Any],
    blockers: list[str],
    economic_ready: bool,
    passing_candidates: set[str],
) -> dict[str, Any]:
    status = (
        "NOT_EVALUATED_BLOCKED_INPUT_DATA"
        if not economic_ready
        else "THRESHOLDS_MET_RESEARCH_ONLY"
        if passing_candidates
        else "NO_GO_THRESHOLDS_NOT_MET"
    )
    return {
        "report_schema": SCHEMA,
        "run_status": "BLOCKED_INPUT_DATA" if blockers else "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": SOURCE_RUN_ID,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "coverage_manifest_sha256": source_report["manifest_sha256"],
            "segment_a_rows_sha256": segment_hash,
            "symbol": "BTCUSDT",
            "ohlcv_venue": "Nobitex",
            "timeframe": "1h",
            "development_rows": len(candles),
            "reserved_final_90_day_holdout_rows": A_HOLDOUT,
            "holdout_used": False,
            "development_start_utc": utc_text(candles["date"].iloc[0]),
            "development_end_utc": utc_text(candles["date"].iloc[-1]),
        },
        "protocol": {
            "horizons_hours": list(HORIZONS),
            "profit_taking_multipliers": list(PT_MULTIPLIERS),
            "stop_loss_multipliers": list(SL_MULTIPLIERS),
            "barrier_candidate_count": len(outcomes_by_config),
            "model_candidate_count_planned": 2 * len(outcomes_by_config),
            "model_candidate_count_executed": len(feature_groups) * len(outcomes_by_config),
            "model_candidate_count_blocked_for_missing_book": (
                0 if orderbook_status == "VALIDATED_RELATIVE_FEATURES"
                else len(outcomes_by_config)
            ),
            "diagnostic_scenario_count": len(barrier_scenarios),
            "oos_prediction_row_count": len(prediction_rows),
            "fee_bps_per_side": FEE_BPS_SIDE,
            "slippage_bps_per_side": list(SLIPPAGE_BPS),
            "round_trip_total_cost_bps": [
                2.0 * (FEE_BPS_SIDE + slip) for slip in SLIPPAGE_BPS
            ],
            "stress_acceptance_requires_all_slippage_scenarios": True,
            "reference_order_notional_quote_currency": REFERENCE_ORDER_NOTIONAL_QUOTE,
            "liquidity_participation_cap_fraction": PARTICIPATION_CAP,
            "orderbook_venue": "Binance USD-M Futures BTCUSDT",
            "orderbook_feature_policy": (
                "Only normalized top-5/top-20 imbalance and relative best-level spread "
                "are used; absolute depth is not used for Nobitex capacity or as a model feature."
            ),
            "book_snapshot_max_age_seconds": BOOK_MAX_AGE_SECONDS,
            "book_join": "backward point-in-time join; stale or missing snapshot fails closed",
            "probability_threshold_fixed_not_tuned": PROBABILITY_THRESHOLD,
        },
        "data_availability": {
            "orderbook_status": orderbook_status,
            "orderbook_csv": str(orderbook_path) if orderbook_path else None,
            "orderbook_rows": orderbook_rows,
            "orderbook_manifest_sha256": orderbook_manifest_hash,
            "liquidity": liquidity,
            "blockers": blockers,
        },
        "barrier_outcome_diagnostics_not_execution_backtest": barrier_scenarios,
        "walk_forward": {
            "feature_groups": list(feature_groups),
            "fold_metrics": fold_rows,
            "candidate_cost_stress_metrics": stress_rows,
            "purge": "training label_end_index < test_start; test label_end_index < test_stop",
            "counterfactual_predictions_are_not_economic_evidence": True,
        },
        "economic_evaluation": {
            "status": status,
            "executable_trade_count": sum(
                1 for row in prediction_rows if row["executable_trade"]
            ),
            "candidate_pass_count": len(passing_candidates),
            "required_oos_trade_count": ">=100 executable trades",
            "required_win_rate": ">60%",
            "required_net_expectancy": ">0 after costs",
            "required_absolute_max_drawdown": "<15%",
            "all_gates_must_pass": True,
        },
        "cross_venue_limitations": [
            "Binance Order Book is used only for relative directional imbalance and relative spread.",
            "Absolute Binance depth is never interpreted as Nobitex depth or liquidity.",
            "Reference research order notional is fixed at 100 quote units and is not a production stake setting.",
            "Order size must not exceed 1% of contemporaneous, unit-verified Nobitex quote notional.",
            "All 58,655 source Segment A candles have zero volume, so current sample grants zero order capacity.",
            "The pinned OHLCV artifact contains no historical Binance Level-2 snapshots; no synthetic Order Book is generated.",
            "Book data requires a manifest, checksum, source identity and owner-asserted permission status before features are used.",
            "Stress costs are 25 bps per-side fee plus 20/35/50 bps per-side slippage, for 90/120/150 bps round trip.",
        ],
        "safety_invariants": dict(SAFETY),
    }


def run_experiment(
    input_dir: Path,
    output_dir: Path,
    commit_sha: str,
    orderbook_csv: Path | None,
    orderbook_manifest: Path | None,
    volume_unit: str,
) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("research_commit_sha_invalid")
    candles, source_report, segment_hash = _load_development(input_dir)
    outcomes_by_config, barrier_scenarios = _event_outcomes(candles)
    liquidity, capacity = _liquidity_capacity(candles, volume_unit)
    orderbook_result = _load_orderbook_features(
        candles, orderbook_csv, orderbook_manifest
    )
    orderbook_status, book_features, book_rows, manifest_hash, book_blockers = orderbook_result
    blockers = book_blockers + _liquidity_blockers(liquidity)
    price_features = _price_features(candles)
    feature_groups = _build_feature_groups(price_features, book_features)
    fold_rows, prediction_rows, stress_rows = _evaluate_grid(
        candles, outcomes_by_config, feature_groups, capacity
    )
    economic_ready = (
        book_features is not None
        and liquidity["status"] == "READY_FOR_CAP_CHECK"
        and int(liquidity["rows_with_reference_order_capacity"]) > 0
    )
    passing_candidates = _robust_passing_candidates(stress_rows)
    report = _build_report(
        commit_sha,
        candles,
        source_report,
        segment_hash,
        outcomes_by_config,
        barrier_scenarios,
        feature_groups,
        fold_rows,
        prediction_rows,
        stress_rows,
        orderbook_status,
        orderbook_csv,
        book_rows,
        manifest_hash,
        liquidity,
        blockers,
        economic_ready,
        passing_candidates,
    )
    if report["safety_invariants"] != SAFETY:
        raise ValueError("research_safety_invariant_violation")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "triple-barrier-orderbook-report.json", report)
    _write_csv(output_dir / "barrier-outcome-diagnostics.csv", barrier_scenarios)
    _write_csv(output_dir / "oos-predictions.csv", prediction_rows)
    _write_csv(output_dir / "oos-candidate-cost-stress.csv", stress_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--orderbook-csv", type=Path, default=None)
    parser.add_argument("--orderbook-manifest", type=Path, default=None)
    parser.add_argument(
        "--nobitex-volume-unit",
        choices=("unknown", "base_asset", "quote_asset"),
        default="unknown",
    )
    args = parser.parse_args()
    report = run_experiment(
        args.input_dir,
        args.output_dir,
        args.commit_sha,
        args.orderbook_csv,
        args.orderbook_manifest,
        args.nobitex_volume_unit,
    )
    print(
        json.dumps(
            {
                "commit_sha": report["created_by_commit_sha"],
                "run_status": report["run_status"],
                "barrier_scenarios": report["protocol"]["diagnostic_scenario_count"],
                "executed_model_candidates": report["protocol"][
                    "model_candidate_count_executed"
                ],
                "oos_predictions": report["protocol"]["oos_prediction_row_count"],
                "orderbook_status": report["data_availability"]["orderbook_status"],
                "liquidity_status": report["data_availability"]["liquidity"]["status"],
                "economic_status": report["economic_evaluation"]["status"],
                "Champion": report["safety_invariants"]["Champion"],
                "order_permission": report["safety_invariants"]["order_permission"],
                "trading_authorization": report["safety_invariants"][
                    "trading_authorization"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
