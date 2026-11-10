"""Train the Nobitex AI ensemble from a verified OHLCV-only dataset."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from pandas import DataFrame

from research.ai_ensemble.ensemble import (
    StackingConfig,
    StackingEnsemble,
    evaluate_walk_forward,
)
from research.ai_ensemble.features import (
    REQUIRED_OHLCV_COLUMNS,
    build_causal_features,
)
from research.ai_ensemble.chronos_features import (
    CHRONOS_CONTEXT_LENGTH,
    CHRONOS_FEATURE_COLUMN,
    CHRONOS_MODEL_ID,
    CHRONOS_MODEL_REVISION,
    CHRONOS_PREDICTION_LENGTH,
    CHRONOS_REFRESH_CANDLES,
    build_chronos_features,
    load_chronos_pipeline,
)
from scripts.fetch_nobitex_public_ohlcv import (
    PREHACK_END_TIMESTAMP,
    PREHACK_EXPECTED_CANDLES,
    PREHACK_START_TIMESTAMP,
)
from research.ai_ensemble.labels import (
    LABEL_HORIZON_CANDLES,
    ROUND_TRIP_COST_BPS,
    build_net_profit_targets,
)
from research.ai_ensemble.risk import KellySizingConfig


def _read_timestamped_csv(path: Path) -> DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"research_input_file_missing:{path}")
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"research_csv_missing_columns:{','.join(missing)}")
    parsed = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"research_csv_invalid_date:{path.name}")
    frame["date"] = parsed
    return frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_dataset_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"research_manifest_missing:{path}")
    try:
        manifest: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("research_manifest_unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("research_manifest_must_be_object")
    return manifest


def _validate_manifest_fields(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "interval",
        "row_count",
        "requested_days",
        "missing_intervals",
        "csv_sha256",
        "start_utc",
        "end_utc",
        "commit_sha",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"research_manifest_missing_fields:{','.join(missing)}")


def _manifest_integer(
    manifest: dict[str, Any],
    field: str,
    error_code: str,
    *,
    minimum: int | None = None,
) -> int:
    value = manifest[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(error_code)
    if minimum is not None and value < minimum:
        raise ValueError(error_code)
    return cast(int, value)


def _validate_expected_candle_count(
    manifest: dict[str, Any],
    row_count: int,
    requested_days: int,
) -> None:
    expected = manifest.get("expected_candles", requested_days * 24)
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError("research_manifest_expected_candles_mismatch")
    if expected != row_count:
        raise ValueError("research_manifest_expected_candles_mismatch")


def _validate_manifest_row_count(
    manifest: dict[str, Any],
    ohlcv: DataFrame,
) -> tuple[int, int]:
    if manifest["interval"] != "1h":
        raise ValueError("research_dataset_interval_must_be_1h")
    row_count = _manifest_integer(
        manifest, "row_count", "research_manifest_row_count_invalid"
    )
    if row_count != len(ohlcv):
        raise ValueError("research_manifest_row_count_mismatch")
    requested_days = _manifest_integer(
        manifest,
        "requested_days",
        "research_manifest_requested_window_mismatch",
        minimum=1,
    )
    _validate_expected_candle_count(manifest, row_count, requested_days)
    if manifest["missing_intervals"] != 0:
        raise ValueError("research_manifest_contains_missing_intervals")
    return row_count, requested_days


def _validate_manifest_content_hash(
    manifest: dict[str, Any],
    ohlcv_path: Path,
) -> None:
    if manifest["csv_sha256"] != _file_sha256(ohlcv_path):
        raise ValueError("research_dataset_sha256_manifest_mismatch")


def _validate_manifest_date_range(
    manifest: dict[str, Any],
    ohlcv: DataFrame,
) -> None:
    try:
        expected_start = pd.Timestamp(manifest["start_utc"])
        expected_end = pd.Timestamp(manifest["end_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("research_manifest_timestamp_invalid") from exc
    if expected_start.tzinfo is None or expected_end.tzinfo is None:
        raise ValueError("research_manifest_timestamps_must_be_timezone_aware")
    actual_start = ohlcv["date"].iloc[0]
    actual_end = ohlcv["date"].iloc[-1]
    if expected_start.tz_convert("UTC") != actual_start:
        raise ValueError("research_manifest_date_range_mismatch")
    if expected_end.tz_convert("UTC") != actual_end:
        raise ValueError("research_manifest_date_range_mismatch")


def _validate_manifest_commit(
    manifest: dict[str, Any],
    expected_commit_sha: str | None,
) -> None:
    recorded_commit = manifest["commit_sha"]
    if not isinstance(recorded_commit, str) or not recorded_commit:
        raise ValueError("research_manifest_commit_sha_missing")
    if expected_commit_sha and recorded_commit != expected_commit_sha:
        raise ValueError("research_manifest_commit_sha_mismatch")


def _validate_manifest_safety(manifest: dict[str, Any]) -> None:
    if manifest.get("public_api_only") is True:
        required_zero_fields = (
            "private_api_calls",
            "exchange_calls",
            "real_order_attempts",
        )
        for field in required_zero_fields:
            if manifest.get(field) != 0:
                raise ValueError(f"research_manifest_safety_violation:{field}")
        if manifest.get("credentials_used") is not False:
            raise ValueError("research_manifest_credentials_must_not_be_used")
    if manifest.get("order_permission", False) is not False:
        raise ValueError("research_manifest_order_permission_must_be_false")
    if manifest.get("Champion") is not None:
        raise ValueError("research_manifest_champion_must_be_null")
    if manifest.get("trading_authorization", "NO-GO") != "NO-GO":
        raise ValueError("research_manifest_trading_authorization_must_be_no_go")


def _validate_nobitex_source_identity(manifest: dict[str, Any]) -> None:
    if manifest.get("source_policy") != "NOBITEX_ONLY_NO_BACKFILL":
        raise ValueError("research_manifest_source_policy_must_be_nobitex_only")
    if manifest.get("data_sources") != ["Nobitex public UDF history API"]:
        raise ValueError("research_manifest_data_sources_mismatch")
    if manifest.get("source_overrides") not in (None, []):
        raise ValueError("research_manifest_external_source_override_forbidden")
    if manifest.get("dataset_window") != "PRE_HACK_NOBITEX_ONLY":
        raise ValueError("research_manifest_window_policy_mismatch")


def _validate_prehack_manifest_bounds(manifest: dict[str, Any]) -> None:
    if manifest.get("start_timestamp") != PREHACK_START_TIMESTAMP:
        raise ValueError("research_manifest_prehack_start_mismatch")
    if manifest.get("end_timestamp") != PREHACK_END_TIMESTAMP:
        raise ValueError("research_manifest_prehack_end_mismatch")
    if manifest.get("expected_candles") != PREHACK_EXPECTED_CANDLES:
        raise ValueError("research_manifest_prehack_count_mismatch")
    if manifest.get("requested_days") != 1325:
        raise ValueError("research_manifest_prehack_days_mismatch")


def _validate_prehack_timestamp_grid(ohlcv: DataFrame) -> None:
    observed = ohlcv["date"].astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000
    expected = np.arange(
        PREHACK_START_TIMESTAMP,
        PREHACK_END_TIMESTAMP + 3600,
        3600,
        dtype=np.int64,
    )
    if not np.array_equal(observed, expected):
        raise ValueError("research_manifest_prehack_timestamp_coverage_mismatch")


def _validate_manifest_source_policy(
    manifest: dict[str, Any],
    ohlcv: DataFrame,
) -> None:
    if manifest.get("public_api_only") is not True:
        return
    _validate_nobitex_source_identity(manifest)
    _validate_prehack_manifest_bounds(manifest)
    _validate_prehack_timestamp_grid(ohlcv)


def _validate_dataset_manifest(
    *,
    manifest_path: Path,
    ohlcv_path: Path,
    ohlcv: DataFrame,
    expected_commit_sha: str | None,
) -> dict[str, Any]:
    manifest = _read_dataset_manifest(manifest_path)
    _validate_manifest_fields(manifest)
    row_count, requested_days = _validate_manifest_row_count(manifest, ohlcv)
    _validate_manifest_content_hash(manifest, ohlcv_path)
    _validate_manifest_date_range(manifest, ohlcv)
    _validate_manifest_commit(manifest, expected_commit_sha)
    _validate_manifest_safety(manifest)
    _validate_manifest_source_policy(manifest, ohlcv)
    return manifest


def _finite_feature_positions(features: DataFrame) -> np.ndarray:
    finite_mask = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    positions = np.flatnonzero(finite_mask)
    if len(positions) == 0:
        raise ValueError("research_feature_warmup_left_no_eligible_rows")
    return positions


def _feature_position_map(positions: np.ndarray) -> dict[int, int]:
    return {
        int(market_index): model_index
        for model_index, market_index in enumerate(positions.tolist())
    }


def _mapped_target_record(
    record: Any,
    model_index: int,
    feature_position: dict[int, int],
) -> tuple[int, int, float] | None:
    target = record["target"]
    label_end = record["label_end_idx"]
    net_return = record["net_return"]
    if pd.isna(target) or pd.isna(label_end) or pd.isna(net_return):
        return None
    original_end = int(label_end)
    if original_end not in feature_position:
        raise ValueError("research_label_end_missing_from_feature_timeline")
    compressed_end = feature_position[original_end]
    if compressed_end < model_index:
        raise ValueError("research_label_end_precedes_signal")
    return int(target), compressed_end, float(net_return)


def _populate_target_arrays(
    targets: DataFrame,
    market_positions: np.ndarray,
    feature_position: dict[int, int],
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    labels = np.zeros(row_count, dtype=int)
    label_ends = np.full(row_count, -1, dtype=int)
    net_returns = np.zeros(row_count, dtype=float)
    eligible: list[int] = []
    for model_index, market_index in enumerate(market_positions.tolist()):
        mapped = _mapped_target_record(
            targets.iloc[int(market_index)], model_index, feature_position
        )
        if mapped is None:
            continue
        target, compressed_end, net_return = mapped
        labels[model_index] = target
        label_ends[model_index] = compressed_end
        net_returns[model_index] = net_return
        eligible.append(model_index)
    return labels, label_ends, net_returns, eligible


def _validate_training_labels(labels: np.ndarray, eligible: list[int]) -> None:
    if len(eligible) < 500:
        raise ValueError("research_fewer_than_500_fully_labeled_rows")
    if len(np.unique(labels[eligible])) != 2:
        raise ValueError("research_labels_do_not_contain_both_classes")


def _build_aligned_training_set(
    ohlcv: DataFrame,
    chronos_pipeline: Any,
) -> tuple[DataFrame, DataFrame, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    baseline_features = build_causal_features(ohlcv)
    chronos_features = build_chronos_features(ohlcv, pipeline=chronos_pipeline)
    candidate_features = baseline_features.copy()
    candidate_features[CHRONOS_FEATURE_COLUMN] = chronos_features[
        CHRONOS_FEATURE_COLUMN
    ]
    # Exclude the Chronos context warm-up from both models so every row,
    # target, purge, embargo and outer fold is identical.
    positions = _finite_feature_positions(candidate_features)
    position_map = _feature_position_map(positions)
    baseline_matrix = baseline_features.iloc[positions].reset_index(drop=True)
    candidate_matrix = candidate_features.iloc[positions].reset_index(drop=True)
    targets = build_net_profit_targets(ohlcv)
    labels, label_ends, net_returns, eligible = _populate_target_arrays(
        targets, positions, position_map, len(candidate_matrix)
    )
    _validate_training_labels(labels, eligible)
    return (
        baseline_matrix,
        candidate_matrix,
        labels,
        label_ends,
        net_returns,
        eligible,
    )


def _metric_delta(candidate: object, baseline: object) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(
        baseline, (int, float)
    ):
        return None
    return float(candidate) - float(baseline)


def run_training(
    *,
    ohlcv_path: Path,
    manifest_path: Path,
    model_path: Path,
    model_manifest_path: Path,
    report_path: Path,
    commit_sha: str | None = None,
    chronos_pipeline: Any | None = None,
) -> dict[str, Any]:
    ohlcv = _read_timestamped_csv(ohlcv_path)
    expected_commit = commit_sha if commit_sha and commit_sha != "UNKNOWN" else None
    dataset_manifest = _validate_dataset_manifest(
        manifest_path=manifest_path,
        ohlcv_path=ohlcv_path,
        ohlcv=ohlcv,
        expected_commit_sha=expected_commit,
    )
    if chronos_pipeline is None:
        active_pipeline, chronos_metadata = load_chronos_pipeline()
    else:
        active_pipeline = chronos_pipeline
        chronos_metadata = {
            "model_id": CHRONOS_MODEL_ID,
            "model_revision": CHRONOS_MODEL_REVISION,
            "model_license": "Apache-2.0",
            "chronos_forecasting_version": "INJECTED_TEST_PIPELINE",
            "weight_sha256": "INJECTED_TEST_PIPELINE",
            "device": "test",
            "context_length": CHRONOS_CONTEXT_LENGTH,
            "prediction_length": CHRONOS_PREDICTION_LENGTH,
            "forecast_refresh_candles": CHRONOS_REFRESH_CANDLES,
            "forecast_type": "zero_shot",
        }
    (
        baseline_features,
        candidate_features,
        labels,
        label_ends,
        net_returns,
        eligible,
    ) = _build_aligned_training_set(ohlcv, active_pipeline)
    config = StackingConfig()
    validation_args = (
        labels.tolist(),
        label_ends.tolist(),
        net_returns.tolist(),
        eligible,
    )
    baseline_validation = evaluate_walk_forward(
        baseline_features, *validation_args, config=config
    )
    candidate_validation = evaluate_walk_forward(
        candidate_features, *validation_args, config=config
    )
    model = StackingEnsemble(config).fit(
        candidate_features, labels.tolist(), label_ends.tolist(), eligible
    )
    effective_commit = commit_sha or dataset_manifest["commit_sha"]
    if effective_commit == "UNKNOWN":
        effective_commit = dataset_manifest["commit_sha"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "alix.nobitex.ai-ensemble.v2",
            "model": model,
            "baseline_feature_columns": list(baseline_features.columns),
            "feature_columns": list(candidate_features.columns),
            "chronos": chronos_metadata,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "label_horizon_candles": LABEL_HORIZON_CANDLES,
            "training_rows": len(eligible),
            "dataset_csv_sha256": dataset_manifest["csv_sha256"],
            "dataset_manifest_sha256": _file_sha256(manifest_path),
            "code_commit_sha": effective_commit,
        },
        model_path,
    )
    baseline_metrics = cast(dict[str, object], baseline_validation["aggregate"])
    candidate_metrics = cast(dict[str, object], candidate_validation["aggregate"])
    comparison = {
        "net_profit_factor_delta": _metric_delta(
            candidate_metrics["profit_factor"], baseline_metrics["profit_factor"]
        ),
        "win_rate_delta": _metric_delta(
            candidate_metrics["win_rate"], baseline_metrics["win_rate"]
        ),
        "max_drawdown_delta": _metric_delta(
            candidate_metrics["max_drawdown"], baseline_metrics["max_drawdown"]
        ),
        "trade_count_delta": cast(int, candidate_metrics["trade_count"])
        - cast(int, baseline_metrics["trade_count"]),
    }
    model_manifest: dict[str, Any] = {
        "schema_version": "alix.nobitex.ai-ensemble-model-manifest.v2",
        "artifact": {
            "path": str(model_path),
            "sha256": _file_sha256(model_path),
        },
        "dataset": {
            "csv_sha256": dataset_manifest["csv_sha256"],
            "manifest_sha256": _file_sha256(manifest_path),
            "row_count": dataset_manifest["row_count"],
            "requested_days": dataset_manifest["requested_days"],
            "expected_candles": dataset_manifest.get(
                "expected_candles", dataset_manifest["row_count"]
            ),
            "interval": dataset_manifest["interval"],
            "provider": dataset_manifest.get("provider", "UNSPECIFIED"),
            "symbol": dataset_manifest.get("symbol", "UNSPECIFIED"),
            "start_timestamp": dataset_manifest.get("start_timestamp"),
            "end_timestamp": dataset_manifest.get("end_timestamp"),
        },
        "model": {
            "base_models": ["LightGBM", "XGBoost"],
            "regime_model": "3-state diagonal Gaussian HMM (forward-filtered)",
            "meta_learner": "LogisticRegression trained on out-of-fold outputs",
            "baseline_feature_columns": list(baseline_features.columns),
            "candidate_feature_columns": list(candidate_features.columns),
        },
        "chronos": chronos_metadata,
        "labeling": {
            "method": "ATR_TRIPLE_BARRIER",
            "horizon_candles": LABEL_HORIZON_CANDLES,
            "take_profit_atr": 2.0,
            "stop_loss_atr": 1.0,
            "entry_reference": "NEXT_BAR_OPEN",
            "collision_policy": "STOP_LOSS_FIRST",
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "validation": {
            "method": candidate_validation["validation_method"],
            "fold_count": candidate_validation["fold_count"],
            "baseline_aggregate": baseline_metrics,
            "candidate_aggregate": candidate_metrics,
            "cost_sensitivity_90_bps": candidate_validation["cost_sensitivity_90_bps"],
        },
        "code_commit_sha": effective_commit,
        "promotion_status": "BLOCKED_NO_ECONOMIC_ACCEPTANCE",
        "trading_authorization": "NO-GO",
    }
    model_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    model_manifest_path.write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema_version": "alix.nobitex.ai-ensemble-training-report.v3",
        "dataset": {
            "kind": (
                "PUBLIC_MARKET_DATA"
                if dataset_manifest.get("public_api_only") is True
                else "NON_PRODUCTION_TEST_FIXTURE"
            ),
            "provider": dataset_manifest.get("provider", "UNSPECIFIED"),
            "data_sources": dataset_manifest.get(
                "data_sources", [dataset_manifest.get("provider", "UNSPECIFIED")]
            ),
            "source_policy": dataset_manifest.get("source_policy", "TEST_FIXTURE"),
            "dataset_window": dataset_manifest.get("dataset_window", "TEST_FIXTURE"),
            "symbol": dataset_manifest.get("symbol", "UNSPECIFIED"),
            "pair": dataset_manifest.get("pair", "UNSPECIFIED"),
            "interval": dataset_manifest["interval"],
            "start_utc": dataset_manifest["start_utc"],
            "end_utc": dataset_manifest["end_utc"],
            "start_timestamp": dataset_manifest.get("start_timestamp"),
            "end_timestamp": dataset_manifest.get("end_timestamp"),
            "requested_days": dataset_manifest["requested_days"],
            "row_count": dataset_manifest["row_count"],
            "expected_candles": dataset_manifest.get(
                "expected_candles", dataset_manifest["row_count"]
            ),
            "eligible_labeled_rows": len(eligible),
            "csv_sha256": dataset_manifest["csv_sha256"],
            "manifest_sha256": _file_sha256(manifest_path),
        },
        "model": {
            "base_models": ["LightGBM", "XGBoost"],
            "regime_model": "3-state diagonal Gaussian HMM (forward-filtered)",
            "meta_learner": "LogisticRegression trained on out-of-fold outputs",
            "baseline_feature_columns": list(baseline_features.columns),
            "candidate_feature_columns": list(candidate_features.columns),
        },
        "chronos": chronos_metadata,
        "labeling": {
            "method": "ATR_TRIPLE_BARRIER",
            "horizon_candles": LABEL_HORIZON_CANDLES,
            "take_profit_atr": 2.0,
            "stop_loss_atr": 1.0,
            "entry_reference": "NEXT_BAR_OPEN",
            "collision_policy": "STOP_LOSS_FIRST",
            "maker_fee_bps_per_side": 25,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "risk": {
            "method": "FRACTIONAL_KELLY",
            "kelly_fraction": KellySizingConfig().kelly_fraction,
            "maximum_risk_fraction": KellySizingConfig().max_risk_fraction,
            "maximum_position_fraction": KellySizingConfig().max_position_fraction,
        },
        "validation": {
            "baseline": baseline_validation,
            "candidate": candidate_validation,
            "fold_count": candidate_validation["fold_count"],
        },
        "baseline_research_result": {
            "trade_count": baseline_metrics["trade_count"],
            "win_rate": baseline_metrics["win_rate"],
            "net_profit_factor": baseline_metrics["profit_factor"],
            "max_drawdown": baseline_metrics["max_drawdown"],
            "net_return": baseline_metrics["compounded_net_return"],
            "statistical_z_score": baseline_metrics["statistical_z_score"],
        },
        "candidate_research_result": {
            "trade_count": candidate_metrics["trade_count"],
            "win_rate": candidate_metrics["win_rate"],
            "net_profit_factor": candidate_metrics["profit_factor"],
            "max_drawdown": candidate_metrics["max_drawdown"],
            "net_return": candidate_metrics["compounded_net_return"],
            "statistical_z_score": candidate_metrics["statistical_z_score"],
        },
        "comparison": comparison,
        "cost_sensitivity_90_bps": candidate_validation["cost_sensitivity_90_bps"],
        "research_result": {
            "trade_count": candidate_metrics["trade_count"],
            "win_rate": candidate_metrics["win_rate"],
            "net_profit_factor": candidate_metrics["profit_factor"],
            "max_drawdown": candidate_metrics["max_drawdown"],
            "net_return": candidate_metrics["compounded_net_return"],
            "statistical_z_score": candidate_metrics["statistical_z_score"],
            "z_score_definition": (
                "one-sample normal-approximation Z-score of non-overlapping "
                "net trade returns against a zero-mean null"
            ),
        },
        "artifact": {
            "path": str(model_path),
            "sha256": _file_sha256(model_path),
            "model_manifest_path": str(model_manifest_path),
            "model_manifest_sha256": _file_sha256(model_manifest_path),
        },
        "safety": {
            "ALIX_DRY_RUN": True,
            "ALIX_ALLOW_LIVE_TRADING": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "economic_acceptance": "NO-GO",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("user_data/models/nobitex-ai-ensemble.joblib"),
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("user_data/models/nobitex-ai-ensemble.manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("user_data/backtest_results/nobitex-ai-ensemble-report.json"),
    )
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    args = parser.parse_args()
    report = run_training(
        ohlcv_path=args.ohlcv,
        manifest_path=args.manifest,
        model_path=args.model,
        model_manifest_path=args.model_manifest,
        report_path=args.report,
        commit_sha=args.commit_sha,
    )
    print(
        json.dumps(
            {
                "status": "RESEARCH_COMPLETE_NOT_PROMOTED",
                "model_path": report["artifact"]["path"],
                "training_rows": report["dataset"]["eligible_labeled_rows"],
                "validation": report["research_result"],
                "safety": report["safety"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
