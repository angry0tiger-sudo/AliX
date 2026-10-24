"""Run Stage B gap-aware walk-forward research on the pinned Stage A artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_market_features,
    forward_return_target,
)
from services.freqai_baseline.walkforward import (
    BASELINE_MODEL_PARAMETERS,
    validate_ohlcv,
)

HOUR = 3600
HORIZON = 12
WINDOW = 90 * 24
A_HOLDOUT = 90 * 24
A_TRAIN = 3 * 365 * 24
B_TRAIN = 180 * 24
FEE_BPS_SIDE = 25.0
SLIPPAGE_BPS_SIDE = 5.0
ROUND_TRIP_COST = 2.0 * (FEE_BPS_SIDE + SLIPPAGE_BPS_SIDE) / 10_000.0
SOURCE_RUN_ID = 40077609244
SOURCE_ARTIFACT = "alix-nobitex-history-coverage-40077609244"
SOURCE_COMMIT = "8212da14e987fa10d7ab28f9aff190e9ec7d6fa6"
SOURCE_CSV_SHA256 = "6b3f3ae006b8dc58d9f0f4092250cb484073984fb2249ed795bc0e12e823994c"
REPORT_SCHEMA = "alix.stage-b.gap-aware-walk-forward.v1"
CSV_NAME = "nobitex-btcusdt-1h-history-coverage.csv"
MANIFEST_NAME = "nobitex-btcusdt-1h-history-coverage.manifest.json"
REPORT_NAME = "nobitex-history-coverage-report.json"


class StageBError(RuntimeError):
    """Raised when input provenance or data cannot support a valid evaluation."""


class FoldSpec:
    """A temporal split with exclusive row endpoints."""

    def __init__(
        self, fold_id: str, train_stop: int, test_start: int, test_stop: int
    ) -> None:
        self.fold_id = fold_id
        self.train_stop = train_stop
        self.test_start = test_start
        self.test_stop = test_stop


def utc_text(value: object) -> str:
    timestamp = pd.Timestamp(cast(Any, value))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise StageBError("timestamp_timezone_missing")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + chr(10),
        encoding="utf-8",
    )
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageBError("source_json_not_object")
    return value


def segment_bounds(
    timestamps: Sequence[int],
) -> tuple[list[tuple[int, int]], list[dict[str, int]]]:
    """Detect only observed discontinuities. Nothing is sorted or filled."""
    if len(timestamps) < 2:
        raise StageBError("insufficient_candles_for_segment_detection")
    starts = [0]
    gaps: list[dict[str, int]] = []
    previous = int(timestamps[0])
    for index in range(1, len(timestamps)):
        current = int(timestamps[index])
        delta = current - previous
        if delta <= 0:
            raise StageBError("timestamps_not_strictly_increasing")
        if delta % HOUR:
            raise StageBError("timestamp_delta_not_hour_aligned")
        if delta > HOUR:
            starts.append(index)
            gaps.append(
                {
                    "left_timestamp": previous,
                    "right_timestamp": current,
                    "missing_hours": delta // HOUR - 1,
                }
            )
        previous = current
    stops = starts[1:] + [len(timestamps)]
    return list(zip(starts, stops, strict=True)), gaps


def check_artifact_metadata(
    manifest: dict[str, Any],
    report: dict[str, Any],
    csv_hash: str,
    manifest_hash: str,
) -> None:
    if csv_hash != SOURCE_CSV_SHA256:
        raise StageBError("source_csv_sha256_mismatch")
    manifest_expected = {
        "csv_sha256": csv_hash,
        "commit_sha": SOURCE_COMMIT,
        "schema_version": "alix.nobitex.history-coverage.v1",
    }
    if any(manifest.get(key) != value for key, value in manifest_expected.items()):
        raise StageBError("source_manifest_contract_mismatch")
    report_expected = {
        "audit_status": "COMPLETED",
        "commit_sha": SOURCE_COMMIT,
        "coverage_verdict": "PARTIAL_OR_GAPPED_HISTORY",
        "csv_sha256": csv_hash,
        "manifest_sha256": manifest_hash,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "observed_candle_count": 70169,
        "inter_observation_gap_count": 1,
        "missing_hours_between_observations": 320,
    }
    if any(report.get(key) != value for key, value in report_expected.items()):
        raise StageBError("source_report_contract_mismatch")


def parse_source_frame(csv_path: Path, raw_bytes: bytes) -> tuple[DataFrame, list[str]]:
    lines = raw_bytes.decode("utf-8").splitlines(keepends=True)
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, ValueError, pd.errors.ParserError):
        raise StageBError("source_csv_parse_failed") from None
    required = ("timestamp", "date", "open", "high", "low", "close", "volume")
    if tuple(frame.columns) != required or len(frame) + 1 != len(lines):
        raise StageBError("source_csv_shape_mismatch")
    if not frame["date"].astype(str).str.endswith("Z").all():
        raise StageBError("source_timestamp_timezone_missing")
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    if timestamps.isna().any() or not np.equal(timestamps, np.floor(timestamps)).all():
        raise StageBError("source_timestamp_invalid")
    frame["timestamp"] = timestamps.astype("int64")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    epoch = (frame["date"].astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64)
    if not np.array_equal(epoch, frame["timestamp"].to_numpy(dtype=np.int64)):
        raise StageBError("source_timestamp_epoch_mismatch")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, lines


def validate_source_ohlcv(frame: DataFrame) -> None:
    columns = ["open", "high", "low", "close", "volume"]
    if not np.isfinite(frame.loc[:, columns].to_numpy(dtype=float)).all():
        raise StageBError("source_ohlcv_non_finite")
    if (frame.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise StageBError("source_price_not_positive")
    if (frame["volume"] < 0).any():
        raise StageBError("source_volume_negative")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise StageBError("source_high_below_open_or_close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise StageBError("source_low_above_open_or_close")
    if (frame["high"] < frame["low"]).any():
        raise StageBError("source_high_below_low")


def read_coverage_artifact(
    input_dir: Path,
) -> tuple[Path, bytes, dict[str, Any], dict[str, Any]]:
    """Read and authenticate the immutable Stage A files before parsing candles."""
    csv_path = input_dir / CSV_NAME
    manifest_path = input_dir / MANIFEST_NAME
    report_path = input_dir / REPORT_NAME
    if not all(path.is_file() for path in (csv_path, manifest_path, report_path)):
        raise StageBError("pinned_stage_a_artifact_files_missing")
    raw_bytes = csv_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    csv_hash = hashlib.sha256(raw_bytes).hexdigest()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    check_artifact_metadata(manifest, report, csv_hash, manifest_hash)
    return csv_path, raw_bytes, manifest, report


def validate_coverage_frame(
    csv_path: Path, raw_bytes: bytes
) -> tuple[DataFrame, list[str]]:
    """Parse the pinned CSV and validate rows without sorting or repairing data."""
    frame, lines = parse_source_frame(csv_path, raw_bytes)
    validate_source_ohlcv(frame)
    if len(frame) != 70169:
        raise StageBError("source_candle_count_unexpected")
    return frame, lines


def split_verified_segments(
    frame: DataFrame, lines: list[str], report: dict[str, Any]
) -> tuple[list[DataFrame], list[str]]:
    """Verify exact segment boundaries and hash each untouched source-row range."""
    ranges, gaps = segment_bounds(frame["timestamp"].to_numpy(dtype=np.int64).tolist())
    if len(ranges) != 2 or len(gaps) != 1 or gaps[0]["missing_hours"] != 320:
        raise StageBError(
            "source_must_contain_exactly_two_segments_and_one_320_hour_gap"
        )
    expected = report.get("contiguous_segments")
    if not isinstance(expected, list) or len(expected) != 2:
        raise StageBError("source_segment_manifest_invalid")

    segments: list[DataFrame] = []
    hashes: list[str] = []
    for number, (start, stop) in enumerate(ranges):
        segment = frame.iloc[start:stop].copy().reset_index(drop=True)
        row_bytes = "".join(lines[start + 1 : stop + 1]).encode("utf-8")
        hashes.append(hashlib.sha256(row_bytes).hexdigest())
        actual = {
            "candle_count": len(segment),
            "start_utc": utc_text(segment["date"].iloc[0]),
            "end_utc": utc_text(segment["date"].iloc[-1]),
        }
        for key, value in actual.items():
            if expected[number].get(key) != value:
                raise StageBError("source_segment_boundary_mismatch")
        segments.append(segment)
    return segments, hashes


def verify_source_artifact(
    input_dir: Path,
) -> tuple[DataFrame, dict[str, Any], list[DataFrame], list[str]]:
    """Authenticate Stage A provenance and return only its observed segments."""
    csv_path, raw_bytes, _manifest, report = read_coverage_artifact(input_dir)
    frame, lines = validate_coverage_frame(csv_path, raw_bytes)
    segments, hashes = split_verified_segments(frame, lines, report)
    return frame, report, segments, hashes


def build_fixed_window_folds(
    stop_index: int,
    initial_train_rows: int,
    horizon: int,
    window_rows: int,
    prefix: str,
) -> list[FoldSpec]:
    """Build expanding training folds with a horizon-sized label purge."""
    if min(stop_index, initial_train_rows, horizon, window_rows) <= 0:
        raise StageBError("walk_forward_configuration_invalid")
    first_test = initial_train_rows + horizon
    if stop_index - first_test <= horizon:
        raise StageBError("insufficient_rows_for_walk_forward")
    result: list[FoldSpec] = []
    test_start = first_test
    number = 1
    while test_start < stop_index:
        test_stop = min(test_start + window_rows, stop_index)
        if test_stop - test_start <= horizon:
            break
        result.append(
            FoldSpec(
                f"{prefix}-{number:02d}", test_start - horizon, test_start, test_stop
            )
        )
        test_start += window_rows
        number += 1
    if not result:
        raise StageBError("walk_forward_produced_no_folds")
    return result


def economic_gate(
    trade_count: int,
    win_rate: float | None,
    expectancy: float | None,
    drawdown_abs: float,
) -> dict[str, Any]:
    """Strict thresholds: 100+, win rate >60%, expectancy >0, drawdown <15%."""
    count_pass = trade_count >= 100
    win_pass = win_rate is not None and win_rate > 0.60
    expectancy_pass = expectancy is not None and expectancy > 0.0
    drawdown_pass = drawdown_abs < 0.15
    passed = count_pass and win_pass and expectancy_pass and drawdown_pass
    status = (
        "INSUFFICIENT_SAMPLE"
        if not count_pass
        else "THRESHOLDS_MET_RESEARCH_ONLY"
        if passed
        else "NO_GO_THRESHOLDS_NOT_MET"
    )
    return {
        "required_trade_count": ">= 100",
        "required_win_rate": "> 0.60",
        "required_net_expectancy_per_trade": "> 0",
        "required_absolute_max_drawdown": "< 0.15",
        "trade_count_pass": count_pass,
        "win_rate_pass": win_pass,
        "expectancy_pass": expectancy_pass,
        "drawdown_pass": drawdown_pass,
        "status": status,
        "authorization_effect": "NONE; Champion remains null and trading stays NO-GO",
    }


def drawdown_abs(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        if equity <= 0.0:
            raise StageBError("net_return_below_full_capital_loss_contract")
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return abs(worst)


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(trade["net_return"]) for trade in trades]
    count = len(returns)
    wins = sum(value > 0.0 for value in returns)
    rate = wins / count if count else None
    expectancy = float(np.mean(returns)) if count else None
    drawdown = drawdown_abs(returns)
    return {
        "trade_count": count,
        "winning_trade_count": wins,
        "losing_or_flat_trade_count": count - wins,
        "win_rate": rate,
        "expectancy_net_per_trade": expectancy,
        "net_return_compounded": float(np.prod(1.0 + np.asarray(returns)) - 1.0)
        if returns
        else 0.0,
        "max_drawdown_abs": drawdown,
        "fee_bps_per_side": FEE_BPS_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_SIDE,
        "round_trip_cost_fraction": ROUND_TRIP_COST,
        "economic_threshold_evaluation": economic_gate(
            count, rate, expectancy, drawdown
        ),
    }


def fit_new_model() -> Any:
    """Do not reuse estimator state across folds or segments."""
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

    return LGBMRegressor(**dict(BASELINE_MODEL_PARAMETERS))


def fit_fold_model(train: DataFrame, feature_columns: list[str]) -> Any:
    """Fit one new model using only this segment's available past-only features."""
    if len(train) < 1 or not feature_columns:
        raise StageBError("fold_training_rows_or_features_empty")
    train_y = train[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(train_y).all():
        raise StageBError("training_target_non_finite")
    model = fit_new_model()
    model.fit(train.loc[:, feature_columns], train_y)
    return model


def predict_fold(
    model: Any,
    test: DataFrame,
    feature_columns: list[str],
    terminal_fold: bool,
) -> tuple[DataFrame, np.ndarray, np.ndarray]:
    """Predict OOS labels, excluding rows whose full horizon would cross a fold."""
    scored_count = len(test) if terminal_fold else len(test) - HORIZON
    scored = test.iloc[:scored_count]
    if scored.empty:
        raise StageBError("fold_has_no_scored_oos_rows")
    predicted = np.asarray(model.predict(scored.loc[:, feature_columns]), dtype=float)
    actual = scored[TARGET_COLUMN].to_numpy(dtype=float)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise StageBError("oos_prediction_invalid")
    if not np.isfinite(actual).all():
        raise StageBError("oos_target_non_finite")
    return scored, predicted, actual


def build_trade_record(
    segment: str,
    evaluation: str,
    fold_id: str,
    entry: pd.Timestamp,
    forecast: float,
    realized: float,
    opportunity: bool,
) -> dict[str, Any] | None:
    """Return a costed research trade only when the pre-registered entry rule passes."""
    if not opportunity or forecast <= ROUND_TRIP_COST:
        return None
    net = float(realized - ROUND_TRIP_COST)
    if net <= -1.0:
        raise StageBError("net_return_below_full_capital_loss_contract")
    return {
        "segment": segment,
        "evaluation": evaluation,
        "fold_id": fold_id,
        "entry_utc": utc_text(entry),
        "exit_utc": utc_text(entry + pd.Timedelta(hours=HORIZON)),
        "horizon_hours": HORIZON,
        "forecast_return": float(forecast),
        "gross_return": float(realized),
        "fee_bps_round_trip": 2 * FEE_BPS_SIDE,
        "slippage_bps_round_trip": 2 * SLIPPAGE_BPS_SIDE,
        "net_return": net,
        "won_after_costs": net > 0.0,
        "order_submitted": False,
    }


def build_oos_rows(
    segment: str,
    evaluation: str,
    fold: FoldSpec,
    test: DataFrame,
    scored: DataFrame,
    predicted: np.ndarray,
    actual: np.ndarray,
    terminal_fold: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create OOS forecast rows and non-overlapping research trades."""
    prediction_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for offset, (forecast, realized) in enumerate(
        zip(predicted.tolist(), actual.tolist(), strict=True)
    ):
        entry = pd.Timestamp(scored["date"].iloc[offset])
        opportunity = offset % HORIZON == 0 and (
            terminal_fold or offset + HORIZON < len(test)
        )
        prediction_rows.append(
            {
                "segment": segment,
                "evaluation": evaluation,
                "fold_id": fold.fold_id,
                "timestamp_utc": utc_text(entry),
                "forecast_return_horizon": forecast,
                "realized_forward_return": realized,
                "round_trip_cost_fraction": ROUND_TRIP_COST,
                "trade_opportunity": opportunity,
                "trade_taken": opportunity and forecast > ROUND_TRIP_COST,
            }
        )
        trade = build_trade_record(
            segment, evaluation, fold.fold_id, entry, forecast, realized, opportunity
        )
        if trade is not None:
            trades.append(trade)
    return prediction_rows, trades


def evaluate_fold(
    segment: str,
    evaluation: str,
    dataset: DataFrame,
    fold: FoldSpec,
    terminal_fold: bool,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Fit one fold, score OOS rows, and emit the fold's research evidence."""
    train = dataset.iloc[: fold.train_stop]
    test = dataset.iloc[fold.test_start : fold.test_stop]
    model = fit_fold_model(train, feature_columns)
    scored, predicted, actual = predict_fold(
        model, test, feature_columns, terminal_fold
    )
    prediction_rows, trades = build_oos_rows(
        segment, evaluation, fold, test, scored, predicted, actual, terminal_fold
    )
    return {
        "segment": segment,
        "evaluation": evaluation,
        "fold_id": fold.fold_id,
        "terminal_fold": terminal_fold,
        "train_rows": len(train),
        "test_rows_available": len(test),
        "oos_rows_scored": len(scored),
        "purged_label_rows_before_test": HORIZON,
        "train_start_utc": utc_text(train["date"].iloc[0]),
        "train_end_utc": utc_text(train["date"].iloc[-1]),
        "test_start_utc": utc_text(test["date"].iloc[0]),
        "test_end_utc": utc_text(test["date"].iloc[-1]),
        "fresh_model_per_fold": True,
        "economic_metrics": summarize_trades(trades),
        "forecast_metrics": {
            "mae": float(np.mean(np.abs(predicted - actual))),
            "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
            "directional_accuracy": float(
                np.mean(np.sign(predicted) == np.sign(actual))
            ),
        },
        "prediction_rows": prediction_rows,
        "trades": trades,
    }


def select_segment_features(
    engineered: DataFrame, candidates: Sequence[str] = FEATURE_COLUMNS
) -> tuple[list[str], list[dict[str, str]]]:
    """Exclude features undefined on usable rows; never fill or repair feature data."""
    selected = [
        column
        for column in candidates
        if engineered[column].replace([np.inf, -np.inf], np.nan).notna().any()
    ]
    excluded: list[dict[str, str]] = [
        {"feature": column, "reason": "undefined_for_all_segment_rows"}
        for column in candidates
        if column not in selected
    ]
    changed = True
    while changed:
        changed = False
        for candidate in list(selected):
            other_columns = [column for column in selected if column != candidate]
            if not other_columns:
                continue
            usable_rows = (
                engineered.loc[:, other_columns]
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .all(axis=1)
            )
            candidate_valid = (
                engineered[candidate].replace([np.inf, -np.inf], np.nan).notna()
            )
            if usable_rows.any() and not candidate_valid.loc[usable_rows].all():
                selected.remove(candidate)
                excluded.append(
                    {
                        "feature": candidate,
                        "reason": "undefined_on_otherwise_usable_rows",
                    }
                )
                changed = True
                break
    if not selected:
        raise StageBError("segment_has_no_usable_features")
    return selected, excluded


def prepare_segment(
    segment: DataFrame, name: str
) -> tuple[DataFrame, str, list[str], list[dict[str, str]]]:
    """Build segment-local feature/target rows without imputing unavailable values."""
    candles = segment.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
    validated, fingerprint = validate_ohlcv(candles, HOUR)
    engineered = add_market_features(validated)
    engineered[TARGET_COLUMN] = forward_return_target(engineered, HORIZON)
    feature_columns, excluded_features = select_segment_features(engineered)
    required = feature_columns + [TARGET_COLUMN]
    numeric = engineered.loc[:, required].replace([np.inf, -np.inf], np.nan)
    valid_rows = numeric.notna().all(axis=1)
    selected_columns = list(dict.fromkeys(["date", "close", *required]))
    dataset = engineered.loc[valid_rows, selected_columns].copy().reset_index(drop=True)
    if dataset.empty:
        raise StageBError(f"segment_{name}_dataset_empty")
    dates = pd.to_datetime(dataset["date"], utc=True, errors="raise")
    if not (dates.diff().dropna() == pd.Timedelta(seconds=HOUR)).all():
        raise StageBError(f"segment_{name}_feature_rows_not_contiguous")
    features = dataset.loc[:, feature_columns].to_numpy(dtype=float)
    if (
        not np.isfinite(features).all()
        or not np.isfinite(dataset[TARGET_COLUMN].to_numpy(dtype=float)).all()
    ):
        raise StageBError(f"segment_{name}_feature_or_target_non_finite")
    return dataset, fingerprint, feature_columns, excluded_features


def evaluate_many(
    segment: str,
    evaluation: str,
    dataset: DataFrame,
    folds: list[FoldSpec],
    out_dir: Path,
    completed: int,
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    for fold in folds:
        result = evaluate_fold(
            segment,
            evaluation,
            dataset,
            fold,
            fold.test_stop == len(dataset),
            feature_columns,
        )
        results.append(result)
        completed += 1
        write_json(
            out_dir / "progress.json",
            {
                "status": "IN_PROGRESS",
                "completed_folds": completed,
                "last_completed_fold": fold.fold_id,
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "economic_acceptance": "NO-GO",
            },
        )
    return results, completed


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    trades = [trade for result in results for trade in result["trades"]]
    return {
        "fold_count": len(results),
        "metrics": summarize_trades(trades),
        "folds": [
            {
                key: value
                for key, value in result.items()
                if key not in ("prediction_rows", "trades")
            }
            for result in results
        ],
        "trades": trades,
        "prediction_rows": [
            row for result in results for row in result["prediction_rows"]
        ],
    }


def installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], empty_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else empty_fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_stage_b(input_dir: Path, output_dir: Path, commit_sha: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, source_report, segments, segment_hashes = verify_source_artifact(input_dir)
    segment_a, segment_b = segments
    dataset_a, fingerprint_a, features_a, excluded_a = prepare_segment(segment_a, "A")
    dataset_b, fingerprint_b, features_b, excluded_b = prepare_segment(segment_b, "B")
    holdout_start = pd.Timestamp(segment_a["date"].iloc[-A_HOLDOUT])
    a_dev_stop = int((dataset_a["date"] < holdout_start).sum())
    if a_dev_stop <= A_TRAIN + HORIZON:
        raise StageBError("segment_a_development_rows_insufficient")
    if dataset_a["date"].iloc[a_dev_stop] != holdout_start:
        raise StageBError("segment_a_holdout_boundary_not_exact")

    a_dev_folds = build_fixed_window_folds(a_dev_stop, A_TRAIN, HORIZON, WINDOW, "A-WF")
    holdout_fold = FoldSpec(
        "A-HOLDOUT-90D", a_dev_stop - HORIZON, a_dev_stop, len(dataset_a)
    )
    if holdout_fold.train_stop < A_TRAIN:
        raise StageBError("segment_a_holdout_training_rows_insufficient")
    b_folds = build_fixed_window_folds(
        len(dataset_b), B_TRAIN, HORIZON, WINDOW, "B-DIAG"
    )
    write_json(
        output_dir / "input-verification.json",
        {
            "coverage_workflow_run_id": SOURCE_RUN_ID,
            "coverage_artifact_name": SOURCE_ARTIFACT,
            "coverage_commit_sha": SOURCE_COMMIT,
            "stage_b_code_commit_sha": commit_sha,
            "csv_sha256": source_report["csv_sha256"],
            "manifest_sha256": source_report["manifest_sha256"],
            "observed_candles": len(frame),
            "gap_hours": 320,
            "segment_a_sha256": segment_hashes[0],
            "segment_b_sha256": segment_hashes[1],
            "interpolation": False,
            "forward_fill": False,
            "backfill": False,
            "resampling": False,
        },
    )

    a_results, completed = evaluate_many(
        "A",
        "walk_forward_development",
        dataset_a,
        a_dev_folds,
        output_dir,
        0,
        features_a,
    )
    b_results, completed = evaluate_many(
        "B",
        "independent_short_history_diagnostic",
        dataset_b,
        b_folds,
        output_dir,
        completed,
        features_b,
    )
    # Final holdout is evaluated last and never used for model/parameter selection.
    holdout_results, completed = evaluate_many(
        "A",
        "untouched_final_holdout",
        dataset_a,
        [holdout_fold],
        output_dir,
        completed,
        features_a,
    )
    a_dev = aggregate(a_results)
    b_diag = aggregate(b_results)
    holdout = aggregate(holdout_results)
    all_results = a_results + b_results + holdout_results
    write_csv(
        output_dir / "oos-predictions.csv",
        [row for result in all_results for row in result["prediction_rows"]],
        [
            "segment",
            "evaluation",
            "fold_id",
            "timestamp_utc",
            "forecast_return_horizon",
            "realized_forward_return",
            "round_trip_cost_fraction",
            "trade_opportunity",
            "trade_taken",
        ],
    )
    write_csv(
        output_dir / "oos-trades.csv",
        a_dev["trades"] + holdout["trades"] + b_diag["trades"],
        [
            "segment",
            "evaluation",
            "fold_id",
            "entry_utc",
            "exit_utc",
            "horizon_hours",
            "forecast_return",
            "gross_return",
            "fee_bps_round_trip",
            "slippage_bps_round_trip",
            "net_return",
            "won_after_costs",
            "order_submitted",
        ],
    )

    details: dict[str, Any] = {}
    for name, segment, dataset, fingerprint, segment_hash, features, excluded in (
        (
            "A",
            segment_a,
            dataset_a,
            fingerprint_a,
            segment_hashes[0],
            features_a,
            excluded_a,
        ),
        (
            "B",
            segment_b,
            dataset_b,
            fingerprint_b,
            segment_hashes[1],
            features_b,
            excluded_b,
        ),
    ):
        details[name] = {
            "input_candle_count": len(segment),
            "valid_feature_target_rows": len(dataset),
            "start_utc": utc_text(segment["date"].iloc[0]),
            "end_utc": utc_text(segment["date"].iloc[-1]),
            "segment_csv_rows_sha256": segment_hash,
            "data_quality_gate_fingerprint": fingerprint,
            "model_feature_columns": features,
            "excluded_features_without_imputation": excluded,
            "fresh_state_within_segment": True,
            "state_shared_with_other_segment": False,
        }
    report = {
        "report_schema": REPORT_SCHEMA,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": SOURCE_RUN_ID,
            "coverage_artifact_name": SOURCE_ARTIFACT,
            "coverage_commit_sha": SOURCE_COMMIT,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "csv_sha256": source_report["csv_sha256"],
            "manifest_sha256": source_report["manifest_sha256"],
            "symbol": "BTCUSDT",
            "timeframe": "1h",
        },
        "methodology": {
            "model": "pinned Phase 4 LightGBM LGBMRegressor",
            "parameters": dict(BASELINE_MODEL_PARAMETERS),
            "fresh_model_per_fold": True,
            "feature_policy": "existing past-only features are computed per segment; features undefined on otherwise usable rows are excluded and recorded, never imputed",
            "target": "12-hour forward close-to-close return",
            "entry_rule": "take a fixed-horizon long-only research trade only when predicted return exceeds modeled round-trip costs",
            "exit_rule": "12 hours, non-overlapping; development trades cannot cross fold boundaries",
            "fee_bps_per_side": FEE_BPS_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_SIDE,
            "minimum_training_rows_A": A_TRAIN,
            "minimum_training_rows_B_diagnostic": B_TRAIN,
            "oos_window_hours": WINDOW,
            "purge_hours": HORIZON,
            "A_holdout_hours": A_HOLDOUT,
            "partial_final_windows_reported": True,
            "hyperparameter_search": False,
            "holdout_used_for_model_selection": False,
            "model_selection_using_B": False,
            "capital_model": "unit-notional compounded trade returns; not position sizing or Risk Gate execution",
            "strategy_backtest_boundary": "research baseline only; not a full Freqtrade strategy or exchange execution backtest",
        },
        "dataset_quality": {
            "coverage_verdict": source_report["coverage_verdict"],
            "expected_hourly_slots": source_report["expected_hourly_slots"],
            "observed_candle_count": source_report["observed_candle_count"],
            "missing_internal_hours": 320,
            "interpolation": False,
            "forward_fill": False,
            "backfill": False,
            "resampling": False,
            "segments_validated_independently": True,
            "segment_specific_feature_availability": True,
            "feature_imputation_performed": False,
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": installed_version("lightgbm"),
        },
        "segments": details,
        "evaluations": {
            "segment_a_walk_forward_development": {
                "training_policy": "expanding; minimum 3 years; 12-hour label purge",
                "result": a_dev,
            },
            "segment_a_final_holdout": {
                "boundary_start_utc": utc_text(holdout_start),
                "reserved_hours": A_HOLDOUT,
                "training_policy": "past A only; 12-hour label purge; evaluated last",
                "used_for_model_or_parameter_selection": False,
                "result": holdout,
            },
            "segment_b_independent_diagnostic": {
                "training_policy": "expanding; 180-day segment-local warm-up; 12-hour purge",
                "not_comparable_to_three_year_A_gate": True,
                "result": b_diag,
            },
        },
        "strict_economic_acceptance": {
            "criteria": {
                "oos_trade_count": ">= 100",
                "win_rate": "> 60%",
                "net_expectancy_per_trade": "> 0 after modeled costs",
                "absolute_max_drawdown": "< 15%",
            },
            "applied_separately_to": [
                "segment_a_walk_forward_development",
                "segment_a_final_holdout",
                "segment_b_independent_diagnostic",
            ],
            "aggregate_across_segments": False,
            "threshold_relaxation": False,
        },
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "economic_acceptance": "NO-GO",
            "model_training_performed": True,
            "model_promoted": False,
            "production_model_updated": False,
            "strategy_registry_updated": False,
            "orders_or_execution_api_used": False,
            "private_api_calls": 0,
            "order_endpoint_calls": 0,
            "real_order_attempts": 0,
        },
    }
    write_json(output_dir / "stage-b-report.json", report)
    write_json(
        output_dir / "progress.json",
        {
            "status": "COMPLETED",
            "completed_folds": completed,
            "planned_folds": completed,
            "holdout_evaluated": True,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "economic_acceptance": "NO-GO",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    args = parser.parse_args()
    try:
        report = run_stage_b(args.input_dir, args.output_dir, args.commit_sha)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "stage-b-failure.json",
            {
                "report_schema": REPORT_SCHEMA,
                "run_status": "FAILED_CLOSED",
                "failure_type": type(error).__name__,
                "failure_code": str(error),
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "economic_acceptance": "NO-GO",
                "orders_or_execution_api_used": False,
            },
        )
        raise
    print(
        json.dumps(
            {
                "run_status": report["run_status"],
                "segment_a_walk_forward": report["evaluations"][
                    "segment_a_walk_forward_development"
                ]["result"]["metrics"],
                "segment_a_holdout": report["evaluations"]["segment_a_final_holdout"][
                    "result"
                ]["metrics"],
                "segment_b_diagnostic": report["evaluations"][
                    "segment_b_independent_diagnostic"
                ]["result"]["metrics"],
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
