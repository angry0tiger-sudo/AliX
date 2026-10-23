"""Run public-data, purged walk-forward Meta-Label research without promotion."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from services.data_quality.gate import validate_candles
from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    add_market_features,
    forward_return_target,
)
from services.meta_labeling.meta_model import (
    MetaLabelModel,
    OOFFold,
    generate_oof_primary_predictions,
)
from services.meta_labeling.triple_barrier import (
    TripleBarrierConfig,
    apply_triple_barrier,
)

REPORT_SCHEMA = "alix.meta-label-economic-experiments.v1"
PUBLIC_PROVIDER = "Nobitex public UDF history API"
RESEARCH_THRESHOLD = 0.70
EXPECTED_ROWS = 8760
TRAINING_START = 1800
FOLD_WIDTH = 600
MAX_HOLDING_BARS = 12
FEE_BPS_PER_SIDE = 25.0
SLIPPAGE_BPS_PER_SIDE = 5.0
MIN_PRIOR_TRADES_FOR_SELECTION = 30
MIN_OOS_TRADES_FOR_ECONOMIC_EVALUATION = 100
MIN_META_TRAIN_ROWS = 100
CLOSED_FLAGS = {
    "economic_acceptance": "NO-GO",
    "order_permission": False,
    "trading_authorization": "NO-GO",
}
PRIMARY_CONFIGS: tuple[tuple[str, float, bool], ...] = (
    ("current_unscaled_c1", 1.0, False),
    ("scaled_c0_1", 0.1, True),
    ("scaled_c1", 1.0, True),
    ("scaled_c10", 10.0, True),
)
META_C_VALUES = (0.1, 1.0, 10.0)
FEATURE_GROUPS = ("baseline", "atr_volume", "atr_volume_trend")
MODEL_FEATURES = "model_features"


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"experiment_json_unreadable:{path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"experiment_json_root_invalid:{path.name}")
    return value


def _validate_manifest_flags(manifest: dict[str, Any]) -> None:
    expected: dict[str, object] = {
        "provider": PUBLIC_PROVIDER,
        "requested_days": 365,
        "row_count": EXPECTED_ROWS,
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        **CLOSED_FLAGS,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted:
            raise ValueError(f"experiment_public_manifest_contract_failed:{field}")


def _validate_ohlcv(candles: DataFrame) -> None:
    required = ("date", "open", "high", "low", "close", "volume")
    if tuple(candles.columns) != required:
        raise ValueError("experiment_ohlcv_schema_invalid")
    if len(candles) != EXPECTED_ROWS:
        raise ValueError("experiment_dataset_rows_invalid")
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="raise")
    if candles["date"].isna().any() or candles["date"].duplicated().any():
        raise ValueError("experiment_timestamps_invalid")
    if not candles["date"].is_monotonic_increasing:
        raise ValueError("experiment_timestamps_not_increasing")
    differences = (
        candles["date"].diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    )
    if not np.all(differences == 3600.0):
        raise ValueError("experiment_hourly_grid_invalid")
    numeric = candles.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("experiment_ohlcv_non_finite")
    opened = numeric["open"].to_numpy(dtype=float)
    high = numeric["high"].to_numpy(dtype=float)
    low = numeric["low"].to_numpy(dtype=float)
    close = numeric["close"].to_numpy(dtype=float)
    volume = numeric["volume"].to_numpy(dtype=float)
    if (
        np.any(np.minimum.reduce([opened, high, low, close]) <= 0)
        or np.any(volume < 0)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
        or np.any(high < low)
    ):
        raise ValueError("experiment_ohlcv_geometry_invalid")
    candles.loc[:, ["open", "high", "low", "close", "volume"]] = numeric


def _load_dataset(
    csv_path: Path, manifest_path: Path, commit_sha: str
) -> tuple[DataFrame, dict[str, Any], str, str]:
    manifest = _read_object(manifest_path)
    _validate_manifest_flags(manifest)
    if manifest.get("commit_sha") != commit_sha:
        raise ValueError("experiment_manifest_commit_sha_mismatch")
    csv_bytes = csv_path.read_bytes()
    data_hash = _sha256_bytes(csv_bytes)
    if manifest.get("csv_sha256") != data_hash:
        raise ValueError("experiment_dataset_sha256_mismatch")
    candles = pd.read_csv(csv_path)
    _validate_ohlcv(candles)
    quality_rows = [
        {str(key): value for key, value in row.items()}
        for row in candles.to_dict(orient="records")
    ]
    quality = validate_candles(
        quality_rows,
        expected_interval_seconds=3600,
        gap_policy="reject",
    )
    if not quality.passed or quality.fingerprint is None:
        raise ValueError("experiment_shared_data_quality_gate_rejected_dataset")
    return candles, manifest, data_hash, quality.fingerprint


def _engineer_experiment_features(candles: DataFrame) -> DataFrame:
    result = add_market_features(candles)
    close = pd.to_numeric(candles["close"], errors="coerce")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    volume = pd.to_numeric(candles["volume"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    returns = close.pct_change(fill_method=None)
    log_volume = np.log1p(volume)
    for window in (7, 14, 28):
        result[f"%-atr_norm_{window}"] = (
            true_range.rolling(window, min_periods=window).mean() / close
        )
    for window in (24, 72):
        result[f"%-volatility_{window}"] = returns.rolling(
            window, min_periods=window
        ).std()
        volume_mean = log_volume.rolling(window, min_periods=window).mean()
        volume_std = log_volume.rolling(window, min_periods=window).std()
        result[f"%-volume_z_{window}"] = (
            log_volume - volume_mean
        ) / volume_std.replace(0.0, np.nan)
    volume_median = volume.rolling(24, min_periods=24).median()
    result["%-volume_ratio_24"] = volume / volume_median.replace(0.0, np.nan) - 1.0
    for window in (24, 72):
        rolling_mean = close.rolling(window, min_periods=window).mean()
        result[f"%-close_vs_sma_{window}"] = close / rolling_mean - 1.0
    for window in (12, 24):
        result[f"%-momentum_{window}"] = close.pct_change(window, fill_method=None)
    rolling_high = high.rolling(24, min_periods=24).max()
    rolling_low = low.rolling(24, min_periods=24).min()
    width = (rolling_high - rolling_low).replace(0.0, np.nan)
    result["%-range_position_24"] = (close - rolling_low) / width
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def _feature_names(frame: DataFrame, group: str) -> list[str]:
    names = list(FEATURE_COLUMNS)
    if group in {"atr_volume", "atr_volume_trend"}:
        names.extend(
            [
                "%-atr_norm_7",
                "%-atr_norm_14",
                "%-atr_norm_28",
                "%-volatility_24",
                "%-volatility_72",
                "%-volume_z_24",
                "%-volume_z_72",
                "%-volume_ratio_24",
            ]
        )
    if group == "atr_volume_trend":
        names.extend(
            [
                "%-close_vs_sma_24",
                "%-close_vs_sma_72",
                "%-momentum_12",
                "%-momentum_24",
                "%-range_position_24",
            ]
        )
    if group not in FEATURE_GROUPS or not set(names).issubset(frame.columns):
        raise ValueError(f"experiment_feature_group_invalid:{group}")
    return names


def _make_folds(row_count: int) -> list[OOFFold]:
    folds = [
        OOFFold(
            0,
            start,
            start,
            min(start + FOLD_WIDTH, row_count - 1),
        )
        for start in range(TRAINING_START, row_count - 100, FOLD_WIDTH)
    ]
    if len(folds) < 8:
        raise ValueError("experiment_walkforward_fold_count_insufficient")
    return folds


def _primary_estimator(name: str, regularization_c: float, standardized: bool) -> Any:
    classifier = LogisticRegression(
        C=regularization_c,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    if name == "current_unscaled_c1" and not standardized:
        return classifier
    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])


def _build_oof_and_labels(
    candles: DataFrame,
    feature_frame: DataFrame,
    feature_names: list[str],
    primary_config: tuple[str, float, bool],
    folds: list[OOFFold],
) -> tuple[DataFrame, DataFrame]:
    feature_values = feature_frame.loc[:, feature_names].copy()
    forward = forward_return_target(candles, horizon=1).to_numpy(dtype=float)
    primary_values = np.where(
        np.isfinite(forward),
        (forward > 0.0).astype(float),
        np.nan,
    )
    oof = generate_oof_primary_predictions(
        feature_values,
        pd.Series(primary_values),
        np.arange(len(candles), dtype=int) + 1,
        folds,
        lambda: _primary_estimator(
            primary_config[0], primary_config[1], primary_config[2]
        ),
    )
    event_indices = oof["event_index"].to_numpy(dtype=int)
    volatility = (
        feature_values["%-volatility_12"].iloc[event_indices].to_numpy(dtype=float)
    )
    eligible = (
        (oof["primary_prediction"].to_numpy(dtype=float) >= 0.5)
        & np.isfinite(volatility)
        & (volatility > 0.0)
        & (event_indices + MAX_HOLDING_BARS < len(candles))
    )
    event_indices = event_indices[eligible]
    volatility = volatility[eligible]
    if len(event_indices) == 0:
        raise ValueError("experiment_no_primary_long_events")
    events = DataFrame(
        {
            "event_index": event_indices,
            "side": np.ones(len(event_indices), dtype=int),
            "target_volatility": volatility,
        }
    )
    labels = apply_triple_barrier(
        candles,
        events,
        TripleBarrierConfig(
            profit_taking_multiplier=1.5,
            stop_loss_multiplier=1.0,
            max_holding_bars=MAX_HOLDING_BARS,
            fee_bps_per_side=FEE_BPS_PER_SIDE,
            slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
            timeframe_seconds=3600,
        ),
    )
    merged = oof.merge(labels, on="event_index", how="inner", validate="one_to_one")
    if merged.empty or merged["meta_label"].nunique() < 2:
        raise ValueError("experiment_meta_labels_empty_or_single_class")
    return merged.sort_values("event_index").reset_index(drop=True), feature_values


def _max_abs_drawdown(returns: list[float]) -> float | None:
    if not returns:
        return None
    if any(not math.isfinite(value) or value <= -1.0 for value in returns):
        return 1.0
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdowns = (peaks - equity) / peaks
    return float(np.max(drawdowns))


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(item["net_return"]) for item in trades]
    count = len(returns)
    positives = [value for value in returns if value > 0.0]
    negatives = [value for value in returns if value < 0.0]
    losses = abs(sum(negatives))
    return {
        "trade_count": count,
        "win_rate": float(len(positives) / count) if count else None,
        "net_expectancy": float(np.mean(returns)) if count else None,
        "total_net_return_simple_sum": float(sum(returns)) if count else None,
        "profit_factor": (
            float(sum(positives) / losses) if count and losses > 0.0 else None
        ),
        "max_abs_drawdown": _max_abs_drawdown(returns),
        "selected_trades": trades,
    }


def _non_overlapping_trades(
    rows: DataFrame, probabilities: np.ndarray, fold_id: int
) -> list[dict[str, Any]]:
    selected = rows.loc[probabilities >= RESEARCH_THRESHOLD].copy()
    selected["meta_probability"] = probabilities[probabilities >= RESEARCH_THRESHOLD]
    trades: list[dict[str, Any]] = []
    last_exit = -1
    for row in selected.sort_values("event_index").to_dict(orient="records"):
        event_index = int(row["event_index"])
        exit_index = int(row["label_end_index"])
        if event_index <= last_exit:
            continue
        last_exit = exit_index
        trades.append(
            {
                "fold_id": fold_id,
                "event_index": event_index,
                "label_end_index": exit_index,
                "meta_probability": float(row["meta_probability"]),
                "meta_label": int(row["meta_label"]),
                "net_return": float(row["net_return"]),
                "exit_reason": str(row["exit_reason"]),
            }
        )
    return trades


def _evaluate_one_fold(
    merged: DataFrame,
    feature_values: DataFrame,
    feature_names: list[str],
    fold_id: int,
    test_start: int,
    test_end: int,
    meta_c: float,
) -> dict[str, Any]:
    test_rows = merged.loc[
        (merged["fold_id"] == fold_id)
        & (merged["event_index"] < test_end)
        & (merged["label_end_index"] < test_end)
    ].copy()
    train_rows = merged.loc[
        (merged["fold_id"] < fold_id)
        & (merged["event_index"] < test_start)
        & (merged["label_end_index"] < test_start)
    ].copy()
    if (
        len(train_rows) < MIN_META_TRAIN_ROWS
        or train_rows["meta_label"].nunique() < 2
        or test_rows.empty
    ):
        return {
            "fold_id": fold_id,
            "status": "SKIPPED_INSUFFICIENT_MATURE_OOS_SAMPLE",
            "train_count": int(len(train_rows)),
            "test_count": int(len(test_rows)),
            "test_start_index": test_start,
            "test_end_exclusive": test_end,
            "selected_trades": [],
        }
    train_indices = train_rows["event_index"].to_numpy(dtype=int)
    test_indices = test_rows["event_index"].to_numpy(dtype=int)
    train_features = feature_values.iloc[train_indices].loc[:, feature_names].copy()
    test_features = feature_values.iloc[test_indices].loc[:, feature_names].copy()
    train_features.index = train_indices
    test_features.index = test_indices
    provenance_columns = (
        "event_index",
        "fold_id",
        "primary_train_max_index",
        "primary_train_max_label_end_index",
    )
    model = MetaLabelModel(regularization_c=meta_c, max_iter=1000)
    model.fit(
        train_features,
        train_rows["meta_label"].to_numpy(dtype=int),
        train_rows.loc[:, list(provenance_columns)].reset_index(drop=True),
        train_rows["label_end_index"].to_numpy(dtype=int),
        evaluation_cutoff_index=test_start,
    )
    probabilities = model.predict_probability(test_features)
    labels = test_rows["meta_label"].to_numpy(dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean(np.square(probabilities - labels)))
    log_loss = float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )
    base_probability = float(train_rows["meta_label"].mean())
    baseline_brier = float(np.mean(np.square(base_probability - labels)))
    baseline_clipped = float(np.clip(base_probability, 1e-12, 1.0 - 1e-12))
    baseline_log_loss = float(
        -np.mean(
            labels * math.log(baseline_clipped)
            + (1.0 - labels) * math.log1p(-baseline_clipped)
        )
    )
    trades = _non_overlapping_trades(test_rows, probabilities, fold_id)
    metrics = _trade_metrics(trades)
    return {
        "fold_id": fold_id,
        "status": "EVALUATED",
        "train_count": int(len(train_rows)),
        "test_count": int(len(test_rows)),
        "test_start_index": test_start,
        "test_end_exclusive": test_end,
        "brier_score": brier,
        "log_loss": log_loss,
        "baseline_brier_score": baseline_brier,
        "baseline_log_loss": baseline_log_loss,
        "brier_skill_vs_constant_baseline": (
            float(1.0 - brier / baseline_brier) if baseline_brier > 0 else None
        ),
        "log_loss_improvement_vs_constant_baseline": baseline_log_loss - log_loss,
        "probability_mean": float(np.mean(probabilities)),
        "probability_p95": float(np.quantile(probabilities, 0.95)),
        "raw_threshold_pass_count": int(np.sum(probabilities >= RESEARCH_THRESHOLD)),
        "all_proposal_mean_net_return": float(test_rows["net_return"].mean()),
        **metrics,
    }


def _candidate_id(feature_group: str, primary: str, meta_c: float) -> str:
    return f"{feature_group}|{primary}|meta_c={meta_c:g}"


def _make_candidate_results(
    candles: DataFrame,
    feature_frame: DataFrame,
    folds: list[OOFFold],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    barrier_end = len(candles)
    for feature_group in FEATURE_GROUPS:
        names = _feature_names(feature_frame, feature_group)
        for primary in PRIMARY_CONFIGS:
            merged, values = _build_oof_and_labels(
                candles, feature_frame, names, primary, folds
            )
            for meta_c in META_C_VALUES:
                identifier = _candidate_id(feature_group, primary[0], meta_c)
                fold_reports = []
                for fold_id, fold in enumerate(folds):
                    end = (
                        folds[fold_id + 1].test_start
                        if fold_id + 1 < len(folds)
                        else barrier_end
                    )
                    fold_reports.append(
                        _evaluate_one_fold(
                            merged,
                            values,
                            names,
                            fold_id,
                            fold.test_start,
                            end,
                            meta_c,
                        )
                    )
                results[identifier] = {
                    "configuration": {
                        "feature_group": feature_group,
                        "feature_names": names,
                        "primary_model": primary[0],
                        "primary_regularization_c": primary[1],
                        "primary_standardized": primary[2],
                        "meta_regularization_c": meta_c,
                        "probability_threshold": RESEARCH_THRESHOLD,
                    },
                    "folds": fold_reports,
                }
    return results


def _flatten_trades(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for fold in folds:
        if fold.get("status") == "EVALUATED":
            trades.extend(fold.get("selected_trades", []))
    return sorted(trades, key=lambda item: int(item["event_index"]))


def _aggregate_candidate(result: dict[str, Any]) -> dict[str, Any]:
    evaluated = [fold for fold in result["folds"] if fold.get("status") == "EVALUATED"]
    trades = _flatten_trades(evaluated)
    metrics = _trade_metrics(trades)
    sample_count = sum(int(fold["test_count"]) for fold in evaluated)
    brier = (
        float(
            np.average(
                [fold["brier_score"] for fold in evaluated],
                weights=[fold["test_count"] for fold in evaluated],
            )
        )
        if evaluated
        else None
    )
    baseline_brier = (
        float(
            np.average(
                [fold["baseline_brier_score"] for fold in evaluated],
                weights=[fold["test_count"] for fold in evaluated],
            )
        )
        if evaluated
        else None
    )
    return {
        "evaluated_fold_count": len(evaluated),
        "skipped_fold_count": len(result["folds"]) - len(evaluated),
        "oos_event_count": sample_count,
        "brier_score_weighted": brier,
        "baseline_brier_score_weighted": baseline_brier,
        "brier_skill_vs_baseline": (
            float(1.0 - brier / baseline_brier)
            if brier is not None and baseline_brier is not None and baseline_brier > 0
            else None
        ),
        **metrics,
    }


def _choose_configuration(
    candidate_results: dict[str, dict[str, Any]],
    fold_id: int,
    baseline_id: str,
) -> tuple[str, str, int]:
    qualified: list[tuple[tuple[float, float, float, int], str, int]] = []
    for identifier, result in candidate_results.items():
        prior = [
            row
            for row in result["folds"]
            if int(row["fold_id"]) < fold_id and row["status"] == "EVALUATED"
        ]
        trades = _flatten_trades(prior)
        if len(trades) < MIN_PRIOR_TRADES_FOR_SELECTION:
            continue
        returns = [float(trade["net_return"]) for trade in trades]
        win_rate = sum(value > 0.0 for value in returns) / len(returns)
        expectancy = float(np.mean(returns))
        if win_rate <= 0.60 or expectancy <= 0.0:
            continue
        brier = float(
            np.average(
                [row["brier_score"] for row in prior],
                weights=[row["test_count"] for row in prior],
            )
        )
        score = (expectancy, win_rate, -brier, len(trades))
        qualified.append((score, identifier, len(trades)))
    if not qualified:
        return (
            baseline_id,
            "fallback_no_prior_candidate_passed_trade_winrate_expectancy_gates",
            0,
        )
    _, identifier, count = max(qualified, key=lambda item: item[0])
    return identifier, "selected_using_prior_oos_trades_only", count


def _evaluate_adaptive_policy(
    candidate_results: dict[str, dict[str, Any]], baseline_id: str, folds: list[OOFFold]
) -> dict[str, Any]:
    chosen_folds: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for fold_id, fold in enumerate(folds):
        identifier, reason, prior_count = _choose_configuration(
            candidate_results, fold_id, baseline_id
        )
        report = candidate_results[identifier]["folds"][fold_id]
        trace.append(
            {
                "fold_id": fold_id,
                "test_start_index": fold.test_start,
                "selected_configuration": identifier,
                "selection_reason": reason,
                "prior_selected_trades_used": prior_count,
                "decision_uses_current_fold_results": False,
            }
        )
        if report["status"] == "EVALUATED":
            chosen_folds.append(report)
    trades = _flatten_trades(chosen_folds)
    return {
        "selection_trace": trace,
        "evaluated_fold_count": len(chosen_folds),
        "fallback_fold_count": sum(
            "fallback_" in str(item["selection_reason"]) for item in trace
        ),
        **_trade_metrics(trades),
    }


def _economic_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    count = int(metrics["trade_count"])
    if count < MIN_OOS_TRADES_FOR_ECONOMIC_EVALUATION:
        status = "INSUFFICIENT_SAMPLE"
    elif (
        float(metrics["win_rate"] or 0.0) > 0.60
        and float(metrics["net_expectancy"] or 0.0) > 0.0
        and float(metrics["max_abs_drawdown"] or 1.0) <= 0.15
    ):
        status = "PASS_RESEARCH_THRESHOLDS"
    else:
        status = "NO-GO"
    return {
        "status": status,
        "minimum_oos_trades": MIN_OOS_TRADES_FOR_ECONOMIC_EVALUATION,
        "required_win_rate_strictly_above": 0.60,
        "required_net_expectancy_strictly_above": 0.0,
        "maximum_abs_drawdown": 0.15,
        "independent_of_production_authorization": True,
    }


def _sort_candidate_summaries(
    candidate_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for identifier, result in candidate_results.items():
        summaries.append(
            {
                "candidate_id": identifier,
                "configuration": result["configuration"],
                **_aggregate_candidate(result),
            }
        )
    summaries.sort(
        key=lambda item: (
            float(item["net_expectancy"] or -1.0),
            float(item["win_rate"] or 0.0),
            -float(item["brier_score_weighted"] or 1.0),
            int(item["trade_count"]),
        ),
        reverse=True,
    )
    return summaries


def run_experiments(
    csv_path: Path, manifest_path: Path, output_path: Path, commit_sha: str
) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("experiment_commit_sha_invalid")
    candles, manifest, dataset_hash, quality_fingerprint = _load_dataset(
        csv_path, manifest_path, commit_sha
    )
    feature_frame = _engineer_experiment_features(candles)
    folds = _make_folds(len(candles))
    candidate_results = _make_candidate_results(candles, feature_frame, folds)
    baseline_id = _candidate_id("baseline", "current_unscaled_c1", 1.0)
    baseline_result = candidate_results[baseline_id]
    baseline_metrics = _aggregate_candidate(baseline_result)
    adaptive = _evaluate_adaptive_policy(candidate_results, baseline_id, folds)
    economic_gate = _economic_gate(adaptive)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "experiment_status": "COMPLETED",
        "commit_sha": commit_sha,
        "source": {
            "provider": PUBLIC_PROVIDER,
            "csv_sha256": dataset_hash,
            "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
            "row_count": int(len(candles)),
            "start": candles["date"].iloc[0].isoformat(),
            "end": candles["date"].iloc[-1].isoformat(),
            "data_quality_fingerprint": quality_fingerprint,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": {
            "walkforward": "expanding, chronological primary OOF; purged mature Meta labels",
            "folds": [
                {
                    "fold_id": index,
                    "train_start": fold.train_start,
                    "train_stop": fold.train_stop,
                    "test_start": fold.test_start,
                    "test_stop_exclusive": fold.test_stop,
                }
                for index, fold in enumerate(folds)
            ],
            "selected_signals": "long-only primary predictions; one active position at a time",
            "trade_overlap_policy": "skip signals until prior selected label_end_index",
            "cross_fold_maturity_policy": "exclude labels maturing at or after next fold start",
            "probability_threshold": RESEARCH_THRESHOLD,
            "threshold_was_optimized": False,
            "fees_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "triple_barrier": {
                "profit_taking_multiplier": 1.5,
                "stop_loss_multiplier": 1.0,
                "max_holding_bars": MAX_HOLDING_BARS,
            },
            "candidate_grid": {
                "feature_groups": list(FEATURE_GROUPS),
                "primary_models": [
                    {
                        "name": name,
                        "regularization_c": c_value,
                        "standardized": standardized,
                    }
                    for name, c_value, standardized in PRIMARY_CONFIGS
                ],
                "meta_regularization_c": list(META_C_VALUES),
                "candidate_count": len(candidate_results),
            },
            "candidate_selection": (
                "select only from prior OOS folds with at least 30 selected trades, "
                "win rate > 0.60 and positive net expectancy; otherwise use baseline"
            ),
            "shared_data_quality_gate": "PASS",
            "feature_lookahead": False,
            "uses_credentials": False,
            "exchange_calls": 0,
            "real_order_attempts": 0,
        },
        "baseline_fixed_configuration": {
            "candidate_id": baseline_id,
            **baseline_metrics,
        },
        "adaptive_walkforward_policy": adaptive,
        "economic_evaluation": economic_gate,
        "candidate_diagnostics": _sort_candidate_summaries(candidate_results),
        "champion": None,
        "economic_acceptance": "NO-GO",
        "promotion_status": "BLOCKED_NO_STRATEGY_LEVEL_EVIDENCE",
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "production_model_updated": False,
        "strategy_registry_updated": False,
        "limitations": [
            "These are public-market research results, not a reconciled Freqtrade or account-level paper ledger.",
            "The tested candidate policy is a long-only primary/meta research rule; it is not equivalent to production strategy fills.",
            "The model grid is exploratory; candidate summaries are not independent confirmatory evidence.",
            "A research gate pass does not authorize a Champion, production model update, registry admission or order execution.",
            "Owner must review strategy-level evidence and CI before any separate promotion PR.",
        ],
    }
    report["adaptive_walkforward_policy"]["economic_gate"] = economic_gate
    if report["economic_acceptance"] != "NO-GO":
        raise ValueError("experiment_production_economic_acceptance_must_remain_no_go")
    if (
        report["champion"] is not None
        or report["order_permission"] is not False
        or report["trading_authorization"] != "NO-GO"
    ):
        raise ValueError("experiment_closed_release_contract_failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_experiments(args.input, args.manifest, args.output, args.commit_sha)
    print(
        json.dumps(
            {
                "experiment_status": report["experiment_status"],
                "candidate_count": report["protocol"]["candidate_grid"][
                    "candidate_count"
                ],
                "oos_trade_count": report["adaptive_walkforward_policy"]["trade_count"],
                "win_rate": report["adaptive_walkforward_policy"]["win_rate"],
                "net_expectancy": report["adaptive_walkforward_policy"][
                    "net_expectancy"
                ],
                "economic_gate": report["economic_evaluation"]["status"],
                "Champion": report["champion"],
                "order_permission": report["order_permission"],
                "trading_authorization": report["trading_authorization"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
