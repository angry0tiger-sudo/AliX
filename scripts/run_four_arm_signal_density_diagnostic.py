"""Diagnostic-only signal-density audit using the immutable Run #10 forecast cache."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from scripts import run_four_arm_experiment as four_arm

RUN10_RUN_ID = 43712086361
RUN10_ARTIFACT_ID = 14163527266
RUN10_ARTIFACT_SHA256 = (
    "b12c3a807316ab52829e6b0b1aa450057133d76afc0aeb54f0ac259f38300679"
)
EXPECTED_CACHE_SHA256 = (
    "aea6c537f3ca79a28e75faf1a118bd57d5dea4fb58b45e477eb401df03d01868"
)
EXPECTED_CACHE_IDENTITY_SHA256 = (
    "f879e5c179cf6451485ef1ce432c7d921f5f4344146d80f08026f7f8cfcf2427"
)
ARMS = ("B", "C", "D")
THRESHOLDS_BPS = (120, 180, 240)
HISTOGRAM_LABELS = (
    "<= -100%",
    "(-100%, -50%]",
    "(-50%, -25%]",
    "(-25%, 0%]",
    "(0%, 25%]",
    "(25%, 50%]",
    "(50%, 100%]",
    "> 100%",
)


class DiagnosticError(RuntimeError):
    """Raised when a diagnostic input or reconciliation contract fails."""


@dataclass(frozen=True, slots=True)
class ForecastSeries:
    returns: npt.NDArray[np.float64]
    valid: npt.NDArray[np.bool_]
    signals: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ExecutionArrays:
    open_values: npt.NDArray[np.float64]
    high_values: npt.NDArray[np.float64]
    low_values: npt.NDArray[np.float64]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise DiagnosticError("diagnostic_json_input_unreadable") from None
    if not isinstance(payload, dict):
        raise DiagnosticError("diagnostic_json_root_invalid")
    return payload


def _load_confirmatory_prefix(
    csv_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, str]:
    if csv_path.name != four_arm.PREHACK_FILENAME:
        raise DiagnosticError("diagnostic_dataset_path_not_registered")
    actual_sha = four_arm._sha256_file(csv_path)
    if actual_sha != four_arm.PREHACK_DATASET_SHA256:
        raise DiagnosticError("diagnostic_dataset_sha256_mismatch")
    four_arm._validate_data_manifest(manifest_path)
    frame = pd.read_csv(csv_path, nrows=four_arm.CONFIRMATORY_ROW_COUNT)
    if len(frame) != four_arm.CONFIRMATORY_ROW_COUNT:
        raise DiagnosticError("diagnostic_confirmatory_prefix_row_count_invalid")
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise DiagnosticError("diagnostic_prefix_schema_invalid")
    frame = frame.rename(columns={"date": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    four_arm._validate_hourly_grid(frame)
    four_arm._validate_ohlcv(frame)
    return frame.reset_index(drop=True), actual_sha


def _load_cache_entries(cache_path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = _load_json_object(cache_path)
    entries = payload.get("entries")
    valid_shape = (
        payload.get("schema_version") == "alix.four-arm.forecast-cache.v1"
        and payload.get("confirmatory_rows") == four_arm.CONFIRMATORY_ROW_COUNT
        and payload.get("anchor_count") == 1_138
        and isinstance(entries, list)
        and len(entries) == 1_138
    )
    if not valid_shape or not isinstance(entries, list):
        raise DiagnosticError("diagnostic_run10_cache_contract_invalid")
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    cache_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if (
        cache_sha != EXPECTED_CACHE_SHA256
        or payload.get("cache_sha256") != EXPECTED_CACHE_SHA256
        or payload.get("identity_sha256") != EXPECTED_CACHE_IDENTITY_SHA256
    ):
        raise DiagnosticError("diagnostic_run10_cache_digest_mismatch")
    if payload.get("model_inference_calls_total") != 2_276:
        raise DiagnosticError("diagnostic_cache_inference_inventory_invalid")
    return entries, canonical


def _build_forecast_cache(
    entries: list[dict[str, Any]], canonical: str
) -> four_arm.ForecastCache:
    expected_anchors = tuple(
        range(
            four_arm.MODEL_CONTEXT_BARS - 1,
            four_arm.CONFIRMATORY_ROW_COUNT - four_arm.MAX_HOLDING_BARS,
            four_arm.FORECAST_REFRESH_BARS,
        )
    )
    anchors: list[int] = []
    forecasts: list[four_arm.Forecast] = []
    for item in entries:
        if not isinstance(item, dict):
            raise DiagnosticError("diagnostic_cache_entry_invalid")
        anchor = item.get("anchor_index")
        kronos = item.get("kronos_return")
        chronos = item.get("chronos_return")
        numeric = isinstance(kronos, (int, float)) and isinstance(chronos, (int, float))
        if type(anchor) is not int or not numeric:
            raise DiagnosticError("diagnostic_cache_entry_fields_invalid")
        if not math.isfinite(float(kronos)) or not math.isfinite(float(chronos)):
            raise DiagnosticError("diagnostic_cache_non_finite_forecast")
        anchors.append(anchor)
        forecasts.append(
            four_arm.Forecast(
                kronos_return=float(kronos),
                chronos_return=float(chronos),
            )
        )
    if tuple(anchors) != expected_anchors:
        raise DiagnosticError("diagnostic_cache_anchor_inventory_invalid")
    return four_arm.ForecastCache(
        anchors=tuple(anchors),
        forecasts=tuple(forecasts),
        identity_sha256=EXPECTED_CACHE_IDENTITY_SHA256,
        cache_sha256=EXPECTED_CACHE_SHA256,
        entries_json=canonical,
    )


def _load_trial_manifests(artifact_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifests: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(artifact_dir.glob("trial-*.json")):
        manifest = _load_json_object(path)
        arm_id = manifest.get("arm_id")
        threshold = manifest.get("threshold_bps")
        if arm_id not in ARMS or type(threshold) is not int:
            continue
        key = (str(arm_id), threshold)
        if key in manifests:
            raise DiagnosticError("diagnostic_duplicate_trial_manifest")
        expected_id = four_arm.TRIAL_ID_MATRIX[threshold][arm_id]
        expected = (
            manifest.get("trial_id") == expected_id
            and manifest.get("dataset_sha256") == four_arm.PREHACK_DATASET_SHA256
            and manifest.get("dataset_rows") == four_arm.CONFIRMATORY_ROW_COUNT
            and manifest.get("forecast_cache_sha256") == EXPECTED_CACHE_SHA256
            and manifest.get("forecast_cache_identity_sha256")
            == EXPECTED_CACHE_IDENTITY_SHA256
            and manifest.get("forecast_cache_computed_for_this_trial") is False
        )
        if not expected:
            raise DiagnosticError("diagnostic_trial_manifest_contract_invalid")
        manifests[key] = manifest
    required = {(arm, threshold) for arm in ARMS for threshold in THRESHOLDS_BPS}
    if set(manifests) != required:
        raise DiagnosticError("diagnostic_trial_manifest_inventory_incomplete")
    return manifests


def _forecast_series(
    cache: four_arm.ForecastCache,
    row_count: int,
    arm_id: str,
    threshold_bps: int,
) -> ForecastSeries:
    returns = np.full(row_count, np.nan, dtype=np.float64)
    valid = np.zeros(row_count, dtype=np.bool_)
    for anchor, forecast in zip(cache.anchors, cache.forecasts, strict=True):
        stop = min(anchor + four_arm.FORECAST_REFRESH_BARS, row_count)
        if bool(valid[anchor:stop].any()):
            raise DiagnosticError("diagnostic_forecast_cache_overlap")
        if arm_id == "B":
            value = forecast.kronos_return
        elif arm_id == "C":
            value = forecast.chronos_return
        else:
            value = (forecast.kronos_return + forecast.chronos_return) / 2.0
        returns[anchor:stop] = value
        valid[anchor:stop] = True
    signals = valid & (returns * 10_000.0 > threshold_bps)
    runner_signals = four_arm._signals_from_forecast_cache(
        cache, row_count, arm_id, threshold_bps
    )
    if not np.array_equal(signals, runner_signals):
        raise DiagnosticError("diagnostic_signal_definition_mismatch")
    return ForecastSeries(returns=returns, valid=valid, signals=signals)


def _execution_arrays(frame: pd.DataFrame) -> tuple[ExecutionArrays, npt.NDArray[np.float64]]:
    data = ExecutionArrays(
        open_values=frame["open"].to_numpy(dtype=np.float64),
        high_values=frame["high"].to_numpy(dtype=np.float64),
        low_values=frame["low"].to_numpy(dtype=np.float64),
    )
    atr = four_arm._calculate_atr(frame)
    return data, atr


def _find_exit(
    data: ExecutionArrays,
    fold: four_arm.Fold,
    atr: npt.NDArray[np.float64],
    signal_index: int,
) -> tuple[int, str]:
    entry_index = signal_index + 1
    entry_price = float(data.open_values[entry_index])
    risk_distance = four_arm.ATR_MULTIPLIER * float(atr[signal_index])
    stop_price = entry_price - risk_distance
    target_price = entry_price + risk_distance
    high_water = entry_price
    last_bar = min(entry_index + four_arm.MAX_HOLDING_BARS - 1, fold.test_end - 1)
    for bar_index in range(entry_index, last_bar + 1):
        bar_open = float(data.open_values[bar_index])
        bar_high = float(data.high_values[bar_index])
        bar_low = float(data.low_values[bar_index])
        if bar_open <= stop_price:
            return bar_index, "gap_stop"
        if bar_open >= target_price:
            return bar_index, "gap_target"
        stop_touched = bar_low <= stop_price
        target_touched = bar_high >= target_price
        if stop_touched:
            reason = "dual_touch_stop_first" if target_touched else "stop_loss"
            return bar_index, reason
        if target_touched:
            return bar_index, "take_profit"
        if bar_index == last_bar:
            return bar_index, "maximum_holding_time"
        high_water = max(high_water, bar_high)
        current_atr = float(atr[bar_index])
        if math.isfinite(current_atr) and current_atr > 0.0:
            stop_price = max(
                stop_price,
                high_water - four_arm.ATR_MULTIPLIER * current_atr,
            )
    raise DiagnosticError("diagnostic_trade_exit_not_reached")


def _histogram_counts(margins: npt.NDArray[np.float64]) -> dict[str, int]:
    masks = (
        margins <= -1.0,
        (margins > -1.0) & (margins <= -0.5),
        (margins > -0.5) & (margins <= -0.25),
        (margins > -0.25) & (margins <= 0.0),
        (margins > 0.0) & (margins <= 0.25),
        (margins > 0.25) & (margins <= 0.5),
        (margins > 0.5) & (margins <= 1.0),
        margins > 1.0,
    )
    return {
        label: int(np.count_nonzero(mask))
        for label, mask in zip(HISTOGRAM_LABELS, masks, strict=True)
    }


def _count_trade_events(
    data: ExecutionArrays,
    fold: four_arm.Fold,
    atr: npt.NDArray[np.float64],
    signals: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], Counter[str]]:
    active = np.zeros(len(atr), dtype=np.bool_)
    exit_reasons: Counter[str] = Counter()
    cursor = fold.test_start
    eligible = signals & np.isfinite(atr)
    while cursor < fold.test_end - 1:
        if not bool(eligible[cursor]):
            cursor += 1
            continue
        exit_index, reason = _find_exit(data, fold, atr, cursor)
        entry_index = cursor + 1
        active[entry_index : exit_index + 1] = True
        exit_reasons[reason] += 1
        cursor = exit_index + 1
    return active, exit_reasons


def _entry_rejections(
    signals: npt.NDArray[np.bool_],
    atr: npt.NDArray[np.float64],
    active: npt.NDArray[np.bool_],
    fold: four_arm.Fold,
) -> dict[str, int]:
    window = slice(fold.test_start, fold.test_end)
    signal = signals[window]
    finite_atr = np.isfinite(atr[window])
    is_open = active[window]
    invalid_atr = int(np.count_nonzero(signal & ~is_open & ~finite_atr))
    suppressed = int(np.count_nonzero(signal & is_open))
    no_next = int(
        bool(signal[-1]) and bool(finite_atr[-1]) and not bool(is_open[-1])
    )
    return {
        "non_finite_atr": invalid_atr,
        "no_valid_next_bar_inside_fold": no_next,
        "suppressed_while_position_open": suppressed,
    }


def _forecast_counts(
    series: ForecastSeries,
    fold: four_arm.Fold,
    threshold_bps: int,
) -> dict[str, Any]:
    window = slice(fold.test_start, fold.test_end)
    covered = series.valid[window]
    signal = series.signals[window]
    valid_bars = int(np.count_nonzero(covered))
    eligible = int(np.count_nonzero(signal))
    below = valid_bars - eligible
    episodes = int(bool(signal[0])) + int(
        np.count_nonzero(signal[1:] & ~signal[:-1])
    )
    margins = (
        series.returns[window][covered] * 10_000.0 - threshold_bps
    ) / threshold_bps
    histogram = _histogram_counts(margins)
    if below + eligible != valid_bars:
        raise DiagnosticError("diagnostic_signal_count_reconciliation_failed")
    if sum(histogram.values()) != valid_bars:
        raise DiagnosticError("diagnostic_histogram_reconciliation_failed")
    return {
        "valid_forecast_bars": valid_bars,
        "below_threshold_bars": below,
        "eligible_signal_bars": eligible,
        "eligible_signal_episodes": episodes,
        "forecast_margin_histogram": histogram,
    }


def _forecast_anchor_count(
    cache: four_arm.ForecastCache, fold: four_arm.Fold
) -> int:
    return sum(
        1
        for anchor in cache.anchors
        if anchor < fold.test_end
        and anchor + four_arm.FORECAST_REFRESH_BARS > fold.test_start
    )


def _evaluate_fold(
    data: ExecutionArrays,
    fold: four_arm.Fold,
    atr: npt.NDArray[np.float64],
    series: ForecastSeries,
    cache: four_arm.ForecastCache,
    threshold_bps: int,
) -> dict[str, Any]:
    active, exit_reasons = _count_trade_events(data, fold, atr, series.signals)
    entry_rejections = _entry_rejections(
        series.signals, atr, active, fold
    )
    forecast_counts = _forecast_counts(series, fold, threshold_bps)
    completed = int(sum(exit_reasons.values()))
    accounted = sum(entry_rejections.values()) + completed
    if forecast_counts["eligible_signal_bars"] != accounted:
        raise DiagnosticError("diagnostic_entry_flow_reconciliation_failed")
    return {
        "fold_id": fold.fold_id,
        "test_start_index": fold.test_start,
        "test_end_index_exclusive": fold.test_end,
        "valid_forecast_anchors": _forecast_anchor_count(cache, fold),
        **forecast_counts,
        "entry_gate_rejections": entry_rejections,
        "entry_attempts": completed,
        "completed_trades": completed,
        "trade_exit_reason_counts": dict(sorted(exit_reasons.items())),
        "risk_gate_rejections": {
            "status": "MISSING_NO_SEPARATE_PRE_ENTRY_RISK_GATE",
            "count": None,
            "technical_reason": (
                "The runner applies ATR barriers after entry; there is no "
                "independent pre-entry risk rejection predicate to count."
            ),
        },
    }


def _manifest_trade_count(manifest: dict[str, Any], fold_id: int) -> int:
    fold_items = manifest.get("folds")
    if not isinstance(fold_items, list):
        raise DiagnosticError("diagnostic_manifest_folds_missing")
    matches = [
        item for item in fold_items
        if isinstance(item, dict) and item.get("fold_id") == fold_id
    ]
    if len(matches) != 1:
        raise DiagnosticError("diagnostic_manifest_fold_inventory_invalid")
    metrics = matches[0].get("metrics")
    if not isinstance(metrics, dict):
        raise DiagnosticError("diagnostic_manifest_fold_metrics_missing")
    cost_50 = metrics.get("50")
    cost_90 = metrics.get("90")
    if not isinstance(cost_50, dict) or not isinstance(cost_90, dict):
        raise DiagnosticError("diagnostic_manifest_cost_metrics_missing")
    trades_50 = cost_50.get("trades")
    trades_90 = cost_90.get("trades")
    if type(trades_50) is not int or trades_50 != trades_90:
        raise DiagnosticError("diagnostic_manifest_trade_count_invalid")
    return trades_50


def _reconcile_manifest(
    manifest: dict[str, Any], fold_results: list[dict[str, Any]]
) -> None:
    for result in fold_results:
        expected = _manifest_trade_count(manifest, int(result["fold_id"]))
        if expected != result["completed_trades"]:
            raise DiagnosticError("diagnostic_fold_trade_count_mismatch")
    totals = sum(int(item["completed_trades"]) for item in fold_results)
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("trades") != totals:
        raise DiagnosticError("diagnostic_aggregate_trade_count_mismatch")
    exit_counts: Counter[str] = Counter()
    for result in fold_results:
        exit_counts.update(result["trade_exit_reason_counts"])
    expected_exits = manifest.get("trade_exit_reasons")
    if not isinstance(expected_exits, dict):
        raise DiagnosticError("diagnostic_manifest_exit_reasons_missing")
    if dict(sorted(exit_counts.items())) != dict(sorted(expected_exits.items())):
        raise DiagnosticError("diagnostic_trade_exit_reason_mismatch")


def _aggregate_cell(
    fold_results: list[dict[str, Any]], threshold_bps: int
) -> dict[str, Any]:
    scalar_keys = (
        "valid_forecast_anchors", "valid_forecast_bars", "below_threshold_bars",
        "eligible_signal_bars", "eligible_signal_episodes", "entry_attempts",
        "completed_trades",
    )
    totals = {
        key: sum(int(item[key]) for item in fold_results) for key in scalar_keys
    }
    rejection_keys = (
        "non_finite_atr", "no_valid_next_bar_inside_fold",
        "suppressed_while_position_open",
    )
    rejections = {
        key: sum(int(item["entry_gate_rejections"][key]) for item in fold_results)
        for key in rejection_keys
    }
    exits: Counter[str] = Counter()
    histogram: Counter[str] = Counter()
    for item in fold_results:
        exits.update(item["trade_exit_reason_counts"])
        histogram.update(item["forecast_margin_histogram"])
    valid = totals["valid_forecast_bars"]
    eligible = totals["eligible_signal_bars"]
    if valid != totals["below_threshold_bars"] + eligible:
        raise DiagnosticError("diagnostic_aggregate_signal_reconciliation_failed")
    if eligible != sum(rejections.values()) + totals["completed_trades"]:
        raise DiagnosticError("diagnostic_aggregate_entry_reconciliation_failed")
    if sum(histogram.values()) != valid:
        raise DiagnosticError("diagnostic_aggregate_histogram_reconciliation_failed")
    return {
        **totals,
        "entry_gate_rejections": rejections,
        "risk_gate_rejections": {
            "status": "MISSING_NO_SEPARATE_PRE_ENTRY_RISK_GATE",
            "count": None,
        },
        "trade_exit_reason_counts": dict(sorted(exits.items())),
        "forecast_margin_histogram": {
            label: int(histogram[label]) for label in HISTOGRAM_LABELS
        },
        "threshold_qualified_fraction": eligible / valid if valid else None,
        "eligible_signal_to_trade_fraction": (
            totals["completed_trades"] / eligible if eligible else None
        ),
        "threshold_bps": threshold_bps,
    }


def _diagnose_cell(
    manifests: dict[tuple[str, int], dict[str, Any]],
    artifact_dir: Path,
    data: ExecutionArrays,
    frame: pd.DataFrame,
    atr: npt.NDArray[np.float64],
    folds: tuple[four_arm.Fold, ...],
    cache: four_arm.ForecastCache,
    arm_id: str,
    threshold_bps: int,
) -> dict[str, Any]:
    manifest = manifests[(arm_id, threshold_bps)]
    series = _forecast_series(cache, len(frame), arm_id, threshold_bps)
    fold_results = [
        _evaluate_fold(data, fold, atr, series, cache, threshold_bps)
        for fold in folds
    ]
    _reconcile_manifest(manifest, fold_results)
    return {
        "arm_id": arm_id,
        "threshold_bps": threshold_bps,
        "reference_trial_id": manifest["trial_id"],
        "folds": fold_results,
        "aggregate": _aggregate_cell(fold_results, threshold_bps),
        "reconciles_to_run10_manifests": True,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Four-Arm Signal-Density Diagnostic",
        "",
        "- Mode: DIAGNOSTIC_ONLY_NOT_A_TRIAL",
        "- Economic metrics computed: false",
        "- New trial records: 0",
        "- Model inference calls in this run: 0",
        "- Holdout access: false",
        "- Confirmatory rows parsed: 27,825",
        "",
        "## Per-arm x threshold x fold counts",
        "",
        "| Arm | Threshold | Fold | Anchors | Covered | Eligible | Below | Episodes | ATR rejects | No next bar | Position suppressed | Completed | Risk rejects |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in report["cells"]:
        for fold in cell["folds"]:
            rejects = fold["entry_gate_rejections"]
            values = (
                cell["arm_id"], cell["threshold_bps"], fold["fold_id"],
                fold["valid_forecast_anchors"], fold["valid_forecast_bars"],
                fold["eligible_signal_bars"], fold["below_threshold_bars"],
                fold["eligible_signal_episodes"], rejects["non_finite_atr"],
                rejects["no_valid_next_bar_inside_fold"],
                rejects["suppressed_while_position_open"],
                fold["completed_trades"],
            )
            lines.append("| " + " | ".join(str(v) for v in values) + " | MISSING |")
    lines.extend(
        [
            "",
            "## Aggregate forecast-return margin histograms",
            "",
            "Margin = (forecast_return_bps - threshold_bps) / threshold_bps.",
            "",
            "| Arm | Threshold | <= -100% | (-100,-50] | (-50,-25] | (-25,0] | (0,25] | (25,50] | (50,100] | >100% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in report["cells"]:
        histogram = cell["aggregate"]["forecast_margin_histogram"]
        values = [histogram[label] for label in HISTOGRAM_LABELS]
        lines.append(
            "| "
            + " | ".join([cell["arm_id"], str(cell["threshold_bps"])])
            + " | "
            + " | ".join(str(v) for v in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Risk gate",
            "",
            "Risk-gate rejection counts are MISSING, not zero. The current runner "
            "has no separate pre-entry risk gate; ATR barriers execute after entry "
            "and are reported as trade exit outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_safety_environment() -> None:
    expected = {
        "ALIX_FOUR_ARM_EXECUTION_AUTHORIZED": "false",
        "ALIX_DRY_RUN": "true",
        "ALIX_ALLOW_LIVE_TRADING": "false",
    }
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise DiagnosticError("diagnostic_safety_environment_invalid")


def _build_report(
    frame: pd.DataFrame,
    dataset_sha: str,
    cache: four_arm.ForecastCache,
    manifests: dict[tuple[str, int], dict[str, Any]],
    artifact_dir: Path,
) -> dict[str, Any]:
    data, atr = _execution_arrays(frame)
    folds = four_arm.build_walk_forward_folds(len(frame))
    cells = [
        _diagnose_cell(
            manifests, artifact_dir, data, frame, atr, folds, cache, arm, threshold
        )
        for arm in ARMS
        for threshold in THRESHOLDS_BPS
    ]
    return {
        "schema_version": "alix.four-arm.signal-density-diagnostic.v1",
        "mode": "DIAGNOSTIC_ONLY_NOT_A_TRIAL",
        "dataset": {
            "csv_sha256": dataset_sha,
            "source_rows": four_arm.PREHACK_ROW_COUNT,
            "confirmatory_rows": len(frame),
            "excluded_tail_rows": four_arm.EXCLUDED_TAIL_ROWS,
            "parsed_rows": len(frame),
            "hourly_grid_verified": True,
        },
        "forecast_cache": {
            "source_run_id": RUN10_RUN_ID,
            "source_artifact_id": RUN10_ARTIFACT_ID,
            "source_artifact_sha256": RUN10_ARTIFACT_SHA256,
            "anchor_count": len(cache.anchors),
            "cache_sha256": cache.cache_sha256,
            "identity_sha256": cache.identity_sha256,
            "reused_without_model_inference": True,
        },
        "folds": [
            {
                "fold_id": fold.fold_id,
                "test_start_index": fold.test_start,
                "test_end_index_exclusive": fold.test_end,
            }
            for fold in folds
        ],
        "safety": {
            "economic_metrics_computed": False,
            "trial_records_created": 0,
            "model_inference_calls": 0,
            "model_weights_downloaded": False,
            "holdout_access": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "GLOBAL_N": "UNKNOWN",
            "DSR_computed": False,
        },
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run10-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _verify_safety_environment()
        frame, dataset_sha = _load_confirmatory_prefix(args.csv, args.manifest)
        entries, canonical = _load_cache_entries(
            args.run10_artifact_dir / "forecast-cache.json"
        )
        cache = _build_forecast_cache(entries, canonical)
        manifests = _load_trial_manifests(args.run10_artifact_dir)
        report = _build_report(
            frame, dataset_sha, cache, manifests, args.run10_artifact_dir
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "diagnostic-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "diagnostic-report.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
        print("SIGNAL_DENSITY_DIAGNOSTIC=PASS")
        print("CELLS=9 FOLDS=27 CACHE_ANCHORS=1138 MODEL_INFERENCE_CALLS=0")
        print("ECONOMIC_METRICS=false TRIAL_RECORDS=0 HOLDOUT_ACCESS=false")
        return 0
    except DiagnosticError as exc:
        print(f"SIGNAL_DENSITY_DIAGNOSTIC=FAIL:{exc}")
        return 2
    except Exception as exc:
        print(f"SIGNAL_DENSITY_DIAGNOSTIC=ERROR:{type(exc).__name__}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
