from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.adaptive_targets import (
    EXIT_REASONS,
    UNLABELED_REASONS,
    apply_adaptive_triple_barrier,
)
from services.ml_core.label_transition_audit import (
    PAIR_NAMES,
    SCENARIOS,
    SYMBOLS,
    TIMEFRAMES,
    _label_audit_counts,
    build_transition_audit,
    collect_transition_events,
    resample_minute_candles,
    validate_minute_candles,
    validate_v4c_report,
)


def minute_frame(days: int = 1) -> DataFrame:
    rows = days * 24 * 60
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="min")
    opened = 100.0 + np.arange(rows, dtype=float) * 0.0001
    close = opened + 0.0001
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": close + 0.01,
            "low": opened - 0.01,
            "close": close,
            "volume": np.ones(rows, dtype=float),
        }
    )


def _distribution(labels: DataFrame, indices: list[int]) -> dict[str, Any]:
    observed = labels.iloc[indices]["label"]
    counts = {
        label: int((observed == label).sum()) for label in ("BUY", "SELL", "HOLD")
    }
    return {
        "total": len(indices),
        "classes": {
            label: {
                "count": count,
                "percent": round(100.0 * count / len(indices), 4) if indices else 0.0,
            }
            for label, count in counts.items()
        },
    }


def _audit_metadata(labels: DataFrame) -> dict[str, Any]:
    counts = _label_audit_counts(labels)
    return {
        "scope": "ALL_TARGET_BARS_BEFORE_SPLIT",
        "labeled_events": counts["labeled_events"],
        "unlabeled_events": counts["unlabeled_events"],
        "exit_reason_distribution": {
            "total": counts["labeled_events"],
            "reasons": {
                reason: {
                    "count": counts["exit_reasons"][reason],
                    "percent": 0.0,
                }
                for reason in EXIT_REASONS
            },
        },
        "unlabeled_reason_distribution": {
            "total": counts["unlabeled_events"],
            "reasons": {
                reason: {
                    "count": counts["unlabeled_reasons"][reason],
                    "percent": 0.0,
                }
                for reason in UNLABELED_REASONS
            },
        },
    }


def _mini_report(frames: dict[str, DataFrame]) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe, minutes in TIMEFRAMES.items():
            bars = resample_minute_candles(frames[symbol], timeframe)
            features_start = int(len(bars) * 0.70)
            features_end = int(len(bars) * 0.85) - 1
            features = __import__(
                "services.ml_core.causal_features",
                fromlist=["build_causal_feature_matrix"],
            ).build_causal_feature_matrix(bars)
            for scenario, (horizon, take_profit, stop_loss) in SCENARIOS.items():
                labels = apply_adaptive_triple_barrier(
                    bars,
                    horizon_candles=horizon,
                    take_profit_multiplier=take_profit,
                    stop_loss_multiplier=stop_loss,
                )
                eligible = (
                    features.notna().all(axis=1)
                    & labels["label"].isin(("BUY", "SELL", "HOLD"))
                    & labels["label_end_idx"].notna()
                )
                indices = [
                    index
                    for index in range(features_start, features_end + 1)
                    if bool(eligible.iloc[index])
                ]
                experiments.append(
                    {
                        "pair": PAIR_NAMES[symbol],
                        "timeframe": timeframe,
                        "scenario": scenario,
                        "triple_barrier": {
                            "horizon_candles": horizon,
                            "take_profit_multiplier": take_profit,
                            "stop_loss_multiplier": stop_loss,
                        },
                        "bar_count": len(bars),
                        "train_rows": 0,
                        "validation_rows": len(indices),
                        "label_distribution": {
                            "train": _distribution(labels, []),
                            "validation": _distribution(labels, indices),
                        },
                        "label_audit": _audit_metadata(labels),
                        "status": "SKIPPED_INSUFFICIENT_SAMPLES",
                        "model_variants": None,
                        "split": {
                            "validation_start_index": features_start,
                            "validation_end_index": features_end,
                        },
                    }
                )
    return {
        "schema_version": "alix.ml-core.real-data-classification-diagnostic.v5",
        "commit_sha": "ac428dd94383173570fe3cf8af510e5b0941336e",
        "requested_days": 1,
        "data_window_selection": {"selected_days": 1},
        "experiment_count": 16,
        "experiments": experiments,
        "triple_barrier_scenarios": [
            {
                "name": name,
                "horizon_candles": values[0],
                "take_profit_multiplier": values[1],
                "stop_loss_multiplier": values[2],
            }
            for name, values in SCENARIOS.items()
        ],
        "trained_experiments": 0,
        "trained_models": 0,
        "safety": {
            "Champion": None,
            "credentials_used": False,
            "order_permission": False,
            "orders_or_execution_api_used": False,
            "private_api_calls": 0,
            "public_api_only": True,
            "real_order_attempts": 0,
            "trading_authorization": "NO-GO",
        },
    }


def _manual_labels(
    bars: DataFrame,
    offsets: list[int],
    reasons: list[str],
    directions: list[str],
) -> tuple[DataFrame, DataFrame]:
    baseline = DataFrame(
        {
            "label": ["HOLD"] * len(bars),
            "exit_reason": ["HORIZON_EXPIRY"] * len(bars),
            "label_end_idx": [index + 24 for index in range(len(bars))],
        }
    )
    short = DataFrame(
        {
            "label": ["HOLD"] * len(bars),
            "exit_reason": ["HORIZON_EXPIRY"] * len(bars),
            "label_end_idx": [index + 12 for index in range(len(bars))],
        }
    )
    for index, (offset, reason, direction) in enumerate(
        zip(offsets, reasons, directions, strict=True)
    ):
        baseline.loc[index, "label"] = direction
        baseline.loc[index, "exit_reason"] = reason
        baseline.loc[index, "label_end_idx"] = index + offset
    return baseline, short


def test_validate_minute_candles_accepts_exact_complete_source() -> None:
    data = minute_frame()

    validated = validate_minute_candles(data, expected_rows=len(data))

    assert len(validated) == 1440
    assert str(validated["date"].dt.tz) == "UTC"


def test_validate_minute_candles_rejects_missing_columns_and_gaps() -> None:
    data = minute_frame()
    with pytest.raises(ValueError, match="ohlcv_missing_columns"):
        validate_minute_candles(data.drop(columns=["volume"]), len(data))
    with pytest.raises(ValueError, match="minute_data_gap_or_interval_mismatch"):
        validate_minute_candles(
            data.drop(index=100).reset_index(drop=True), len(data) - 1
        )


def test_validate_minute_candles_rejects_nonfinite_geometry_and_count() -> None:
    data = minute_frame()
    invalid = data.copy()
    invalid.loc[10, "high"] = np.nan
    with pytest.raises(ValueError, match="ohlcv_value_not_finite"):
        validate_minute_candles(invalid, len(invalid))
    invalid = data.copy()
    invalid.loc[10, "low"] = invalid["high"].to_numpy(dtype=float)[10] + 1.0
    with pytest.raises(ValueError, match="ohlcv_geometry_invalid"):
        validate_minute_candles(invalid, len(invalid))
    with pytest.raises(ValueError, match="minute_candle_count_mismatch"):
        validate_minute_candles(data, len(data) + 1)


def test_resample_minute_candles_keeps_only_complete_epoch_aligned_bars() -> None:
    bars = resample_minute_candles(minute_frame(), "15m")

    assert len(bars) == 96
    assert bars["date"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert bars["volume"].iloc[0] == 15.0
    with pytest.raises(ValueError, match="entry_timeframe_out_of_policy"):
        resample_minute_candles(minute_frame(), "1d")


def test_collect_transition_events_reports_exact_late_touch_buckets() -> None:
    bars = minute_frame().iloc[:40].copy()
    baseline, short = _manual_labels(
        bars,
        [13, 18, 19, 24],
        ["TP_HIT", "SL_HIT", "COLLISION", "TP_HIT"],
        ["BUY", "SELL", "SELL", "BUY"],
    )

    events = collect_transition_events(bars, baseline, short, [0, 1, 2, 3])

    assert [event["late_hit_candle_offset"] for event in events] == [13, 18, 19, 24]
    assert [event["late_hit_bucket"] for event in events] == [
        "13-18",
        "13-18",
        "19-24",
        "19-24",
    ]
    assert events[2]["first_touch_reason"] == "COLLISION"
    assert events[3]["first_touch_timestamp"] == bars["date"].iloc[27].isoformat()


def test_collect_transition_events_rejects_inconsistent_hit_semantics() -> None:
    bars = minute_frame().iloc[:40].copy()
    baseline, short = _manual_labels(bars, [12], ["TP_HIT"], ["BUY"])
    with pytest.raises(ValueError, match="transition_hit_offset_out_of_late_window"):
        collect_transition_events(bars, baseline, short, [0])
    baseline, short = _manual_labels(bars, [13], ["SL_HIT"], ["BUY"])
    with pytest.raises(ValueError, match="baseline_buy_exit_reason_mismatch"):
        collect_transition_events(bars, baseline, short, [0])
    baseline, short = _manual_labels(bars, [13], ["TP_HIT"], ["BUY"])
    short.loc[0, "exit_reason"] = "TP_HIT"
    with pytest.raises(ValueError, match="short_horizon_hold_reason_mismatch"):
        collect_transition_events(bars, baseline, short, [0])


def test_build_transition_audit_checks_all_experiments_and_safety() -> None:
    frames = {symbol: minute_frame() for symbol in SYMBOLS}
    report = _mini_report(frames)

    result = build_transition_audit(report, frames)

    assert result["schema_version"] == "alix.ml-core.label-transition-audit.v1"
    assert len(result["experiments"]) == 8
    assert result["aggregate_transition_summary"]["total"] == 0
    assert result["model_training_performed"] is False
    assert result["safety"]["Champion"] is None
    assert result["safety"]["trading_authorization"] == "NO-GO"
    assert all(
        item["transition_summary"]["total"] == 0 for item in result["experiments"]
    )


def test_build_transition_audit_detects_source_mismatch_and_missing_symbol() -> None:
    frames = {symbol: minute_frame() for symbol in SYMBOLS}
    report = _mini_report(frames)
    short_source = {"BTCUSDT": frames["BTCUSDT"]}
    with pytest.raises(ValueError, match="minute_data_missing:ETHUSDT"):
        build_transition_audit(report, short_source)
    broken = {symbol: frame.copy() for symbol, frame in frames.items()}
    broken["BTCUSDT"] = broken["BTCUSDT"].drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="minute_candle_count_mismatch"):
        build_transition_audit(report, broken)


def test_validate_v4c_report_rejects_provenance_scenario_and_safety_changes() -> None:
    frames = {symbol: minute_frame() for symbol in SYMBOLS}
    report = _mini_report(frames)
    assert len(validate_v4c_report(report)) == 16

    changed = copy.deepcopy(report)
    changed["commit_sha"] = "different"
    with pytest.raises(ValueError, match="v4c_report_commit_mismatch"):
        validate_v4c_report(changed)

    changed = copy.deepcopy(report)
    changed["safety"]["order_permission"] = True
    with pytest.raises(ValueError, match="v4c_safety_invariant_violation"):
        validate_v4c_report(changed)

    changed = copy.deepcopy(report)
    changed["triple_barrier_scenarios"][1]["horizon_candles"] = 10
    with pytest.raises(ValueError, match="v4c_scenario_geometry_mismatch"):
        validate_v4c_report(changed)


def test_build_transition_audit_checks_serialized_validation_labels() -> None:
    frames = {symbol: minute_frame() for symbol in SYMBOLS}
    report = _mini_report(frames)
    target = next(
        item
        for item in report["experiments"]
        if item["pair"] == "BTC/USDT"
        and item["timeframe"] == "15m"
        and item["scenario"] == "Baseline"
    )
    bars = resample_minute_candles(frames["BTCUSDT"], "15m")
    labels = apply_adaptive_triple_barrier(bars, horizon_candles=24)
    start = target["split"]["validation_start_index"]
    end = target["split"]["validation_end_index"]
    features = __import__(
        "services.ml_core.causal_features", fromlist=["build_causal_feature_matrix"]
    ).build_causal_feature_matrix(bars)
    indices = [
        index
        for index in range(start, end + 1)
        if bool(
            features.notna().all(axis=1).iloc[index]
            and labels["label"].iloc[index] in ("BUY", "SELL", "HOLD")
            and pd.notna(labels["label_end_idx"].iloc[index])
        )
    ]
    rows = [
        {
            "row_index": index,
            "actual_label": labels["label"].iloc[index],
            "timestamp": bars["date"].iloc[index].isoformat(),
        }
        for index in indices
    ]
    target["status"] = "TRAINED"
    target["model_variants"] = {
        "weight_none": {"validation_probabilities": rows},
        "weight_balanced": {"validation_probabilities": rows},
    }
    report["trained_experiments"] = 1
    report["trained_models"] = 2

    result = build_transition_audit(report, frames)
    assert len(result["experiments"]) == 8

    target["model_variants"]["weight_none"]["validation_probabilities"][0][
        "actual_label"
    ] = "SELL"
    with pytest.raises(ValueError, match="v4c_validation_actual_label_mismatch"):
        build_transition_audit(report, frames)
