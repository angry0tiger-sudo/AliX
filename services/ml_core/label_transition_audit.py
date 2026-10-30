"""Offline V4-C label-transition audit; this module never trains a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.ml_core.adaptive_targets import (
    EXIT_REASONS,
    UNLABELED_REASONS,
    apply_adaptive_triple_barrier,
)
from services.ml_core.causal_features import build_causal_feature_matrix

REPORT_SCHEMA = "alix.ml-core.real-data-classification-diagnostic.v5"
AUDIT_SCHEMA = "alix.ml-core.label-transition-audit.v1"
SOURCE_COMMIT = "ac428dd94383173570fe3cf8af510e5b0941336e"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PAIR_NAMES = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}
TIMEFRAMES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
SCENARIOS = {
    "Baseline": (24, 2.0, 1.0),
    "Short_Horizon": (12, 2.0, 1.0),
}
LABELS = ("BUY", "SELL", "HOLD")
MINUTE_COLUMNS = ("date", "open", "high", "low", "close", "volume")
VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85
WEIGHTING_POLICIES = ("weight_none", "weight_balanced")
SAFE_VALUES = {
    "Champion": None,
    "credentials_used": False,
    "order_permission": False,
    "orders_or_execution_api_used": False,
    "private_api_calls": 0,
    "public_api_only": True,
    "real_order_attempts": 0,
    "trading_authorization": "NO-GO",
}


def _mapping(value: Any, error: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(error)
    return cast(dict[str, Any], value)


def _integer(value: Any, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(error)
    return value


def validate_v4c_report(
    report: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Validate provenance, experiment matrix, scenario geometry and safety gates."""
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("v4c_report_schema_mismatch")
    if report.get("commit_sha") != SOURCE_COMMIT:
        raise ValueError("v4c_report_commit_mismatch")
    if report.get("experiment_count") != 16:
        raise ValueError("v4c_experiment_count_mismatch")

    window = _mapping(
        report.get("data_window_selection"), "data_window_metadata_missing"
    )
    selected_days = _integer(window.get("selected_days"), "selected_days_invalid")
    if selected_days < 1 or selected_days != report.get("requested_days"):
        raise ValueError("v4c_data_window_mismatch")

    safety = _mapping(report.get("safety"), "safety_metadata_missing")
    if any(safety.get(key) != value for key, value in SAFE_VALUES.items()):
        raise ValueError("v4c_safety_invariant_violation")

    scenario_rows = report.get("triple_barrier_scenarios")
    if not isinstance(scenario_rows, list):
        raise ValueError("v4c_scenario_metadata_missing")
    observed_scenarios: dict[str, tuple[int, float, float]] = {}
    for raw in scenario_rows:
        scenario = _mapping(raw, "v4c_scenario_metadata_invalid")
        name = str(scenario.get("name"))
        if name not in SCENARIOS:
            raise ValueError("v4c_scenario_out_of_policy")
        observed_scenarios[name] = (
            _integer(scenario.get("horizon_candles"), "v4c_horizon_invalid"),
            float(scenario.get("take_profit_multiplier")),
            float(scenario.get("stop_loss_multiplier")),
        )
    if observed_scenarios != SCENARIOS:
        raise ValueError("v4c_scenario_geometry_mismatch")

    raw_experiments = report.get("experiments")
    if not isinstance(raw_experiments, list) or len(raw_experiments) != 16:
        raise ValueError("v4c_experiment_matrix_invalid")

    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    trained_experiments = 0
    trained_models = 0
    for raw in raw_experiments:
        item = _mapping(raw, "v4c_experiment_invalid")
        pair = str(item.get("pair"))
        symbol = next(
            (
                candidate
                for candidate, pair_name in PAIR_NAMES.items()
                if pair_name == pair
            ),
            "",
        )
        timeframe = str(item.get("timeframe"))
        scenario_name = str(item.get("scenario"))
        if (
            symbol not in SYMBOLS
            or timeframe not in TIMEFRAMES
            or scenario_name not in SCENARIOS
        ):
            raise ValueError("v4c_experiment_key_out_of_policy")
        key = (symbol, timeframe, scenario_name)
        if key in indexed:
            raise ValueError("v4c_duplicate_experiment")
        expected = SCENARIOS[scenario_name]
        barrier = _mapping(item.get("triple_barrier"), "v4c_barrier_metadata_missing")
        actual = (
            _integer(barrier.get("horizon_candles"), "v4c_horizon_invalid"),
            float(barrier.get("take_profit_multiplier")),
            float(barrier.get("stop_loss_multiplier")),
        )
        if actual != expected:
            raise ValueError("v4c_experiment_barrier_mismatch")

        status = item.get("status")
        if status == "TRAINED":
            trained_experiments += 1
            variants = _mapping(
                item.get("model_variants"), "v4c_trained_models_missing"
            )
            if set(variants) != set(WEIGHTING_POLICIES):
                raise ValueError("v4c_weighting_policy_matrix_invalid")
            trained_models += len(variants)
        elif status in {
            "SKIPPED_INSUFFICIENT_SAMPLES",
            "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY",
        }:
            if item.get("model_variants") is not None:
                raise ValueError("v4c_skipped_experiment_has_models")
        else:
            raise ValueError("v4c_experiment_status_invalid")
        indexed[key] = item

    expected_keys = {
        (symbol, timeframe, scenario)
        for symbol in SYMBOLS
        for timeframe in TIMEFRAMES
        for scenario in SCENARIOS
    }
    if set(indexed) != expected_keys:
        raise ValueError("v4c_experiment_matrix_incomplete")
    if report.get("trained_experiments") != trained_experiments:
        raise ValueError("v4c_trained_experiment_count_mismatch")
    if report.get("trained_models") != trained_models:
        raise ValueError("v4c_trained_model_count_mismatch")
    return indexed


def validate_minute_candles(candles: DataFrame, expected_rows: int) -> DataFrame:
    """Validate the original gap-free one-minute source before any resampling."""
    missing = sorted(set(MINUTE_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    result = candles.loc[:, list(MINUTE_COLUMNS)].copy().reset_index(drop=True)
    dates = pd.to_datetime(result["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not bool(np.all(deltas == 60.0)):
        raise ValueError("minute_data_gap_or_interval_mismatch")
    result["date"] = dates

    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    numeric = result.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(numeric).all()):
        raise ValueError("ohlcv_value_not_finite")
    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if bool(np.any(np.column_stack((opened, high, low, close)) <= 0)):
        raise ValueError("ohlcv_price_must_be_positive")
    if bool(np.any(volume < 0)):
        raise ValueError("ohlcv_volume_negative")
    if bool(
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    ):
        raise ValueError("ohlcv_geometry_invalid")
    if len(result) != expected_rows:
        raise ValueError("minute_candle_count_mismatch")
    return result


def resample_minute_candles(candles: DataFrame, timeframe: str) -> DataFrame:
    """Mirror V4-C's epoch-aligned, complete-bar resampling without model imports."""
    if timeframe not in TIMEFRAMES:
        raise ValueError("entry_timeframe_out_of_policy")
    minutes = TIMEFRAMES[timeframe]
    indexed = candles.set_index("date")
    rule = f"{minutes}min"
    grouped = indexed.resample(rule, origin="epoch", closed="left", label="left")
    bars = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    result = bars.loc[counts == minutes].dropna().reset_index()
    if result.empty:
        raise ValueError(f"no_complete_resampled_bars:{timeframe}")
    return result.loc[:, list(MINUTE_COLUMNS)]


def _label_audit_counts(labels: DataFrame) -> dict[str, Any]:
    labeled = labels["label"].isin(LABELS)
    exits = labels["exit_reason"].dropna()
    unlabeled = labels["unlabeled_reason"].dropna()
    return {
        "labeled_events": int(labeled.sum()),
        "unlabeled_events": int(labels["label"].isna().sum()),
        "exit_reasons": {
            reason: int((exits == reason).sum()) for reason in EXIT_REASONS
        },
        "unlabeled_reasons": {
            reason: int((unlabeled == reason).sum()) for reason in UNLABELED_REASONS
        },
    }


def _validate_label_audit(labels: DataFrame, experiment: dict[str, Any]) -> None:
    actual = _label_audit_counts(labels)
    reported = _mapping(experiment.get("label_audit"), "v4c_label_audit_missing")
    exits = _mapping(
        _mapping(
            reported.get("exit_reason_distribution"), "v4c_exit_audit_missing"
        ).get("reasons"),
        "v4c_exit_reason_counts_missing",
    )
    unlabeled = _mapping(
        _mapping(
            reported.get("unlabeled_reason_distribution"),
            "v4c_unlabeled_audit_missing",
        ).get("reasons"),
        "v4c_unlabeled_reason_counts_missing",
    )
    if actual["labeled_events"] != reported.get("labeled_events"):
        raise ValueError("v4c_labeled_event_count_mismatch")
    if actual["unlabeled_events"] != reported.get("unlabeled_events"):
        raise ValueError("v4c_unlabeled_event_count_mismatch")
    for reason in EXIT_REASONS:
        if actual["exit_reasons"][reason] != exits.get(reason, {}).get("count"):
            raise ValueError("v4c_exit_reason_count_mismatch")
    for reason in UNLABELED_REASONS:
        if actual["unlabeled_reasons"][reason] != unlabeled.get(reason, {}).get(
            "count"
        ):
            raise ValueError("v4c_unlabeled_reason_count_mismatch")


def _eligible_indices(
    bars: DataFrame, labels: DataFrame, start_index: int, end_index: int
) -> list[int]:
    features = build_causal_feature_matrix(bars)
    if len(features) != len(labels):
        raise ValueError("feature_label_row_count_mismatch")
    eligible = (
        features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    return [
        index
        for index in range(start_index, end_index + 1)
        if bool(eligible.iloc[index])
    ]


def _validate_label_distribution(
    labels: DataFrame, indices: list[int], experiment: dict[str, Any]
) -> None:
    observed = labels.iloc[indices]["label"]
    distribution = _mapping(
        _mapping(
            experiment.get("label_distribution"), "v4c_label_distribution_missing"
        ).get("validation"),
        "v4c_validation_label_distribution_missing",
    )
    classes = _mapping(
        distribution.get("classes"), "v4c_validation_label_counts_missing"
    )
    if distribution.get("total") != len(indices):
        raise ValueError("v4c_validation_row_count_mismatch")
    for label in LABELS:
        expected = _integer(
            _mapping(classes.get(label), "v4c_validation_class_missing").get("count"),
            "v4c_validation_class_count_invalid",
        )
        actual = int((observed == label).sum())
        if actual != expected:
            raise ValueError("v4c_validation_label_count_mismatch")


def _validate_serialized_validation(
    experiment: dict[str, Any],
    labels: DataFrame,
    bars: DataFrame,
    indices: list[int],
) -> None:
    variants = _mapping(experiment.get("model_variants"), "v4c_trained_models_missing")
    expected_indices = indices
    for policy in WEIGHTING_POLICIES:
        variant = _mapping(variants.get(policy), "v4c_model_variant_missing")
        raw_rows = variant.get("validation_probabilities")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(expected_indices):
            raise ValueError("v4c_validation_probability_rows_mismatch")
        for raw, expected_index in zip(raw_rows, expected_indices, strict=True):
            row = _mapping(raw, "v4c_validation_probability_row_invalid")
            row_index = _integer(
                row.get("row_index"), "v4c_validation_row_index_invalid"
            )
            if row_index != expected_index:
                raise ValueError("v4c_validation_row_index_mismatch")
            if row.get("actual_label") != labels.iloc[row_index]["label"]:
                raise ValueError("v4c_validation_actual_label_mismatch")
            timestamp = pd.Timestamp(row.get("timestamp"))
            if timestamp != pd.Timestamp(bars["date"].iloc[row_index]):
                raise ValueError("v4c_validation_timestamp_mismatch")


def collect_transition_events(
    bars: DataFrame,
    baseline_labels: DataFrame,
    short_labels: DataFrame,
    validation_indices: list[int],
) -> list[dict[str, Any]]:
    """Return only 24-candle directional hits that become 12-candle HOLD expiry."""
    events: list[dict[str, Any]] = []
    for index in validation_indices:
        baseline = baseline_labels.iloc[index]
        short = short_labels.iloc[index]
        baseline_label = str(baseline["label"])
        short_label = str(short["label"])
        if baseline_label not in {"BUY", "SELL"} or short_label != "HOLD":
            continue

        offset = int(baseline["label_end_idx"]) - index
        if not 13 <= offset <= 24:
            raise ValueError("transition_hit_offset_out_of_late_window")
        if short["exit_reason"] != "HORIZON_EXPIRY":
            raise ValueError("short_horizon_hold_reason_mismatch")
        if int(short["label_end_idx"]) - index != 12:
            raise ValueError("short_horizon_expiry_offset_mismatch")
        reason = str(baseline["exit_reason"])
        if baseline_label == "BUY" and reason != "TP_HIT":
            raise ValueError("baseline_buy_exit_reason_mismatch")
        if baseline_label == "SELL" and reason not in {"SL_HIT", "COLLISION"}:
            raise ValueError("baseline_sell_exit_reason_mismatch")

        bucket = "13-18" if offset <= 18 else "19-24"
        hit_index = int(baseline["label_end_idx"])
        events.append(
            {
                "row_index": int(index),
                "event_timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
                "baseline_label": baseline_label,
                "short_horizon_label": "HOLD",
                "short_horizon_exit_reason": "HORIZON_EXPIRY",
                "late_hit_candle_offset": offset,
                "late_hit_bucket": bucket,
                "first_touch_direction": baseline_label,
                "first_touch_reason": reason,
                "first_touch_row_index": hit_index,
                "first_touch_timestamp": pd.Timestamp(
                    bars["date"].iloc[hit_index]
                ).isoformat(),
            }
        )
    return events


def _transition_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"13-18": 0, "19-24": 0}
    directions = {"BUY": 0, "SELL": 0}
    reasons = {reason: 0 for reason in ("TP_HIT", "SL_HIT", "COLLISION")}
    for event in events:
        buckets[str(event["late_hit_bucket"])] += 1
        directions[str(event["first_touch_direction"])] += 1
        reasons[str(event["first_touch_reason"])] += 1
    return {
        "total": len(events),
        "late_hit_candle_buckets": buckets,
        "first_touch_direction_counts": directions,
        "first_touch_reason_counts": reasons,
    }


def build_transition_audit(
    report: dict[str, Any],
    minute_frames: dict[str, DataFrame],
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Rebuild both V4-C label sets from source OHLCV and audit the validation window."""
    indexed = validate_v4c_report(report)
    selected_days = _integer(
        _mapping(
            report.get("data_window_selection"), "data_window_metadata_missing"
        ).get("selected_days"),
        "selected_days_invalid",
    )
    expected_minute_rows = selected_days * 24 * 60
    results: list[dict[str, Any]] = []
    source_data: dict[str, Any] = {}

    for symbol in SYMBOLS:
        if symbol not in minute_frames:
            raise ValueError(f"minute_data_missing:{symbol}")
        candles = validate_minute_candles(minute_frames[symbol], expected_minute_rows)
        source_data[symbol] = {
            "minute_rows": len(candles),
            "first_timestamp": pd.Timestamp(candles["date"].iloc[0]).isoformat(),
            "last_timestamp": pd.Timestamp(candles["date"].iloc[-1]).isoformat(),
            "sha256": (source_hashes or {}).get(symbol),
        }
        for timeframe in TIMEFRAMES:
            bars = resample_minute_candles(candles, timeframe)
            pair_results = {
                scenario: indexed[(symbol, timeframe, scenario)]
                for scenario in SCENARIOS
            }
            if any(
                item.get("bar_count") != len(bars) for item in pair_results.values()
            ):
                raise ValueError("v4c_resampled_bar_count_mismatch")

            start = int(len(bars) * VALIDATION_START_FRACTION)
            end = int(len(bars) * VALIDATION_END_FRACTION) - 1
            if end <= start:
                raise ValueError("v4c_validation_window_invalid")

            labels_by_scenario: dict[str, DataFrame] = {}
            indices_by_scenario: dict[str, list[int]] = {}
            for scenario_name, (horizon, take_profit, stop_loss) in SCENARIOS.items():
                experiment = pair_results[scenario_name]
                split = _mapping(experiment.get("split"), "v4c_split_metadata_missing")
                if split.get("validation_start_index") != start:
                    raise ValueError("v4c_validation_start_index_mismatch")
                if split.get("validation_end_index") != end:
                    raise ValueError("v4c_validation_end_index_mismatch")
                labels = apply_adaptive_triple_barrier(
                    bars,
                    horizon_candles=horizon,
                    take_profit_multiplier=take_profit,
                    stop_loss_multiplier=stop_loss,
                )
                _validate_label_audit(labels, experiment)
                valid_indices = _eligible_indices(bars, labels, start, end)
                if experiment.get("validation_rows") != len(valid_indices):
                    raise ValueError("v4c_validation_eligible_row_count_mismatch")
                _validate_label_distribution(labels, valid_indices, experiment)
                if experiment.get("status") == "TRAINED":
                    _validate_serialized_validation(
                        experiment, labels, bars, valid_indices
                    )
                labels_by_scenario[scenario_name] = labels
                indices_by_scenario[scenario_name] = valid_indices

            validation_indices = sorted(
                set(indices_by_scenario["Baseline"])
                & set(indices_by_scenario["Short_Horizon"])
            )
            events = collect_transition_events(
                bars,
                labels_by_scenario["Baseline"],
                labels_by_scenario["Short_Horizon"],
                validation_indices,
            )
            results.append(
                {
                    "symbol": symbol,
                    "pair": PAIR_NAMES[symbol],
                    "timeframe": timeframe,
                    "validation_window": {
                        "start_row_index": start,
                        "end_row_index": end,
                        "shared_eligible_rows": len(validation_indices),
                    },
                    "baseline": {
                        "horizon_candles": SCENARIOS["Baseline"][0],
                        "status": pair_results["Baseline"]["status"],
                    },
                    "short_horizon": {
                        "horizon_candles": SCENARIOS["Short_Horizon"][0],
                        "status": pair_results["Short_Horizon"]["status"],
                    },
                    "transition_summary": _transition_summary(events),
                    "events": events,
                }
            )

    return {
        "schema_version": AUDIT_SCHEMA,
        "source_artifact": {
            "schema_version": REPORT_SCHEMA,
            "commit_sha": SOURCE_COMMIT,
            "requested_days": report["requested_days"],
        },
        "source_data": source_data,
        "audit_scope": "V4_C_CHRONOLOGICAL_VALIDATION_WINDOW",
        "transition_definition": (
            "Baseline label BUY/SELL with first TP/SL touch at candles 13-24, "
            "while Short_Horizon label is HOLD/HORIZON_EXPIRY at candle 12."
        ),
        "experiments": results,
        "aggregate_transition_summary": _transition_summary(
            [event for experiment in results for event in experiment["events"]]
        ),
        "safety": dict(SAFE_VALUES),
        "model_training_performed": False,
        "production_target_modified": False,
    }


def audit_report_files(
    report_path: Path,
    btc_candles_path: Path,
    eth_candles_path: Path,
) -> dict[str, Any]:
    """Load report and matching raw CSV inputs, then build the auditable JSON result."""
    report_bytes = report_path.read_bytes()
    try:
        report = _mapping(json.loads(report_bytes), "v4c_report_json_invalid")
    except json.JSONDecodeError as exc:
        raise ValueError("v4c_report_json_invalid") from exc
    frames: dict[str, DataFrame] = {}
    hashes: dict[str, str] = {}
    for symbol, path in (
        ("BTCUSDT", btc_candles_path),
        ("ETHUSDT", eth_candles_path),
    ):
        file_bytes = path.read_bytes()
        hashes[symbol] = hashlib.sha256(file_bytes).hexdigest()
        frames[symbol] = pd.read_csv(path)
    result = build_transition_audit(report, frames, hashes)
    result["source_artifact"]["sha256"] = hashlib.sha256(report_bytes).hexdigest()
    result["source_artifact"]["filename"] = report_path.name
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit V4-C 24-to-12 candle label transitions using original OHLCV."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--btc-1m-csv", type=Path, required=True)
    parser.add_argument("--eth-1m-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_report_files(
            args.report,
            args.btc_1m_csv,
            args.eth_1m_csv,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "experiments": len(result["experiments"]),
                "transitions": result["aggregate_transition_summary"]["total"],
                "model_training_performed": False,
                "Champion": result["safety"]["Champion"],
                "trading_authorization": result["safety"]["trading_authorization"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
