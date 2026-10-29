"""Public-market diagnostic classification with ATR labels; no economic evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import metrics  # type: ignore[import-untyped]

from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.causal_features import build_causal_feature_matrix
from services.ml_core.purged_embargo import create_purged_embargo_split

SYMBOLS = ("BTCUSDT", "ETHUSDT")
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
LABELS = ("BUY", "SELL", "HOLD")
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
FEATURE_WINDOW = 14
VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85
OVERFIT_F1_GAP_THRESHOLD = 0.15
MIN_TRAIN_SAMPLES_PER_CLASS = 20
MIN_VALIDATION_SAMPLES_PER_CLASS = 5
REPORT_SCHEMA = "alix.ml-core.real-data-classification-diagnostic.v2"


def _validated_minute_timestamps(values: Series) -> Series:
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not bool(np.all(deltas == 60.0)):
        raise ValueError("minute_data_gap_or_interval_mismatch")
    return dates


def _validate_minute_numeric_values(result: DataFrame) -> None:
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    values = result.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(values).all()):
        raise ValueError("ohlcv_value_not_finite")
    _validate_minute_price_geometry(result)


def _validate_minute_price_geometry(result: DataFrame) -> None:
    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if bool(np.any(np.column_stack((opened, high, low, close)) <= 0)):
        raise ValueError("ohlcv_price_must_be_positive")
    if bool(np.any(volume < 0)):
        raise ValueError("ohlcv_volume_negative")
    invalid_geometry = (
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    )
    if bool(invalid_geometry):
        raise ValueError("ohlcv_geometry_invalid")


def _validate_minute_candles(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")
    result = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    result["date"] = _validated_minute_timestamps(result["date"])
    _validate_minute_numeric_values(result)
    return result


def resample_ohlcv(candles: DataFrame, timeframe: str) -> DataFrame:
    """Build target bars only from complete, gap-free one-minute public candles."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("entry_timeframe_out_of_policy")
    source = _validate_minute_candles(candles)
    minutes = TIMEFRAME_MINUTES[timeframe]
    indexed = source.set_index("date")
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
    complete = counts == minutes
    result = bars.loc[complete].dropna().reset_index()
    if result.empty:
        raise ValueError(f"no_complete_resampled_bars:{timeframe}")
    return result.loc[:, list(OHLCV_COLUMNS)]


def label_distribution(values: Series) -> dict[str, Any]:
    """Return counts and percentages for the fixed BUY/SELL/HOLD label contract."""
    observed = values.loc[values.isin(LABELS)]
    total = int(len(observed))
    counts = observed.value_counts()
    classes: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        count = int(counts.get(label, 0))
        classes[label] = {
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
        }
    return {"total": total, "classes": classes}


def classification_metrics(actual: Series, predicted: Series) -> dict[str, Any]:
    """Calculate class-wise and macro classification metrics only."""
    report = cast(
        dict[str, Any],
        metrics.classification_report(
            actual.tolist(),
            predicted.tolist(),
            labels=list(LABELS),
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    )
    macro = report["macro avg"]
    per_class: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        row = report[label]
        per_class[label] = {
            "precision": round(float(row["precision"]), 6),
            "recall": round(float(row["recall"]), 6),
            "f1": round(float(row["f1-score"]), 6),
            "support": int(row["support"]),
        }
    return {
        "macro": {
            "precision": round(float(macro["precision"]), 6),
            "recall": round(float(macro["recall"]), 6),
            "f1": round(float(macro["f1-score"]), 6),
        },
        "accuracy": round(float(report["accuracy"]), 6),
        "per_class": per_class,
    }


def _class_counts(values: Series) -> dict[str, int]:
    """Count each label in the fixed classification contract."""
    return {label: int((values == label).sum()) for label in LABELS}


def _classes_below_threshold(counts: dict[str, int], minimum: int) -> list[str]:
    """Return required classes with less than the requested sample count."""
    return [label for label in LABELS if counts[label] < minimum]


def sample_gate_diagnostics(
    train_labels: Series, validation_labels: Series
) -> dict[str, Any]:
    """Reject experiments without enough examples of every required class."""
    train_counts = _class_counts(train_labels)
    validation_counts = _class_counts(validation_labels)
    missing_train_classes = _classes_below_threshold(train_counts, 1)
    insufficient_train_classes = _classes_below_threshold(
        train_counts, MIN_TRAIN_SAMPLES_PER_CLASS
    )
    insufficient_validation_classes = _classes_below_threshold(
        validation_counts, MIN_VALIDATION_SAMPLES_PER_CLASS
    )

    if missing_train_classes:
        status = "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY"
    elif insufficient_train_classes or insufficient_validation_classes:
        status = "SKIPPED_INSUFFICIENT_SAMPLES"
    else:
        status = "PASS"

    return {
        "status": status,
        "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
        "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "missing_train_classes": missing_train_classes,
        "insufficient_train_classes": insufficient_train_classes,
        "insufficient_validation_classes": insufficient_validation_classes,
    }


def validation_classification_diagnostics(
    actual: Series, predicted: Series
) -> dict[str, Any]:
    """Return a fixed-order validation confusion matrix and prediction counts."""
    matrix = metrics.confusion_matrix(
        actual.tolist(), predicted.tolist(), labels=list(LABELS)
    )
    return {
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_are": "actual",
            "columns_are": "predicted",
            "matrix": [[int(value) for value in row] for row in matrix.tolist()],
        },
        "prediction_distribution": label_distribution(predicted),
    }


def _validation_window(row_count: int) -> tuple[int, int]:
    validation_start = int(row_count * VALIDATION_START_FRACTION)
    validation_end = int(row_count * VALIDATION_END_FRACTION) - 1
    if validation_end <= validation_start:
        raise ValueError("validation_window_invalid")
    return validation_start, validation_end


def _label_end_indices(labels: DataFrame) -> list[int]:
    return [
        index if pd.isna(value) else int(value)
        for index, value in enumerate(labels["label_end_idx"].tolist())
    ]


def _eligible_sample_indices(features: DataFrame, labels: DataFrame) -> set[int]:
    eligible_mask = (
        features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    return {int(index) for index in np.flatnonzero(eligible_mask.to_numpy())}


def _eligible_train_indices(
    train_candidates: tuple[int, ...],
    eligible_indices: set[int],
    train_limit: int,
) -> list[int]:
    return [
        index
        for index in train_candidates
        if index in eligible_indices and index < train_limit
    ]


def _eligible_validation_indices(
    validation_candidates: tuple[int, ...], eligible_indices: set[int]
) -> list[int]:
    return [index for index in validation_candidates if index in eligible_indices]


def _split_diagnostics(
    split: Any,
    eligible_indices: set[int],
    row_count: int,
    validation_start: int,
    validation_end: int,
    train_limit: int,
) -> dict[str, int]:
    purged_train_rows = sum(
        index in eligible_indices and index < train_limit
        for index in split.purged_indices
    )
    eligible_gap_rows = sum(
        index in eligible_indices
        for index in range(max(0, train_limit), validation_start)
    )
    return {
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "purged_train_rows": int(purged_train_rows),
        "embargo_size_bars": int(split.embargo_size),
        "eligible_pre_validation_embargo_rows": int(eligible_gap_rows),
        "post_validation_embargo_bars": len(split.embargo_indices),
        "future_rows_excluded_for_chronology": row_count - validation_end - 1,
    }


def purged_temporal_indices(
    features: DataFrame, labels: DataFrame
) -> tuple[list[int], list[int], dict[str, int]]:
    """Create a chronological split, actual-label-window purge, and conservative gap."""
    row_count = len(labels)
    if len(features) != row_count or row_count < 100:
        raise ValueError("insufficient_or_misaligned_rows_for_temporal_split")

    validation_start, validation_end = _validation_window(row_count)
    ends = _label_end_indices(labels)
    eligible_indices = _eligible_sample_indices(features, labels)
    split = create_purged_embargo_split(
        ends,
        validation_start=validation_start,
        validation_end=validation_end,
        longest_feature_window=FEATURE_WINDOW,
    )
    train_limit = validation_start - split.embargo_size
    train_indices = _eligible_train_indices(
        split.train_indices, eligible_indices, train_limit
    )
    validation_indices = _eligible_validation_indices(
        split.validation_indices, eligible_indices
    )
    diagnostics = _split_diagnostics(
        split,
        eligible_indices,
        row_count,
        validation_start,
        validation_end,
        train_limit,
    )
    return train_indices, validation_indices, diagnostics


def _new_classifier() -> Any:
    """Load the pinned tree classifier only inside the remote diagnostic workflow."""
    lightgbm = importlib.import_module("lightgbm")
    classifier = getattr(lightgbm, "LGBMClassifier")
    return classifier(
        objective="multiclass",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _skipped_result(
    pair: str,
    timeframe: str,
    bars: DataFrame,
    labels: DataFrame,
    train_indices: list[int],
    validation_indices: list[int],
    split_diagnostics: dict[str, int],
    reason: str,
    sample_gate: dict[str, Any],
) -> dict[str, Any]:
    train_labels = labels.iloc[train_indices]["label"]
    validation_labels = labels.iloc[validation_indices]["label"]
    return {
        "pair": pair,
        "timeframe": timeframe,
        "status": reason,
        "bar_count": int(len(bars)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "label_distribution": {
            "train": label_distribution(train_labels),
            "validation": label_distribution(validation_labels),
        },
        "classification_metrics": None,
        "confusion_matrix": None,
        "prediction_distribution": None,
        "sample_gate": sample_gate,
        "overfitting_assessment": {
            "status": "NOT_ASSESSED",
            "macro_f1_gap_train_minus_validation": None,
        },
        "split": split_diagnostics,
    }


def _insufficient_bar_result(
    pair: str, timeframe: str, bars: DataFrame, labels: DataFrame
) -> dict[str, Any]:
    """Report short real-data windows without pretending a train/validation split exists."""
    empty_distribution = label_distribution(Series(dtype=object))
    return {
        "pair": pair,
        "timeframe": timeframe,
        "status": "SKIPPED_INSUFFICIENT_SAMPLES",
        "bar_count": int(len(bars)),
        "train_rows": 0,
        "validation_rows": 0,
        "label_distribution": {
            "train": empty_distribution,
            "validation": empty_distribution,
            "available_unpartitioned": label_distribution(labels["label"]),
        },
        "classification_metrics": None,
        "confusion_matrix": None,
        "prediction_distribution": None,
        "sample_gate": {
            "status": "NOT_EVALUATED",
            "reason": "INSUFFICIENT_BARS",
            "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
            "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
            "train_class_counts": None,
            "validation_class_counts": None,
        },
        "overfitting_assessment": {
            "status": "NOT_ASSESSED",
            "macro_f1_gap_train_minus_validation": None,
        },
        "split": {
            "status": "NOT_SPLIT_INSUFFICIENT_BARS",
            "minimum_required_bars": 100,
        },
    }


def run_experiment(
    pair: str, timeframe: str, minute_candles: DataFrame
) -> dict[str, Any]:
    """Fit one tree classifier for one pair/timeframe and report classification only."""
    bars = resample_ohlcv(minute_candles, timeframe)
    labels = apply_adaptive_triple_barrier(bars)
    if len(bars) < 100:
        return _insufficient_bar_result(pair, timeframe, bars, labels)
    features = build_causal_feature_matrix(bars)
    train_indices, validation_indices, split_info = purged_temporal_indices(
        features, labels
    )
    train_y = labels.iloc[train_indices]["label"]
    validation_y = labels.iloc[validation_indices]["label"]
    sample_gate = sample_gate_diagnostics(train_y, validation_y)

    base_result: dict[str, Any] = {
        "pair": pair,
        "timeframe": timeframe,
        "bar_count": int(len(bars)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "label_distribution": {
            "train": label_distribution(train_y),
            "validation": label_distribution(validation_y),
        },
        "sample_gate": sample_gate,
        "split": split_info,
    }
    if sample_gate["status"] != "PASS":
        return _skipped_result(
            pair,
            timeframe,
            bars,
            labels,
            train_indices,
            validation_indices,
            split_info,
            str(sample_gate["status"]),
            sample_gate,
        )

    x_train = features.iloc[train_indices]
    x_validation = features.iloc[validation_indices]
    model = _new_classifier()
    model.fit(x_train, train_y)
    train_prediction = pd.Series(model.predict(x_train), index=train_y.index)
    validation_prediction = pd.Series(
        model.predict(x_validation), index=validation_y.index
    )
    train_metrics = classification_metrics(train_y, train_prediction)
    validation_metrics = classification_metrics(validation_y, validation_prediction)
    f1_gap = round(train_metrics["macro"]["f1"] - validation_metrics["macro"]["f1"], 6)
    overfitting_status = (
        "POSSIBLE_OVERFITTING"
        if f1_gap >= OVERFIT_F1_GAP_THRESHOLD
        else "NO_LARGE_MACRO_F1_GAP_OBSERVED"
    )
    return {
        **base_result,
        "status": "TRAINED",
        **validation_classification_diagnostics(validation_y, validation_prediction),
        "classification_metrics": {
            "train_in_sample": train_metrics,
            "validation": validation_metrics,
        },
        "overfitting_assessment": {
            "status": overfitting_status,
            "macro_f1_gap_train_minus_validation": f1_gap,
            "heuristic_threshold": OVERFIT_F1_GAP_THRESHOLD,
            "caveat": "Single chronological split; diagnostic heuristic, not a statistical proof.",
        },
    }


def _load_manifested_data(
    input_dir: Path, symbol: str, days: int, commit_sha: str
) -> tuple[DataFrame, dict[str, Any]]:
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"public_data_artifact_missing:{symbol}")

    manifest_bytes = manifest_path.read_bytes()
    manifest = cast(dict[str, Any], json.loads(manifest_bytes.decode("utf-8")))
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    expected_rows = days * 24 * 60
    checks = (
        manifest.get("symbol") == symbol,
        manifest.get("interval") == "1m",
        manifest.get("requested_days") == days,
        manifest.get("row_count") == expected_rows,
        manifest.get("commit_sha") == commit_sha,
        manifest.get("csv_sha256") == csv_digest,
        manifest.get("provider") == "Binance Spot public klines API",
        manifest.get("public_api_only") is True,
        manifest.get("credentials_used") is False,
        manifest.get("private_api_calls") == 0,
        manifest.get("real_order_attempts") == 0,
        manifest.get("gap_policy")
        == "reject; no interpolation, resampling, or synthetic fill",
        manifest.get("missing_minute_intervals") == 0,
        manifest.get("fetch_page_size_candles") == 1000,
        manifest.get("inter_page_delay_seconds") == 0.25,
    )
    if not all(checks):
        raise ValueError(f"public_data_manifest_validation_failed:{symbol}")

    candles = _validate_minute_candles(pd.read_csv(csv_path))
    if len(candles) != expected_rows:
        raise ValueError(f"public_data_row_count_mismatch:{symbol}")
    provenance = {
        "symbol": symbol,
        "pair": f"{symbol[:-4]}/USDT",
        "provider": manifest["provider"],
        "interval": "1m",
        "row_count": int(len(candles)),
        "requested_days": int(manifest["requested_days"]),
        "missing_minute_intervals": int(manifest["missing_minute_intervals"]),
        "fetch_page_size_candles": int(manifest["fetch_page_size_candles"]),
        "inter_page_delay_seconds": float(manifest["inter_page_delay_seconds"]),
        "start_utc": candles["date"].iloc[0].isoformat(),
        "end_utc": candles["date"].iloc[-1].isoformat(),
        "csv_sha256": csv_digest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "public_api_only": True,
        "credentials_used": False,
    }
    return candles, provenance


def _distribution_cell(distribution: dict[str, Any], label: str) -> str:
    row = distribution["classes"][label]
    return f"{row['percent']:.2f}% ({row['count']})"


def _render_label_distribution_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render actual target-label distributions for every experiment."""
    lines = [
        "### Label distribution",
        "",
        "| Pair | TF | Train n | BUY % (n) | SELL % (n) | HOLD % (n) | Validation n | BUY % (n) | SELL % (n) | HOLD % (n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in experiments:
        distribution = item["label_distribution"]
        train = distribution["train"]
        validation = distribution["validation"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {item['train_rows']} "
            f"| {_distribution_cell(train, 'BUY')} | {_distribution_cell(train, 'SELL')} "
            f"| {_distribution_cell(train, 'HOLD')} | {item['validation_rows']} "
            f"| {_distribution_cell(validation, 'BUY')} "
            f"| {_distribution_cell(validation, 'SELL')} "
            f"| {_distribution_cell(validation, 'HOLD')} |"
        )
    return lines


def _render_macro_metrics_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render macro metrics and the diagnostic Train/Validation F1 gap."""
    lines = [
        "### Macro classification metrics and overfitting screen",
        "",
        "| Pair | TF | Train Precision | Train Recall | Train F1 | Validation Precision | Validation Recall | Validation F1 | F1 gap | Assessment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in experiments:
        classification = item["classification_metrics"]
        if classification is None:
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | N/A | N/A | N/A "
                f"| N/A | N/A | N/A | N/A | {item['status']} |"
            )
            continue
        train = classification["train_in_sample"]["macro"]
        validation = classification["validation"]["macro"]
        gap = item["overfitting_assessment"]["macro_f1_gap_train_minus_validation"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {train['precision']:.4f} "
            f"| {train['recall']:.4f} | {train['f1']:.4f} "
            f"| {validation['precision']:.4f} | {validation['recall']:.4f} "
            f"| {validation['f1']:.4f} | {gap:.4f} "
            f"| {item['overfitting_assessment']['status']} |"
        )
    return lines


def _render_classwise_metrics_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render class-wise Validation metrics with per-class support."""
    lines = [
        "### Validation class-wise metrics",
        "",
        "| Pair | TF | Class | Precision | Recall | F1 | Support |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in experiments:
        classification = item["classification_metrics"]
        if classification is None:
            continue
        for label in LABELS:
            row = classification["validation"]["per_class"][label]
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {label} "
                f"| {row['precision']:.4f} | {row['recall']:.4f} "
                f"| {row['f1']:.4f} | {row['support']} |"
            )
    return lines


def _render_prediction_distribution_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render the model's predicted class distribution on Validation."""
    lines = [
        "### Validation prediction distribution",
        "",
        "| Pair | TF | Validation n | BUY % (n) | SELL % (n) | HOLD % (n) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in experiments:
        predictions = item.get("prediction_distribution")
        if predictions is None:
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | N/A | N/A | N/A | N/A |"
            )
            continue
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {predictions['total']} "
            f"| {_distribution_cell(predictions, 'BUY')} "
            f"| {_distribution_cell(predictions, 'SELL')} "
            f"| {_distribution_cell(predictions, 'HOLD')} |"
        )
    return lines


def _render_confusion_matrix_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render fixed-order Validation confusion matrices."""
    lines = [
        "### Validation confusion matrices",
        "",
        "| Pair | TF | Actual class (rows) | Predicted BUY | Predicted SELL | Predicted HOLD |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for item in experiments:
        matrix = item.get("confusion_matrix")
        for index, label in enumerate(LABELS):
            values = (
                ("N/A", "N/A", "N/A")
                if matrix is None
                else tuple(matrix["matrix"][index])
            )
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {label} "
                f"| {values[0]} | {values[1]} | {values[2]} |"
            )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render classification diagnostics for the GitHub Actions summary."""
    experiments = report["experiments"]
    lines = [
        "## AliX real-data diagnostic baseline",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Scope: classification diagnostics only. No economic metrics are calculated.",
        "",
    ]
    lines.extend(_render_label_distribution_section(experiments))
    lines.extend(["", *_render_macro_metrics_section(experiments)])
    lines.extend(["", *_render_classwise_metrics_section(experiments)])
    lines.extend(["", *_render_prediction_distribution_section(experiments)])
    lines.extend(["", *_render_confusion_matrix_section(experiments)])
    lines.extend(
        [
            "",
            "The overfitting screen uses the in-sample versus validation macro-F1 gap "
            f"of {OVERFIT_F1_GAP_THRESHOLD:.2f} as a heuristic. It is not proof.",
            "",
            "Safety: Champion=null, order_permission=false, "
            "trading_authorization=NO-GO, real_order_attempts=0, private_api_calls=0.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostic(
    input_dir: Path, output_path: Path, *, days: int, commit_sha: str
) -> dict[str, Any]:
    if days not in (1, 7, 30, 90, 730):
        raise ValueError("diagnostic_days_out_of_policy")
    datasets: dict[str, DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        candles, manifest = _load_manifested_data(input_dir, symbol, days, commit_sha)
        datasets[symbol] = candles
        provenance.append(manifest)

    experiments: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        pair = f"{symbol[:-4]}/USDT"
        for timeframe in TIMEFRAME_MINUTES:
            experiments.append(run_experiment(pair, timeframe, datasets[symbol]))

    trained_count = sum(item["status"] == "TRAINED" for item in experiments)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "diagnostic_scope": "CLASSIFICATION_ONLY",
        "training_scope": "DIAGNOSTIC_CLASSIFICATION_ONLY",
        "economic_evaluation_performed": False,
        "economic_metrics_reported": False,
        "requested_days": days,
        "data_window_selection": {
            "configured_days": int(
                os.environ.get("ALIX_DIAGNOSTIC_CONFIGURED_DAYS", str(days))
            ),
            "selected_days": days,
            "fallback_reason": os.environ.get(
                "ALIX_DIAGNOSTIC_FALLBACK_REASON", "NONE"
            ),
            "policy": (
                "Fetch the configured window in individually validated pages of "
                "at most 1,000 one-minute candles with a 0.25-second delay between "
                "pages; retry incomplete pages and transient HTTP failures with "
                "bounded retries; never shorten the requested window or synthesize "
                "missing candles."
            ),
            "fetch_page_size_candles": provenance[0]["fetch_page_size_candles"],
            "inter_page_delay_seconds": provenance[0]["inter_page_delay_seconds"],
        },
        "sample_gate_policy": {
            "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
            "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
            "required_classes": list(LABELS),
        },
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "allowed_timeframes": list(TIMEFRAME_MINUTES),
        "labels": list(LABELS),
        "model": {
            "family": "LightGBM LGBMClassifier",
            "n_estimators": 80,
            "max_depth": 4,
            "num_leaves": 15,
            "random_state": 42,
            "class_weight": "balanced",
        },
        "split_policy": {
            "validation_start_fraction": VALIDATION_START_FRACTION,
            "validation_end_fraction": VALIDATION_END_FRACTION,
            "chronological_training_only": True,
            "label_window_purge": "create_purged_embargo_split",
            "pre_validation_embargo": True,
            "post_validation_embargo": True,
            "label_horizon_candles": 24,
            "atr_period": 14,
            "take_profit_atr_multiple": 2.0,
            "stop_loss_atr_multiple": 1.0,
        },
        "trained_models": trained_count,
        "experiment_count": len(experiments),
        "experiments": experiments,
        "safety": {
            "public_api_only": True,
            "credentials_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(render_markdown(report))
    print(
        json.dumps(
            {
                "report_path": str(output_path),
                "commit_sha": commit_sha,
                "experiment_count": len(experiments),
                "trained_models": trained_count,
                "economic_evaluation_performed": False,
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(1, 7, 30, 90, 730), default=730)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_diagnostic(
        args.input_dir,
        args.output,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    if report["experiment_count"] != 8:
        raise RuntimeError("diagnostic_experiment_count_mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
