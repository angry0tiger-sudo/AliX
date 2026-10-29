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
from sklearn.metrics import classification_report  # type: ignore[import-untyped]

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
REPORT_SCHEMA = "alix.ml-core.real-data-classification-diagnostic.v1"


def _validate_minute_candles(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    result = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
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
    values = result.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(values).all()):
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
        classification_report(
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


def purged_temporal_indices(
    features: DataFrame, labels: DataFrame
) -> tuple[list[int], list[int], dict[str, int]]:
    """Create a chronological split, actual-label-window purge, and conservative gap."""
    row_count = len(labels)
    if len(features) != row_count or row_count < 100:
        raise ValueError("insufficient_or_misaligned_rows_for_temporal_split")

    validation_start = int(row_count * VALIDATION_START_FRACTION)
    validation_end = int(row_count * VALIDATION_END_FRACTION) - 1
    if validation_end <= validation_start:
        raise ValueError("validation_window_invalid")

    ends: list[int] = []
    for index, value in enumerate(labels["label_end_idx"].tolist()):
        ends.append(index if pd.isna(value) else int(value))

    eligible_mask = (
        features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    eligible_indices = {
        int(index) for index in np.flatnonzero(eligible_mask.to_numpy())
    }
    split = create_purged_embargo_split(
        ends,
        validation_start=validation_start,
        validation_end=validation_end,
        longest_feature_window=FEATURE_WINDOW,
    )

    train_limit = validation_start - split.embargo_size
    train_indices = [
        index
        for index in split.train_indices
        if index in eligible_indices and index < train_limit
    ]
    validation_indices = [
        index for index in split.validation_indices if index in eligible_indices
    ]
    purged_rows = [
        index
        for index in split.purged_indices
        if index in eligible_indices and index < train_limit
    ]
    eligible_gap_rows = sum(
        index in eligible_indices
        for index in range(max(0, train_limit), validation_start)
    )
    diagnostics = {
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "purged_train_rows": len(purged_rows),
        "embargo_size_bars": split.embargo_size,
        "eligible_pre_validation_embargo_rows": int(eligible_gap_rows),
        "post_validation_embargo_bars": len(split.embargo_indices),
        "future_rows_excluded_for_chronology": row_count - validation_end - 1,
    }
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
        "overfitting_assessment": {
            "status": "NOT_ASSESSED",
            "macro_f1_gap_train_minus_validation": None,
        },
        "split": split_diagnostics,
    }


def run_experiment(
    pair: str, timeframe: str, minute_candles: DataFrame
) -> dict[str, Any]:
    """Fit one tree classifier for one pair/timeframe and report classification only."""
    bars = resample_ohlcv(minute_candles, timeframe)
    features = build_causal_feature_matrix(bars)
    labels = apply_adaptive_triple_barrier(bars)
    train_indices, validation_indices, split_info = purged_temporal_indices(
        features, labels
    )
    train_y = labels.iloc[train_indices]["label"]
    validation_y = labels.iloc[validation_indices]["label"]

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
        "split": split_info,
    }
    if len(train_indices) < 50 or len(validation_indices) < 10:
        return _skipped_result(
            pair,
            timeframe,
            bars,
            labels,
            train_indices,
            validation_indices,
            split_info,
            "SKIPPED_INSUFFICIENT_SAMPLES",
        )
    if train_y.nunique() < 2:
        return _skipped_result(
            pair,
            timeframe,
            bars,
            labels,
            train_indices,
            validation_indices,
            split_info,
            "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY",
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
    stem = f"nobitex-{symbol.lower()}-1m"
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
        manifest.get("provider") == "Nobitex public UDF history API",
        manifest.get("public_api_only") is True,
        manifest.get("credentials_used") is False,
        manifest.get("private_api_calls") == 0,
        manifest.get("real_order_attempts") == 0,
        manifest.get("gap_policy")
        == "reject; no interpolation, resampling, or synthetic fill",
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


def render_markdown(report: dict[str, Any]) -> str:
    """Render label distributions and classification diagnostics for the Actions summary."""
    lines = [
        "## AliX real-data diagnostic baseline",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Scope: classification diagnostics only. No economic metrics are calculated.",
        "",
        "### Label distribution",
        "",
        "| Pair | TF | Train n | BUY % (n) | SELL % (n) | HOLD % (n) | Validation n | BUY % (n) | SELL % (n) | HOLD % (n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["experiments"]:
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
    lines.extend(
        [
            "",
            "### Macro classification metrics and overfitting screen",
            "",
            "| Pair | TF | Train Precision | Train Recall | Train F1 | Validation Precision | Validation Recall | Validation F1 | F1 gap | Assessment |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["experiments"]:
        metrics = item["classification_metrics"]
        if metrics is None:
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | N/A | N/A | N/A "
                f"| N/A | N/A | N/A | N/A | {item['status']} |"
            )
            continue
        train = metrics["train_in_sample"]["macro"]
        validation = metrics["validation"]["macro"]
        gap = item["overfitting_assessment"]["macro_f1_gap_train_minus_validation"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {train['precision']:.4f} "
            f"| {train['recall']:.4f} | {train['f1']:.4f} "
            f"| {validation['precision']:.4f} | {validation['recall']:.4f} "
            f"| {validation['f1']:.4f} | {gap:.4f} "
            f"| {item['overfitting_assessment']['status']} |"
        )
    lines.extend(
        [
            "",
            "### Validation class-wise metrics",
            "",
            "| Pair | TF | Class | Precision | Recall | F1 | Support |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["experiments"]:
        metrics = item["classification_metrics"]
        if metrics is None:
            continue
        for label in LABELS:
            row = metrics["validation"]["per_class"][label]
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {label} "
                f"| {row['precision']:.4f} | {row['recall']:.4f} "
                f"| {row['f1']:.4f} | {row['support']} |"
            )
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
    if days not in (30, 90):
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
        "provider": "Nobitex public UDF history API",
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
    parser.add_argument("--days", type=int, choices=(30, 90), default=90)
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
