"""Evidence-producing audit of the V4-E 12-candle hierarchical target contract."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.ml_core.adaptive_targets import (
    EXIT_REASONS,
    HIERARCHICAL_HORIZON_CANDLES,
    LABEL_HORIZON_CANDLES,
    STAGE1_TARGET_COLUMN,
    STAGE2_TARGET_COLUMN,
    UNLABELED_REASONS,
    apply_adaptive_hierarchical_targets,
    apply_adaptive_triple_barrier,
)

OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
ALLOWED_DAYS = (1, 7, 30, 90, 730)
DATA_SCHEMA = "alix.binance.public-ohlcv.v1"
REPORT_SCHEMA = "alix.ml-core.hierarchical-target-audit.v1"
PROVIDER = "Binance Spot public klines API"
PUBLIC_HOST = "data-api.binance.vision"
PUBLIC_ENDPOINT = "/api/v3/klines"
GAP_POLICY = "reject; no interpolation, resampling, or synthetic fill"


def _parse_manifest_timestamp(value: object, field: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest_timestamp_invalid:{field}")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"manifest_timestamp_invalid:{field}")
    return pd.Timestamp(parsed)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    symbol: str,
    days: int,
    commit_sha: str,
    csv_digest: str,
) -> None:
    expected_rows = days * 1440
    expected_values = (
        (manifest.get("schema_version") == DATA_SCHEMA, "schema_version"),
        (manifest.get("provider") == PROVIDER, "provider"),
        (manifest.get("host") == PUBLIC_HOST, "host"),
        (manifest.get("endpoint") == PUBLIC_ENDPOINT, "endpoint"),
        (manifest.get("symbol") == symbol, "symbol"),
        (manifest.get("pair") == f"{symbol[:-4]}/USDT", "pair"),
        (manifest.get("interval") == "1m", "interval"),
        (manifest.get("requested_days") == days, "requested_days"),
        (manifest.get("row_count") == expected_rows, "row_count"),
        (manifest.get("missing_intervals") == 0, "missing_intervals"),
        (manifest.get("missing_minute_intervals") == 0, "missing_minute_intervals"),
        (manifest.get("gap_policy") == GAP_POLICY, "gap_policy"),
        (manifest.get("fetch_page_size_candles") == 1000, "fetch_page_size_candles"),
        (manifest.get("inter_page_delay_seconds") == 0.25, "inter_page_delay_seconds"),
        (manifest.get("csv_sha256") == csv_digest, "csv_sha256"),
        (manifest.get("commit_sha") == commit_sha, "commit_sha"),
        (
            manifest.get("data_authority")
            == "PUBLIC_MARKET_DATA_ONLY_NOT_ACCOUNT_ORDER_OR_FILL_TRUTH",
            "data_authority",
        ),
        (manifest.get("public_api_only") is True, "public_api_only"),
        (manifest.get("credentials_used") is False, "credentials_used"),
        (manifest.get("private_api_calls") == 0, "private_api_calls"),
        (manifest.get("exchange_calls") == 0, "exchange_calls"),
        (manifest.get("real_order_attempts") == 0, "real_order_attempts"),
        (manifest.get("order_permission") is False, "order_permission"),
        (manifest.get("Champion") is None, "Champion"),
        (manifest.get("trading_authorization") == "NO-GO", "trading_authorization"),
    )
    failures = [name for valid, name in expected_values if not valid]
    if failures:
        raise ValueError("public_data_manifest_validation_failed:" + ",".join(failures))


def _validate_minute_frame(candles: DataFrame, *, expected_rows: int) -> DataFrame:
    if tuple(candles.columns) != OHLCV_COLUMNS:
        raise ValueError("public_data_ohlcv_schema_invalid")
    if len(candles) != expected_rows or candles.empty:
        raise ValueError("public_data_row_count_mismatch")

    result = candles.copy()
    dates = pd.to_datetime(result["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("public_data_timestamp_invalid")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("public_data_timestamps_not_unique_and_increasing")
    timestamp_ns = dates.astype("int64").to_numpy()
    if len(timestamp_ns) > 1 and not np.all(np.diff(timestamp_ns) == 60_000_000_000):
        raise ValueError("public_data_minute_gap_detected")
    result["date"] = dates

    for column in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(result[column], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"public_data_non_finite:{column}")
        result[column] = values

    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if (
        np.any(opened <= 0)
        or np.any(high <= 0)
        or np.any(low <= 0)
        or np.any(close <= 0)
        or np.any(volume < 0)
    ):
        raise ValueError("public_data_price_or_volume_out_of_range")
    if (
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    ):
        raise ValueError("public_data_ohlc_geometry_invalid")
    return result.loc[:, list(OHLCV_COLUMNS)].reset_index(drop=True)


def load_manifested_dataset(
    input_dir: Path,
    *,
    symbol: str,
    days: int,
    commit_sha: str,
) -> tuple[DataFrame, dict[str, Any]]:
    """Load one exact public Binance dataset and fail closed on provenance gaps."""
    if symbol not in SYMBOLS:
        raise ValueError("public_data_symbol_out_of_policy")
    if isinstance(days, bool) or days not in ALLOWED_DAYS:
        raise ValueError("public_data_days_out_of_policy")
    if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
        raise ValueError("commit_sha_invalid")

    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"public_data_artifact_missing:{symbol}")

    manifest_bytes = manifest_path.read_bytes()
    manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("public_data_manifest_invalid")
    manifest = manifest_payload
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _validate_manifest(
        manifest,
        symbol=symbol,
        days=days,
        commit_sha=commit_sha,
        csv_digest=csv_digest,
    )

    raw = pd.read_csv(csv_path)
    candles = _validate_minute_frame(raw, expected_rows=days * 1440)
    manifest_start = _parse_manifest_timestamp(manifest.get("start_utc"), "start_utc")
    manifest_end = _parse_manifest_timestamp(manifest.get("end_utc"), "end_utc")
    if candles["date"].iloc[0] != manifest_start:
        raise ValueError("public_data_start_timestamp_mismatch")
    if candles["date"].iloc[-1] != manifest_end:
        raise ValueError("public_data_end_timestamp_mismatch")

    provenance = {
        "symbol": symbol,
        "pair": str(manifest["pair"]),
        "provider": PROVIDER,
        "interval": "1m",
        "requested_days": days,
        "row_count": int(len(candles)),
        "start_utc": manifest_start.isoformat(),
        "end_utc": manifest_end.isoformat(),
        "csv_sha256": csv_digest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "missing_minute_intervals": 0,
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
    }
    return candles, provenance


def resample_ohlcv(candles: DataFrame, timeframe: str) -> DataFrame:
    """Create only complete epoch-aligned bars from validated minute candles."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("entry_timeframe_out_of_policy")
    minutes = TIMEFRAME_MINUTES[timeframe]
    source = _validate_minute_frame(candles, expected_rows=len(candles))
    indexed = source.set_index("date")
    grouped = indexed.resample(
        f"{minutes}min", origin="epoch", closed="left", label="left"
    )
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
    if tuple(result.columns) != OHLCV_COLUMNS:
        raise ValueError("resampled_ohlcv_schema_invalid")
    timestamp_ns = pd.to_datetime(result["date"], utc=True).astype("int64").to_numpy()
    expected_delta = minutes * 60_000_000_000
    if len(timestamp_ns) > 1 and not np.all(np.diff(timestamp_ns) == expected_delta):
        raise ValueError(f"resampled_timeframe_gap_detected:{timeframe}")
    return result.loc[:, list(OHLCV_COLUMNS)]


def _expected_stage_targets(labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    stage1 = np.full(len(labels), np.nan, dtype=float)
    stage2 = np.full(len(labels), np.nan, dtype=float)
    for index, label in enumerate(labels.tolist()):
        if label in ("BUY", "SELL"):
            stage1[index] = 1.0
            stage2[index] = 1.0 if label == "BUY" else 0.0
        elif label == "HOLD":
            stage1[index] = 0.0
        elif not pd.isna(label):
            raise ValueError("hierarchical_target_label_out_of_policy")
    return stage1, stage2


def _nullable_values_match(actual: pd.Series, expected: np.ndarray) -> bool:
    values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    return values.shape == expected.shape and bool(
        np.array_equal(values, expected, equal_nan=True)
    )


def _validate_label_end_indices(targets: DataFrame) -> int:
    maximum_offset = 0
    for position, (_, row) in enumerate(targets.iterrows()):
        label = row["label"]
        end_value = row["label_end_idx"]
        reason = row["unlabeled_reason"]
        if pd.isna(label):
            if not pd.isna(end_value) or reason not in UNLABELED_REASONS:
                raise ValueError("hierarchical_unlabeled_row_contract_invalid")
            continue
        if pd.isna(end_value):
            raise ValueError("hierarchical_label_end_index_missing")
        end_number = float(end_value)
        if not np.isfinite(end_number) or not end_number.is_integer():
            raise ValueError("hierarchical_label_end_index_invalid")
        offset = int(end_number) - position
        if offset < 1 or offset > HIERARCHICAL_HORIZON_CANDLES:
            raise ValueError("hierarchical_label_end_index_out_of_horizon")
        if label == "HOLD" and offset != HIERARCHICAL_HORIZON_CANDLES:
            raise ValueError("hierarchical_no_hit_end_index_invalid")
        if label in ("BUY", "SELL") and reason not in EXIT_REASONS[:3]:
            raise ValueError("hierarchical_hit_exit_reason_invalid")
        maximum_offset = max(maximum_offset, offset)
    return maximum_offset


def _count_late_hits(short_targets: DataFrame, long_targets: DataFrame) -> int:
    late_count = 0
    for position, (_, row) in enumerate(long_targets.iterrows()):
        if row["label"] not in ("BUY", "SELL"):
            continue
        end_value = row["label_end_idx"]
        if pd.isna(end_value):
            raise ValueError("legacy_hit_end_index_missing")
        offset = int(end_value) - position
        if offset < 1 or offset > LABEL_HORIZON_CANDLES:
            raise ValueError("legacy_hit_end_index_out_of_horizon")
        short = short_targets.iloc[position]
        if offset <= HIERARCHICAL_HORIZON_CANDLES:
            if short["label"] != row["label"] or short["label_end_idx"] != end_value:
                raise ValueError("hierarchical_early_hit_mismatch")
        else:
            if (
                short["label"] != "HOLD"
                or short["exit_reason"] != "HORIZON_EXPIRY"
                or short["label_end_idx"] != position + HIERARCHICAL_HORIZON_CANDLES
            ):
                raise ValueError("hierarchical_late_hit_must_remain_no_hit")
            late_count += 1
    return late_count


def audit_hierarchical_targets(
    candles: DataFrame, *, pair: str, timeframe: str
) -> dict[str, Any]:
    """Verify V4-E targets against independent 12- and 24-candle label runs."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("entry_timeframe_out_of_policy")
    if not pair or "/" not in pair:
        raise ValueError("pair_invalid")

    targets = apply_adaptive_hierarchical_targets(candles)
    short_reference = apply_adaptive_triple_barrier(
        candles, horizon_candles=HIERARCHICAL_HORIZON_CANDLES
    )
    long_reference = apply_adaptive_triple_barrier(
        candles, horizon_candles=LABEL_HORIZON_CANDLES
    )
    for column in ("label", "exit_reason", "unlabeled_reason", "label_end_idx"):
        if not targets[column].equals(short_reference[column]):
            raise ValueError(f"hierarchical_base_target_mismatch:{column}")

    expected_stage1, expected_stage2 = _expected_stage_targets(short_reference["label"])
    if not _nullable_values_match(targets[STAGE1_TARGET_COLUMN], expected_stage1):
        raise ValueError("hierarchical_stage1_mapping_invalid")
    if not _nullable_values_match(targets[STAGE2_TARGET_COLUMN], expected_stage2):
        raise ValueError("hierarchical_stage2_mapping_invalid")

    maximum_offset = _validate_label_end_indices(targets)
    late_hit_count = _count_late_hits(short_reference, long_reference)
    stage1 = targets[STAGE1_TARGET_COLUMN]
    stage2 = targets[STAGE2_TARGET_COLUMN]
    hit_count = int(stage1.eq(1.0).sum())
    no_hit_count = int(stage1.eq(0.0).sum())
    unlabeled_count = int(stage1.isna().sum())
    stage2_buy_count = int(stage2.eq(1.0).sum())
    stage2_sell_count = int(stage2.eq(0.0).sum())

    if hit_count != stage2_buy_count + stage2_sell_count:
        raise ValueError("hierarchical_stage2_support_mismatch")
    if int(targets["label"].eq("HOLD").sum()) != no_hit_count:
        raise ValueError("hierarchical_no_hit_support_mismatch")

    labels = targets["label"]
    return {
        "pair": pair,
        "timeframe": timeframe,
        "bar_count": int(len(candles)),
        "target_rows": int(len(targets)),
        "label_distribution": {
            "BUY": int(labels.eq("BUY").sum()),
            "SELL": int(labels.eq("SELL").sum()),
            "HOLD": int(labels.eq("HOLD").sum()),
            "UNLABELED": unlabeled_count,
        },
        "stage1_distribution": {
            "NO_HIT": no_hit_count,
            "HIT": hit_count,
            "UNLABELED": unlabeled_count,
        },
        "stage2_distribution": {
            "SELL": stage2_sell_count,
            "BUY": stage2_buy_count,
            "UNLABELED": int(stage2.isna().sum()),
        },
        "late_hit_transition_count_13_to_24": late_hit_count,
        "maximum_observed_label_offset": maximum_offset,
        "label_end_index_contract": "FIRST_HIT_OR_DECISION_CANDLE_12",
        "stage1_contract": "HIT_WITHIN_1_TO_12_ELSE_NO_HIT_AT_CANDLE_12",
        "stage2_contract": "DIRECTION_ONLY_WHEN_STAGE1_EQUALS_1",
        "collision_policy": "STOP_LOSS_WINS_AND_IS_MAPPED_TO_SELL",
        "target_contract_status": "PASS",
    }


def run_hierarchical_target_audit(
    input_dir: Path, *, days: int, commit_sha: str
) -> dict[str, Any]:
    """Audit two manifested public datasets across the four allowed entry frames."""
    if isinstance(days, bool) or days not in ALLOWED_DAYS:
        raise ValueError("public_data_days_out_of_policy")
    datasets: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        minute_candles, provenance = load_manifested_dataset(
            input_dir, symbol=symbol, days=days, commit_sha=commit_sha
        )
        datasets.append(provenance)
        for timeframe in TIMEFRAME_MINUTES:
            bars = resample_ohlcv(minute_candles, timeframe)
            experiments.append(
                audit_hierarchical_targets(
                    bars, pair=str(provenance["pair"]), timeframe=timeframe
                )
            )

    return {
        "schema_version": REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "audit_scope": "V4_E_HIERARCHICAL_TARGET_CONTRACT_ONLY",
        "requested_days": days,
        "provider": PROVIDER,
        "input_datasets": datasets,
        "allowed_entry_timeframes": list(TIMEFRAME_MINUTES),
        "target_architecture": "HIERARCHICAL_HIT_THEN_DIRECTION",
        "target_horizon_candles": HIERARCHICAL_HORIZON_CANDLES,
        "experiments": experiments,
        "experiment_count": len(experiments),
        "late_hit_transition_count_13_to_24": sum(
            item["late_hit_transition_count_13_to_24"] for item in experiments
        ),
        "model_training_performed": False,
        "feature_matrix_built": False,
        "economic_evaluation_performed": False,
        "production_target_modified": False,
        "safety": {
            "Champion": None,
            "order_permission": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "trading_authorization": "NO-GO",
        },
    }
