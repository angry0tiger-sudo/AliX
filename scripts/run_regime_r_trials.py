"""Owner-authorized pre-hack-only HGBQ-CQR walk-forward evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from research.regime_r.ngboost_cqr.adapters import (
    OWNER_EXECUTION_ENV,
    HGBQCQRAdapter,
)
from research.regime_r.ngboost_cqr.contracts import TRIAL_IDS

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.csv"
MANIFEST_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.manifest.json"
CONFIG_PATH = ROOT / "config/regime_r/hgbq_cqr_preregistration.json"
REPORT_PATH = ROOT / "docs/research/results/G4_G5_evaluation_results.md"
ADR_PATH = ROOT / "docs/adr/ADR-053-regime-r-prehack-g4-g5-results.md"

EXPECTED_DATA_SHA256 = "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
EXPECTED_CONFIG_SHA256 = "92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa"
EXPECTED_ROWS = 31801
COSTS_BPS = (50, 90)
MAX_PURGE_BARS = 24

FEATURE_COLUMNS = (
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "realized_volatility_6",
    "realized_volatility_24",
    "range_fraction_1",
    "atr_fraction_14",
    "log_volume_change_1",
    "volume_zscore_24",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
FEATURE_CONTRACT = "\n".join(
    (
        "log_return_k=ln(close[t]/close[t-k]); k=1,4,24",
        "realized_volatility_n=sqrt(sum(trailing log_return_1 squared)); n=6,24",
        "range_fraction_1=(high[t]-low[t])/close[t]",
        "true_range=max(high-low,abs(high-prev_close),abs(low-prev_close))",
        "atr_fraction_14=Wilder_RMA(true_range,14)/close",
        "log_volume_change_1=log1p(volume[t])-log1p(volume[t-1])",
        "volume_zscore_24=trailing_zscore(volume,24,ddof=0); constant_window=0; warmup=missing",
        "hour_sin_cos=2pi*(UTC_hour+UTC_minute/60)/24",
        "weekday_sin_cos=2pi*UTC_dayofweek/7; Monday=0",
        "all rolling features are right-aligned and use closed bars through t only",
        "no backfill, centering, future normalization, or future-derived feature",
    )
)
TARGET_CONTRACT = (
    "gross_return(t,h)=close[t+h]/open[t+1]-1; "
    "decision after close[t]; label_end_index=t+h"
)

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_start: int
    train_end: int
    calibration_start: int
    calibration_end: int
    embargo_start: int
    embargo_end: int
    test_start: int
    test_end: int
    train_start_utc: str
    train_end_utc: str
    calibration_start_utc: str
    calibration_end_utc: str
    embargo_start_utc: str
    embargo_end_utc: str
    test_start_utc: str
    test_end_utc: str


@dataclass(frozen=True)
class FoldSamples:
    train_indices: IntArray
    calibration_indices: IntArray
    test_indices: IntArray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_owner_authorization() -> None:
    if os.environ.get(OWNER_EXECUTION_ENV) != "true":
        raise PermissionError("regime_r_evaluation_requires_exact_owner_authorization")
    if os.environ.get("ALIX_DRY_RUN") != "true":
        raise PermissionError("regime_r_evaluation_requires_dry_run_true")
    if os.environ.get("ALIX_ALLOW_LIVE_TRADING") != "false":
        raise PermissionError("regime_r_evaluation_requires_live_trading_false")


def load_and_verify_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    expected_csv = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.csv"
    if CSV_PATH != expected_csv or "holdout" in str(CSV_PATH).lower():
        raise RuntimeError("regime_r_dataset_path_contract_invalid")

    actual_data_sha = sha256_bytes(CSV_PATH.read_bytes())
    if actual_data_sha != EXPECTED_DATA_SHA256:
        raise RuntimeError("regime_r_prehack_csv_sha256_mismatch")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("regime_r_prehack_manifest_invalid")
    manifest_contract = (
        manifest.get("csv_sha256") == EXPECTED_DATA_SHA256
        and manifest.get("dataset_window") == "PRE_HACK_NOBITEX_ONLY"
        and manifest.get("source_policy") == "NOBITEX_ONLY_NO_BACKFILL"
        and manifest.get("row_count") == EXPECTED_ROWS
        and manifest.get("expected_candles") == EXPECTED_ROWS
        and manifest.get("missing_intervals") == 0
        and manifest.get("public_api_only") is True
        and manifest.get("credentials_used") is False
        and manifest.get("private_api_calls") == 0
        and manifest.get("real_order_attempts") == 0
        and manifest.get("trading_authorization") == "NO-GO"
    )
    if not manifest_contract:
        raise RuntimeError("regime_r_prehack_manifest_contract_invalid")

    actual_config_sha = sha256_bytes(CONFIG_PATH.read_bytes())
    if actual_config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("hgbq_preregistration_config_sha256_mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("hgbq_preregistration_config_invalid")

    frame = pd.read_csv(CSV_PATH)
    expected_columns = ["date", "open", "high", "low", "close", "volume"]
    if list(frame.columns) != expected_columns or len(frame) != EXPECTED_ROWS:
        raise RuntimeError("regime_r_prehack_csv_schema_or_row_count_invalid")

    timestamps = pd.to_datetime(frame["date"], utc=True, errors="raise")
    timestamp_ns = timestamps.astype("int64").to_numpy(dtype=np.int64)
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise RuntimeError("regime_r_prehack_timestamps_invalid")
    if not np.all(np.diff(timestamp_ns) == 3_600_000_000_000):
        raise RuntimeError("regime_r_prehack_hourly_grid_invalid")

    ohlcv = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    if not np.isfinite(ohlcv).all():
        raise RuntimeError("regime_r_prehack_ohlcv_non_finite")
    open_values = ohlcv[:, 0]
    high_values = ohlcv[:, 1]
    low_values = ohlcv[:, 2]
    close_values = ohlcv[:, 3]
    volume_values = ohlcv[:, 4]
    if (
        np.any(open_values <= 0)
        or np.any(high_values <= 0)
        or np.any(low_values <= 0)
        or np.any(close_values <= 0)
        or np.any(volume_values < 0)
        or np.any(
            high_values < np.maximum.reduce((open_values, low_values, close_values))
        )
        or np.any(
            low_values > np.minimum.reduce((open_values, high_values, close_values))
        )
    ):
        raise RuntimeError("regime_r_prehack_ohlc_invariants_invalid")

    frame = frame.drop(columns=["date"]).copy()
    frame["timestamp"] = timestamps
    return frame, manifest, config


def _wilder_atr_fraction(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1, skipna=True)

    atr = pd.Series(np.nan, index=frame.index, dtype="float64")
    if len(frame) < 14:
        return atr
    atr.iloc[13] = float(true_range.iloc[:14].mean())
    for index in range(14, len(frame)):
        atr.iloc[index] = (atr.iloc[index - 1] * 13.0 + true_range.iloc[index]) / 14.0
    return atr / close


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    volume = frame["volume"].astype("float64")
    log_close = np.log(close)
    log_return_1 = log_close.diff(1)

    volume_mean = volume.rolling(window=24, min_periods=24).mean()
    volume_std = volume.rolling(window=24, min_periods=24).std(ddof=0)
    volume_zscore = (volume - volume_mean) / volume_std
    volume_zscore = volume_zscore.mask(volume_std.eq(0.0), 0.0)

    timestamp = pd.DatetimeIndex(frame["timestamp"])
    time_of_day = timestamp.hour.to_numpy(dtype=np.float64) + (
        timestamp.minute.to_numpy(dtype=np.float64) / 60.0
    )
    weekday = timestamp.dayofweek.to_numpy(dtype=np.float64)

    values = {
        "log_return_1": log_return_1,
        "log_return_4": log_close.diff(4),
        "log_return_24": log_close.diff(24),
        "realized_volatility_6": np.sqrt(
            log_return_1.pow(2).rolling(window=6, min_periods=6).sum()
        ),
        "realized_volatility_24": np.sqrt(
            log_return_1.pow(2).rolling(window=24, min_periods=24).sum()
        ),
        "range_fraction_1": (high - low) / close,
        "atr_fraction_14": _wilder_atr_fraction(frame),
        "log_volume_change_1": np.log1p(volume).diff(1),
        "volume_zscore_24": volume_zscore,
        "hour_sin": np.sin(2.0 * np.pi * time_of_day / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * time_of_day / 24.0),
        "weekday_sin": np.sin(2.0 * np.pi * weekday / 7.0),
        "weekday_cos": np.cos(2.0 * np.pi * weekday / 7.0),
    }
    result = pd.DataFrame(values, index=frame.index).loc[:, list(FEATURE_COLUMNS)]
    return result.replace([np.inf, -np.inf], np.nan)


def build_target(frame: pd.DataFrame, horizon_bars: int) -> FloatArray:
    if horizon_bars not in (1, 4, 24):
        raise ValueError("regime_r_target_horizon_not_preregistered")
    target = frame["close"].shift(-horizon_bars) / frame["open"].shift(-1) - 1.0
    return target.to_numpy(dtype=np.float64)


def _month_timestamp(anchor: pd.Timestamp, offset: int) -> pd.Timestamp:
    return anchor + pd.DateOffset(months=offset)


def build_folds(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Fold, ...]:
    protocol = config["walk_forward_protocol"]
    expected = (
        protocol.get("fold_count") == 3
        and protocol.get("training_window_months") == 6
        and protocol.get("calibration_window_months") == 1
        and protocol.get("embargo_window_months") == 1
        and protocol.get("test_window_months") == 1
        and protocol.get("maximum_target_horizon_hours") == MAX_PURGE_BARS
        and protocol.get("purge_policy", {}).get("enabled") is True
    )
    if not expected:
        raise RuntimeError("regime_r_walk_forward_contract_incompatible")

    anchor = pd.Timestamp(manifest["start_utc"])
    if anchor.tzinfo is None:
        raise RuntimeError("regime_r_t0_must_be_timezone_aware")
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    timestamp_ns = timestamps.asi8
    folds: list[Fold] = []

    for item in protocol["folds"]:
        offsets = (
            item["training_month_offsets"],
            item["calibration_month_offsets"],
            item["embargo_month_offsets"],
            item["test_month_offsets"],
        )
        boundaries = (
            _month_timestamp(anchor, int(offsets[0][0])),
            _month_timestamp(anchor, int(offsets[0][1])),
            _month_timestamp(anchor, int(offsets[1][0])),
            _month_timestamp(anchor, int(offsets[1][1])),
            _month_timestamp(anchor, int(offsets[2][0])),
            _month_timestamp(anchor, int(offsets[2][1])),
            _month_timestamp(anchor, int(offsets[3][0])),
            _month_timestamp(anchor, int(offsets[3][1])),
        )
        indices = tuple(
            int(np.searchsorted(timestamp_ns, boundary.value, side="left"))
            for boundary in boundaries
        )
        if not all(0 <= index < len(frame) for index in indices[:-1]):
            raise RuntimeError("regime_r_fold_boundary_outside_dataset")
        if indices[-1] > len(frame):
            raise RuntimeError("regime_r_fold_test_end_outside_dataset")
        interval_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
        for start_index, end_index in interval_pairs:
            if indices[start_index] >= indices[end_index]:
                raise RuntimeError("regime_r_fold_interval_empty")
        if not (
            indices[1] == indices[2]
            and indices[3] == indices[4]
            and indices[5] == indices[6]
        ):
            raise RuntimeError("regime_r_fold_intervals_not_contiguous")

        folds.append(
            Fold(
                fold_id=str(item["fold_id"]),
                train_start=indices[0],
                train_end=indices[1],
                calibration_start=indices[2],
                calibration_end=indices[3],
                embargo_start=indices[4],
                embargo_end=indices[5],
                test_start=indices[6],
                test_end=indices[7],
                train_start_utc=boundaries[0].isoformat(),
                train_end_utc=boundaries[1].isoformat(),
                calibration_start_utc=boundaries[2].isoformat(),
                calibration_end_utc=boundaries[3].isoformat(),
                embargo_start_utc=boundaries[4].isoformat(),
                embargo_end_utc=boundaries[5].isoformat(),
                test_start_utc=boundaries[6].isoformat(),
                test_end_utc=boundaries[7].isoformat(),
            )
        )
    return tuple(folds)


def build_fold_samples(
    features: pd.DataFrame,
    target: FloatArray,
    horizon_bars: int,
    fold: Fold,
) -> FoldSamples:
    feature_values = features.to_numpy(dtype=np.float64)
    if feature_values.shape[0] != target.shape[0]:
        raise RuntimeError("regime_r_feature_target_length_mismatch")
    row_indices = np.arange(len(target), dtype=np.int64)
    finite = np.isfinite(feature_values).all(axis=1) & np.isfinite(target)
    mature = row_indices + horizon_bars < len(target)

    train_mask = (
        (row_indices >= fold.train_start)
        & (row_indices < fold.train_end)
        & (row_indices + MAX_PURGE_BARS < fold.calibration_start)
        & finite
        & mature
    )
    calibration_mask = (
        (row_indices >= fold.calibration_start)
        & (row_indices < fold.calibration_end)
        & (row_indices + MAX_PURGE_BARS < fold.embargo_start)
        & finite
        & mature
    )
    test_mask = (
        (row_indices >= fold.test_start)
        & (row_indices < fold.test_end)
        & (row_indices + horizon_bars < fold.test_end)
        & finite
        & mature
    )

    train_indices = row_indices[train_mask]
    calibration_indices = row_indices[calibration_mask]
    test_indices = row_indices[test_mask]
    if (
        train_indices.size < 100
        or calibration_indices.size < 10
        or test_indices.size < 1
    ):
        raise RuntimeError(
            "regime_r_fold_sample_support_insufficient:"
            f"{fold.fold_id}:train={train_indices.size}:"
            f"calibration={calibration_indices.size}:test={test_indices.size}"
        )
    if train_indices[-1] + MAX_PURGE_BARS >= fold.calibration_start:
        raise RuntimeError("regime_r_train_purge_contract_failed")
    if calibration_indices[-1] + MAX_PURGE_BARS >= fold.embargo_start:
        raise RuntimeError("regime_r_calibration_purge_contract_failed")
    return FoldSamples(train_indices, calibration_indices, test_indices)


def maximum_drawdown(equity_curve: list[float]) -> float | None:
    values = np.asarray(equity_curve, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
        return None
    peaks = np.maximum.accumulate(values)
    return float(np.max((peaks - values) / peaks))


def profit_factor(trades: list[dict[str, Any]]) -> float | None:
    returns = np.asarray(
        [float(trade["net_return"]) for trade in trades],
        dtype=np.float64,
    )
    if returns.size == 0 or not np.isfinite(returns).all():
        return None
    gross_profit = float(returns[returns > 0.0].sum())
    gross_loss = float(np.abs(returns[returns < 0.0].sum()))
    if gross_loss <= 0.0:
        return None
    return gross_profit / gross_loss


def simulate_costed_ledger(
    frame: pd.DataFrame,
    test_start: int,
    test_end: int,
    horizon_bars: int,
    test_indices: IntArray,
    lower_bounds: FloatArray,
    cost_bps: int,
) -> dict[str, Any]:
    if cost_bps not in COSTS_BPS:
        raise ValueError("regime_r_cost_scenario_not_preregistered")
    if test_indices.shape != lower_bounds.shape:
        raise ValueError("regime_r_test_forecast_shape_invalid")

    lower_by_index = {
        int(index): float(value)
        for index, value in zip(test_indices, lower_bounds, strict=True)
    }
    opens = frame["open"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    cost_fraction = cost_bps / 10_000.0

    equity = 1.0
    curve = [equity]
    position: dict[str, float | int] | None = None
    trades: list[dict[str, Any]] = []
    above = 0
    abstain = 0
    skipped = 0
    unexecutable = 0

    for index in range(test_start, test_end):
        if position is not None:
            entry_index = int(position["entry_index"])
            exit_index = int(position["exit_index"])
            entry_price = float(position["entry_price"])
            base_equity = float(position["base_equity"])
            if index >= entry_index:
                if index < exit_index:
                    equity = base_equity * closes[index] / entry_price
                elif index == exit_index:
                    gross = closes[index] / entry_price - 1.0
                    net = gross - cost_fraction
                    equity = base_equity * (1.0 + net)
                    trades.append(
                        {
                            "entry_index": entry_index,
                            "exit_index": exit_index,
                            "entry_time": timestamps[entry_index].isoformat(),
                            "exit_time": timestamps[exit_index].isoformat(),
                            "entry_price": entry_price,
                            "exit_price": float(closes[index]),
                            "gross_return": float(gross),
                            "round_trip_cost_bps": cost_bps,
                            "net_return": float(net),
                            "equity_after": float(equity),
                        }
                    )
                    position = None
                else:
                    raise RuntimeError("regime_r_position_exit_index_skipped")

        lower = lower_by_index.get(index)
        if lower is not None:
            if lower > cost_fraction:
                above += 1
                if position is not None:
                    skipped += 1
                else:
                    entry_index = index + 1
                    exit_index = index + horizon_bars
                    if entry_index >= test_end or exit_index >= test_end:
                        unexecutable += 1
                    else:
                        entry_price = float(opens[entry_index])
                        if not math.isfinite(entry_price) or entry_price <= 0.0:
                            raise RuntimeError(
                                "regime_r_invalid_hypothetical_entry_price"
                            )
                        position = {
                            "entry_index": entry_index,
                            "exit_index": exit_index,
                            "entry_price": entry_price,
                            "base_equity": equity,
                        }
            else:
                abstain += 1
        curve.append(float(equity))

    if position is not None:
        raise RuntimeError("regime_r_unclosed_test_position")
    return {
        "trades": trades,
        "equity_curve": curve,
        "final_equity": float(equity),
        "net_profit_factor": profit_factor(trades),
        "max_drawdown": maximum_drawdown(curve),
        "trade_count": len(trades),
        "forecast_count": len(test_indices),
        "signal_count_above_lower_bound": above,
        "abstention_count": abstain,
        "skipped_signal_position_open": skipped,
        "unexecutable_signal_count": unexecutable,
        "net_return": float(equity - 1.0),
    }


def aggregate_fold_ledgers(ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    all_trades: list[dict[str, Any]] = []
    pooled_curve = [1.0]
    equity_base = 1.0
    for ledger in ledgers:
        all_trades.extend(ledger["trades"])
        pooled_curve.extend(
            equity_base * float(value) for value in ledger["equity_curve"][1:]
        )
        equity_base *= float(ledger["final_equity"])
    return {
        "trades": all_trades,
        "equity_curve": pooled_curve,
        "final_equity": equity_base,
        "net_profit_factor": profit_factor(all_trades),
        "max_drawdown": maximum_drawdown(pooled_curve),
        "trade_count": len(all_trades),
        "net_return": equity_base - 1.0,
    }


def fold_stability_count(fold_metrics: list[dict[str, Any]]) -> int:
    return sum(
        1
        for metrics in fold_metrics
        if metrics["net_profit_factor"] is not None
        and float(metrics["net_profit_factor"]) >= 1.10
        and metrics["max_drawdown"] is not None
        and float(metrics["max_drawdown"]) <= 0.15
        and int(metrics["trade_count"]) > 0
    )


def scenario_acceptance(
    metrics: dict[str, Any],
    stability: int,
) -> dict[str, Any]:
    gates = {
        "net_pf_gte_1_10": metrics["net_profit_factor"] is not None
        and float(metrics["net_profit_factor"]) >= 1.10,
        "max_dd_lte_15_percent": metrics["max_drawdown"] is not None
        and float(metrics["max_drawdown"]) <= 0.15,
        "trades_gte_100": int(metrics["trade_count"]) >= 100,
        "fold_stability_gte_3": stability >= 3,
        "dsr_gt_zero": False,
    }
    return {
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "dsr": None,
        "dsr_status": "NOT_COMPUTABLE_GLOBAL_N_UNKNOWN",
    }


def trial_identity(trial_id: str) -> tuple[str, int]:
    parts = trial_id.split("-")
    if len(parts) != 5:
        raise RuntimeError("regime_r_trial_id_format_invalid")
    horizon = parts[-2]
    candidate = int(parts[-1])
    if horizon not in ("1h", "4h", "24h") or not 1 <= candidate <= 10:
        raise RuntimeError("regime_r_trial_id_not_in_preregistered_matrix")
    return horizon, candidate


def run_trial(
    trial_id: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    folds: tuple[Fold, ...],
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    horizon, slot = trial_identity(trial_id)
    horizon_bars = int(horizon[:-1])
    target = build_target(frame, horizon_bars)
    values = features.to_numpy(dtype=np.float64)

    fold_results: dict[str, list[dict[str, Any]]] = {"50": [], "90": []}
    fold_ledgers: dict[str, list[dict[str, Any]]] = {"50": [], "90": []}
    trade_rows: list[dict[str, Any]] = []

    for fold in folds:
        samples = build_fold_samples(features, target, horizon_bars, fold)
        adapter = HGBQCQRAdapter(horizon=horizon)
        adapter.fit(
            values[samples.train_indices],
            target[samples.train_indices],
            values[samples.calibration_indices],
            target[samples.calibration_indices],
        )
        _, intervals = adapter.predict_batch_intervals(values[samples.test_indices])
        lower_bounds = np.asarray(
            [interval.lower_bound for interval in intervals],
            dtype=np.float64,
        )

        for cost in COSTS_BPS:
            ledger = simulate_costed_ledger(
                frame,
                fold.test_start,
                fold.test_end,
                horizon_bars,
                samples.test_indices,
                lower_bounds,
                cost,
            )
            fold_ledgers[str(cost)].append(ledger)
            fold_results[str(cost)].append(
                {
                    "fold_id": fold.fold_id,
                    "train_start_utc": fold.train_start_utc,
                    "train_end_utc": fold.train_end_utc,
                    "calibration_start_utc": fold.calibration_start_utc,
                    "calibration_end_utc": fold.calibration_end_utc,
                    "embargo_start_utc": fold.embargo_start_utc,
                    "embargo_end_utc": fold.embargo_end_utc,
                    "test_start_utc": fold.test_start_utc,
                    "test_end_utc": fold.test_end_utc,
                    "training_rows": int(samples.train_indices.size),
                    "calibration_rows": int(samples.calibration_indices.size),
                    "test_rows": int(samples.test_indices.size),
                    "metrics": {
                        key: value
                        for key, value in ledger.items()
                        if key not in ("trades", "equity_curve")
                    },
                }
            )
            for trade in ledger["trades"]:
                trade_rows.append(
                    {
                        "trial_id": trial_id,
                        "horizon": horizon,
                        "candidate_slot": slot,
                        "fold_id": fold.fold_id,
                        "cost_bps": cost,
                        **trade,
                    }
                )

    scenarios: dict[str, Any] = {}
    for cost in COSTS_BPS:
        cost_key = str(cost)
        fold_metrics = [item["metrics"] for item in fold_results[cost_key]]
        pooled = aggregate_fold_ledgers(fold_ledgers[cost_key])
        stability = fold_stability_count(fold_metrics)
        pooled_metrics = {
            key: value
            for key, value in pooled.items()
            if key not in ("trades", "equity_curve")
        }
        scenarios[cost_key] = {
            "pooled_metrics": pooled_metrics,
            "fold_stability_passed": stability,
            "folds": fold_results[cost_key],
            "acceptance": scenario_acceptance(pooled_metrics, stability),
        }

    duplicate_of = f"NOBITEX-R1-NGBOOST-{horizon}-01" if slot > 1 else None
    result = {
        "trial_id": trial_id,
        "horizon": horizon,
        "candidate_slot": slot,
        "duplicate_of": duplicate_of,
        "unique_candidate_configuration": slot == 1,
        "duplicate_reason": None
        if slot == 1
        else "The G3 JSON has no candidate-specific parameter, feature, or seed difference for this slot; this slot repeats slot 01.",
        "run_id": run_id,
        "fit_calls": len(folds),
        "cost_scenarios": scenarios,
        "status": "COMPLETED",
    }
    return result, trade_rows


def fmt(value: Any, percentage: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.2%}" if percentage else f"{float(value):.4f}"


def write_report(report: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    lines = [
        "# AliX Regime R — G4/G5 HGBQ-CQR Pre-hack Evaluation",
        "",
        "Status: COMPLETED — EXPLORATORY PRE-HACK ONLY; NOT CONFIRMATORY",
        f"Code commit: {report["code_commit_sha"]}",
        f"Workflow run: {report["workflow_run_id"]}",
        f"Dataset SHA-256: {report["dataset"]["csv_sha256"]}",
        f"G3 configuration SHA-256: {report["configuration_sha256"]}",
        f"Feature contract SHA-256: {report["feature_contract_sha256"]}",
        f"Target contract SHA-256: {report["target_contract_sha256"]}",
        "",
        "## Scope and interpretation",
        "",
        "The Owner-authorized input is the fixed, historically exposed public pre-hack Nobitex BTC/USDT 1h dataset. ADR-048 explicitly classifies this dataset as exploratory, not a clean confirmatory holdout. All trades below are hypothetical OHLCV reference fills. No post-hack Holdout bytes were read or copied.",
        "",
        "The 30 IDs are retained for registry traceability. Within each horizon, slots 02-10 have no candidate-specific settings in the immutable G3 JSON and are exact duplicates of slot 01. They are not independent model configurations.",
        "",
        "GLOBAL_N remains UNKNOWN. A numeric DSR is NOT_COMPUTABLE and its positive gate is failed closed. No DSR value or guessed multiplicity count is fabricated.",
        "",
        "## Frozen protocol",
        "",
        f"- Dataset: {report["dataset"]["row_count"]} rows; {report["dataset"]["start_utc"]} through {report["dataset"]["end_utc"]}.",
        "- T0 is the manifest start timestamp; no outcome-dependent anchor was selected.",
        "- Each fold uses six months training, one month calibration, one month embargo and one month test.",
        "- Full 24-bar purge applies before the calibration and embargo boundaries.",
        f"- Volume quality: {report['dataset']['volume_zero_rows']} zero-volume rows and {report['dataset']['volume_positive_rows']} positive-volume rows; zero-volume windows use neutral volume Z-score 0 after the 24-bar warm-up.",
        "- Target: gross_return(t,h) = close[t+h] / open[t+1] - 1.",
        "- Features are causal and right-aligned. Their exact operational definitions and hash are listed above.",
        "- One hypothetical open position at a time per horizon/cost ledger; overlapping signals are counted as skipped.",
        "- Entry reference is next candle open; exit reference is horizon candle close.",
        "- Round-trip cost is deducted once per completed hypothetical trade.",
        "",
        "## Per-trial economic results",
        "",
        "| Trial ID | Horizon | Duplicate of | Cost bps | NetPF | MaxDD | Trades | Fold stability | DSR | All gates |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for trial in trials:
        for cost in COSTS_BPS:
            scenario = trial["cost_scenarios"][str(cost)]
            metrics = scenario["pooled_metrics"]
            passed = scenario["acceptance"]["all_gates_passed"]
            lines.append(
                f"| {trial['trial_id']} | {trial['horizon']} | "
                f"{trial['duplicate_of'] or '—'} | {cost} | "
                f"{fmt(metrics['net_profit_factor'])} | "
                f"{fmt(metrics['max_drawdown'], True)} | "
                f"{metrics['trade_count']} | "
                f"{scenario['fold_stability_passed']}/3 | "
                "NOT_COMPUTABLE | "
                f"{'PASS' if passed else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Fold-level economic results",
            "",
            "| Trial ID | Cost bps | Fold | NetPF | MaxDD | Trades | Signals above cost | Abstentions | Skipped while occupied |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trial in trials:
        for cost in COSTS_BPS:
            for fold in trial["cost_scenarios"][str(cost)]["folds"]:
                metrics = fold["metrics"]
                lines.append(
                    f"| {trial['trial_id']} | {cost} | {fold['fold_id']} | "
                    f"{fmt(metrics['net_profit_factor'])} | "
                    f"{fmt(metrics['max_drawdown'], True)} | "
                    f"{metrics['trade_count']} | "
                    f"{metrics['signal_count_above_lower_bound']} | "
                    f"{metrics['abstention_count']} | "
                    f"{metrics['skipped_signal_position_open']} |"
                )
    lines.extend(
        [
            "",
            "## Trial accounting and verdict",
            "",
            f"- Registered IDs: {report['trial_counts']['registered_slots']}.",
            f"- Unique model configurations: {report['trial_counts']['unique_candidate_configurations']}.",
            f"- Exact duplicate slots: {report['trial_counts']['duplicate_slots']}.",
            f"- Fit calls: {report['trial_counts']['fit_calls']}.",
            "- GLOBAL_N: UNKNOWN.",
            "- DSR: NOT_COMPUTABLE_GLOBAL_N_UNKNOWN.",
            "- DSR > 0: FAIL_CLOSED for every trial.",
            f"- Candidates passing every gate at both costs: {report['acceptance']['full_gate_pass_count']}.",
            f"- Verdict: **{report["verdict"]}**.",
            f"- Verdict reason: {report['verdict_reason']}.",
            "",
            "## Safety invariants",
            "",
            "- real_data_scope: PRE_HACK_ONLY",
            "- holdout_bytes_read: false",
            "- private_api_calls: 0",
            "- real_order_attempts: 0",
            "- paper_orders: 0",
            "- Champion: null",
            "- order_permission: false",
            "- trading_authorization: NO-GO",
            "",
            "This exploratory result is not independent confirmation and does not authorize Holdout access, promotion or trading.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adr(report: dict[str, Any]) -> None:
    content = f"""# ADR-053 — Regime R HGBQ-CQR Pre-hack G4/G5 Evaluation

- Status: {report["verdict"]} — exploratory pre-hack evidence only.
- Date: 2026-11-17
- Branch: research/nobitex-hf-strategy
- Code commit: {report["code_commit_sha"]}
- Workflow run: {report["workflow_run_id"]}
- Dataset SHA-256: {report["dataset"]["csv_sha256"]}
- G3 configuration SHA-256: {report["configuration_sha256"]}
- Feature contract SHA-256: {report["feature_contract_sha256"]}
- Target contract SHA-256: {report["target_contract_sha256"]}

## 1. Authorization and data boundary

The Owner explicitly authorized G4/G5 fitting and exploratory evaluation on the fixed pre-hack dataset only. The exact file is data/nobitex/prehack/nobitex-btcusdt-1h.csv. Its byte-level SHA-256 was verified before parsing. It contains {report["dataset"]["row_count"]} rows from {report["dataset"]["start_utc"]} through {report["dataset"]["end_utc"]}. All {report["dataset"]["volume_zero_rows"]} volume entries are zero; the volume-derived Z-score is assigned neutral zero for constant windows, and those features carry no discriminatory information.

ADR-048 states that this data is historically exposed and is not a clean confirmatory holdout. This run is therefore exploratory. No post-hack Holdout path or byte was opened.

## 2. Frozen model and temporal protocol

The G3 JSON was verified byte-for-byte and was not modified. HGB hyperparameters, quantiles, MAPIE 1.6.0, 90% confidence, horizons 1/4/24 hours, costs 50/90 bps and the three fold offsets remain as preregistered.

T0 is the immutable manifest start timestamp. Each fold follows six months training, one month separate calibration, one month embargo and one month test. A full 24-hour purge applies at the training/calibration and calibration/embargo boundaries.

Feature formulas were operationalized from the names/descriptions in ADR-048 and are fingerprinted by the feature contract SHA above. Target timing follows ADR-048: decision after close[t], entry open[t+1], exit close[t+h], gross return close[t+h]/open[t+1]-1. The hypothetical ledger permits one open position at a time per horizon/cost scenario; overlapping signals are recorded as skipped. Round-trip cost is deducted once per completed trade.

## 3. Trial identity and multiplicity

- Registered slots: {report['trial_counts']['registered_slots']}.
- Unique configurations: {report['trial_counts']['unique_candidate_configurations']}.
- Exact duplicate slots: {report['trial_counts']['duplicate_slots']}.

The ten IDs within each horizon have identical frozen model settings, seed, features, fold plan and cost policy. Slots 02-10 are therefore duplicate evaluations of slot 01, not ten independent configurations. The registry identities are retained, but no extra statistical independence is claimed.

## 4. Economic results

Per-trial, per-cost and per-fold NetPF, MaxDD, completed trades, signal attrition and fold stability are recorded in docs/research/results/G4_G5_evaluation_results.md and the run artifact.

Candidates passing all gates at both costs: {report['acceptance']['full_gate_pass_count']}.

## 5. DSR and verdict

GLOBAL_N remains UNKNOWN. The historical multiplicity cohort and dependence structure are not reconciled. Under ADR-048, a numeric DSR cannot be responsibly reported without comparable candidate return series, required moments, candidate identity/multiplicity reconciliation and documented dependence treatment. DSR is therefore NOT_COMPUTABLE_GLOBAL_N_UNKNOWN and its required positive gate fails closed. No guessed N or numeric DSR is substituted.

The strict verdict is **{report["verdict"]}**. This is an evidence NO-GO for the preregistered run because no scenario passed all gates, and DSR is not computable. It is not proof that every possible HGBQ-CQR design lacks economic edge.

## 6. Safety invariants

- real_data_scope: PRE_HACK_ONLY
- holdout_bytes_read: false
- private_api_calls: 0
- real_order_attempts: 0
- paper_orders: 0
- Champion: null
- order_permission: false
- trading_authorization: NO-GO
- main remains frozen at 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb.

**Decision: {report["verdict"]}.** Do not unlock Holdout or promote a candidate based on this exploratory pre-hack evaluation. Confirmatory evaluation needs a separate approved data window and reconciled multiplicity/DSR evidence.

Architecture First. Evidence Always. Fail Closed.
"""
    ADR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADR_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".artifacts/regime-r-g4-g5",
    )
    args = parser.parse_args()
    require_owner_authorization()
    if len(args.commit_sha) != 40:
        raise ValueError("regime_r_commit_sha_must_be_full_git_sha")
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha is not None and github_sha != args.commit_sha:
        raise RuntimeError("regime_r_workflow_commit_sha_mismatch")

    frame, manifest, config = load_and_verify_inputs()
    features = build_features(frame)
    folds = build_folds(frame, manifest, config)
    trial_ids = tuple(config["trial_family"]["trial_ids"])
    if len(trial_ids) != 30 or set(trial_ids) != set(TRIAL_IDS):
        raise RuntimeError("regime_r_trial_registry_mismatch")

    run_id = os.environ.get("GITHUB_RUN_ID", "local-run")
    trials: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    for trial_id in trial_ids:
        result, trade_rows = run_trial(
            trial_id=trial_id,
            frame=frame,
            features=features,
            folds=folds,
            run_id=run_id,
        )
        trials.append(result)
        all_trades.extend(trade_rows)

    full_pass_count = 0
    for trial in trials:
        both_costs_pass = all(
            trial["cost_scenarios"][str(cost)]["acceptance"]["all_gates_passed"]
            for cost in COSTS_BPS
        )
        if both_costs_pass:
            full_pass_count += 1

    unique_configs = 3
    duplicates = len(trials) - unique_configs
    verdict = "PASS" if full_pass_count > 0 else "NO-GO"
    frame_start = frame["timestamp"].iloc[0].isoformat()
    frame_end = frame["timestamp"].iloc[-1].isoformat()
    report: dict[str, Any] = {
        "schema_version": "alix.regime-r.g4-g5.v1",
        "status": "COMPLETED",
        "code_commit_sha": args.commit_sha,
        "workflow_run_id": run_id,
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "dataset": {
            "path": "data/nobitex/prehack/nobitex-btcusdt-1h.csv",
            "csv_sha256": EXPECTED_DATA_SHA256,
            "row_count": len(frame),
            "start_utc": frame_start,
            "end_utc": frame_end,
            "scope": "PRE_HACK_NOBITEX_ONLY_EXPLORATORY",
            "historically_exposed": True,
            "confirmatory_holdout": False,
        },
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "feature_contract_sha256": sha256_bytes(FEATURE_CONTRACT.encode("utf-8")),
        "feature_contract": FEATURE_CONTRACT.split("\n"),
        "target_contract_sha256": sha256_bytes(TARGET_CONTRACT.encode("utf-8")),
        "target_contract": TARGET_CONTRACT,
        "folds": [asdict(fold) for fold in folds],
        "trial_counts": {
            "registered_slots": len(trials),
            "unique_candidate_configurations": unique_configs,
            "duplicate_slots": duplicates,
            "fit_calls": len(trials) * len(folds),
        },
        "global_n": "UNKNOWN",
        "global_dsr": None,
        "dsr_status": "NOT_COMPUTABLE_GLOBAL_N_UNKNOWN",
        "trials": trials,
        "acceptance": {
            "full_gate_pass_count": full_pass_count,
            "required_net_pf": 1.10,
            "required_max_drawdown": 0.15,
            "required_trade_count": 100,
            "required_fold_stability": 3,
            "required_dsr_gt": 0.0,
            "dsr_gate_passable": False,
        },
        "verdict": verdict,
        "verdict_reason": (
            "GLOBAL_N_UNKNOWN prevents a defensible positive DSR; DSR gate fails closed."
            if full_pass_count == 0
            else "At least one trial cleared every gate at both costs."
        ),
        "safety": {
            "real_data_scope": "PRE_HACK_ONLY",
            "holdout_bytes_read": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "regime-r-g4-g5-results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(all_trades).to_csv(
        args.output_dir / "regime-r-g4-g5-trades.csv",
        index=False,
    )
    write_report(report, trials)
    write_adr(report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "registered_slots": len(trials),
                "unique_candidate_configurations": unique_configs,
                "full_gate_pass_count": full_pass_count,
                "global_dsr": report["global_dsr"],
                "dsr_status": report["dsr_status"],
                "report": str(REPORT_PATH),
                "adr": str(ADR_PATH),
                "trades_artifact": str(args.output_dir / "regime-r-g4-g5-trades.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
