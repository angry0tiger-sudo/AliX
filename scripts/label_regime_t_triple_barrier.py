"""Create the locked, pre-hack-only Regime T triple-barrier event labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.csv"
MANIFEST_PATH = ROOT / "data/nobitex/prehack/nobitex-btcusdt-1h.manifest.json"
G3_PATH = ROOT / "config/regime_r/hgbq_cqr_preregistration.json"
PREREG_PATH = ROOT / "config/regime_t/triple_barrier_preregistration.json"
EXPECTED_DATA_SHA256 = (
    "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
)
EXPECTED_MANIFEST_SHA256 = (
    "b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26"
)
EXPECTED_G3_SHA256 = "92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa"
EXPECTED_PREREG_SHA256 = (
    "b6118f4e7420b26c8caf471ef7281b7b452c031ed7a68b6d8eb5a7d210d103ad"
)
HORIZON_BARS = 24
K = 2.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source_commitments() -> None:
    commitments = (
        (DATA_PATH, EXPECTED_DATA_SHA256),
        (MANIFEST_PATH, EXPECTED_MANIFEST_SHA256),
        (G3_PATH, EXPECTED_G3_SHA256),
        (PREREG_PATH, EXPECTED_PREREG_SHA256),
    )
    if any(sha256_file(path) != expected for path, expected in commitments):
        raise RuntimeError("regime_t_source_sha256_mismatch")


def _load_and_validate_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required_values = (
        manifest.get("dataset_window") == "PRE_HACK_NOBITEX_ONLY",
        manifest.get("csv_sha256") == EXPECTED_DATA_SHA256,
        manifest.get("row_count") == 31801,
        manifest.get("missing_intervals") == 0,
        manifest.get("public_api_only") is True,
        manifest.get("private_api_calls") == 0,
        manifest.get("real_order_attempts") == 0,
    )
    if not all(required_values):
        raise RuntimeError("regime_t_manifest_contract_invalid")
    return manifest


def _validate_ohlc_frame(frame: pd.DataFrame) -> pd.Series:
    expected_columns = ["date", "open", "high", "low", "close", "volume"]
    if list(frame.columns) != expected_columns or len(frame) != 31801:
        raise RuntimeError("regime_t_dataset_schema_or_row_count_invalid")
    timestamps = pd.to_datetime(frame["date"], utc=True, errors="raise")
    timestamp_ns = timestamps.astype("int64").to_numpy(dtype=np.int64)
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise RuntimeError("regime_t_timestamps_invalid")
    if not np.all(np.diff(timestamp_ns) == 3_600_000_000_000):
        raise RuntimeError("regime_t_hourly_grid_invalid")
    ohlc = frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    if not np.isfinite(ohlc).all() or np.any(ohlc <= 0.0):
        raise RuntimeError("regime_t_ohlc_non_finite_or_non_positive")
    opened, high, low, close = ohlc.T
    invalid_high = high < np.maximum.reduce((opened, low, close))
    invalid_low = low > np.minimum.reduce((opened, high, close))
    if invalid_high.any() or invalid_low.any():
        raise RuntimeError("regime_t_ohlc_range_invalid")
    return timestamps


def load_verified_prehack_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the immutable pre-hack CSV and verify all source commitments."""
    _verify_source_commitments()
    manifest = _load_and_validate_manifest()
    frame = pd.read_csv(DATA_PATH)
    timestamps = _validate_ohlc_frame(frame)
    frame = frame.drop(columns=["date"]).copy()
    frame["timestamp"] = timestamps
    return frame, manifest


def wilder_atr_fraction(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute right-aligned Wilder ATR fraction using price data through t."""
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    previous_close = close.shift(1)
    true_range = pd.concat(
        (high - low, (high - previous_close).abs(), (low - previous_close).abs()),
        axis=1,
    ).max(axis=1, skipna=True)
    atr = pd.Series(np.nan, index=frame.index, dtype="float64")
    if len(frame) < period:
        return atr
    atr.iloc[period - 1] = float(true_range.iloc[:period].mean())
    for index in range(period, len(frame)):
        atr.iloc[index] = (
            atr.iloc[index - 1] * (period - 1) + true_range.iloc[index]
        ) / period
    return atr / close


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the eleven frozen price/time features; volume is not accessed."""
    close = frame["close"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    log_close = pd.Series(
        np.log(close.to_numpy(dtype=np.float64)), index=frame.index, dtype="float64"
    )
    return_1 = log_close.diff(1)
    timestamp = pd.DatetimeIndex(frame["timestamp"])
    hour = timestamp.hour.to_numpy(dtype=np.float64) + (
        timestamp.minute.to_numpy(dtype=np.float64) / 60.0
    )
    weekday = timestamp.dayofweek.to_numpy(dtype=np.float64)
    values = {
        "log_return_1": return_1,
        "log_return_4": log_close.diff(4),
        "log_return_24": log_close.diff(24),
        "realized_volatility_6": np.sqrt(
            return_1.pow(2).rolling(6, min_periods=6).sum()
        ),
        "realized_volatility_24": np.sqrt(
            return_1.pow(2).rolling(24, min_periods=24).sum()
        ),
        "range_fraction_1": (high - low) / close,
        "atr_fraction_14": wilder_atr_fraction(frame),
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "weekday_sin": np.sin(2.0 * np.pi * weekday / 7.0),
        "weekday_cos": np.cos(2.0 * np.pi * weekday / 7.0),
    }
    return pd.DataFrame(values, index=frame.index).replace([np.inf, -np.inf], np.nan)


def _resolve_bar_touch(upper_hit: bool, lower_hit: bool) -> str | None:
    """Resolve one OHLC bar without guessing intrabar order."""
    if upper_hit == lower_hit:
        return "AMBIGUOUS" if upper_hit else None
    return "UPPER_FIRST" if upper_hit else "LOWER_FIRST"


def _find_first_touch(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    end: int,
    upper: float,
    lower: float,
) -> tuple[str, int]:
    for future_index in range(start, end + 1):
        outcome = _resolve_bar_touch(
            bool(high[future_index] >= upper), bool(low[future_index] <= lower)
        )
        if outcome is not None:
            return outcome, future_index
    return "TIMEOUT", end


def _label_one_decision(
    decision_index: int,
    opened: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volatility: np.ndarray,
    size: int,
    k: float,
    horizon_bars: int,
) -> dict[str, Any]:
    entry_index = decision_index + 1
    end_index = entry_index + horizon_bars - 1
    if end_index >= size or not np.isfinite(volatility[decision_index]):
        return {
            "label": None,
            "outcome": "UNAVAILABLE",
            "entry_index": entry_index,
            "barrier_end_index": None,
        }
    vol_value = float(volatility[decision_index])
    if vol_value <= 0.0:
        return {
            "label": None,
            "outcome": "UNAVAILABLE",
            "entry_index": entry_index,
            "barrier_end_index": None,
        }
    entry = float(opened[entry_index])
    upper = entry * (1.0 + k * vol_value)
    lower = entry * (1.0 - k * vol_value)
    if lower <= 0.0:
        raise RuntimeError("regime_t_lower_barrier_non_positive")
    outcome, barrier_end = _find_first_touch(
        high, low, entry_index, end_index, upper, lower
    )
    label = 1 if outcome == "UPPER_FIRST" else 0 if outcome == "LOWER_FIRST" else None
    return {
        "label": label,
        "outcome": outcome,
        "entry_index": entry_index,
        "barrier_end_index": barrier_end,
        "entry_price": entry,
        "volatility_fraction": vol_value,
        "upper_barrier": upper,
        "lower_barrier": lower,
    }


def build_triple_barrier_labels(
    frame: pd.DataFrame,
    volatility_fraction: pd.Series | None = None,
    k: float = K,
    horizon_bars: int = HORIZON_BARS,
) -> pd.DataFrame:
    """Label upper-first=1, lower-first=0; censor timeouts and ambiguous bars."""
    if not np.isfinite(k) or k != K or horizon_bars != HORIZON_BARS:
        raise ValueError("regime_t_locked_target_parameters_invalid")
    vol = (
        volatility_fraction
        if volatility_fraction is not None
        else wilder_atr_fraction(frame)
    )
    if len(vol) != len(frame) or not vol.index.equals(frame.index):
        raise ValueError("regime_t_volatility_series_alignment_invalid")
    opened = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    volatility = vol.to_numpy(dtype=np.float64)
    records = [
        _label_one_decision(
            index, opened, high, low, volatility, len(frame), k, horizon_bars
        )
        for index in range(len(frame))
    ]
    result = pd.DataFrame(records)
    result.insert(0, "decision_index", np.arange(len(frame), dtype=np.int64))
    result["label"] = result["label"].astype("Int64")
    result["outcome"] = result["outcome"].astype("string")
    return result


def summarize_labels(labels: pd.DataFrame, dataset_sha256: str) -> dict[str, Any]:
    counts = labels["outcome"].value_counts(dropna=False).to_dict()
    return {
        "schema_version": "alix.regime-t.triple-barrier-labels.v1",
        "status": "COMPLETED",
        "dataset_sha256": dataset_sha256,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "row_count": int(len(labels)),
        "class_counts": {
            "upper_first": int(counts.get("UPPER_FIRST", 0)),
            "lower_first": int(counts.get("LOWER_FIRST", 0)),
        },
        "censored_counts": {
            "timeout": int(counts.get("TIMEOUT", 0)),
            "ambiguous": int(counts.get("AMBIGUOUS", 0)),
            "unavailable": int(counts.get("UNAVAILABLE", 0)),
        },
        "safety": {
            "real_data_scope": "PRE_HACK_ONLY",
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "exchange_network_access": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ALIX_REGIME_T_EXECUTION_AUTHORIZED") != "true":
        raise PermissionError("regime_t_requires_owner_execution_authorization")
    if (
        os.environ.get("ALIX_DRY_RUN") != "true"
        or os.environ.get("ALIX_ALLOW_LIVE_TRADING") != "false"
    ):
        raise PermissionError("regime_t_requires_dry_run_and_live_trading_disabled")
    frame, _ = load_verified_prehack_frame()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    labels = build_triple_barrier_labels(
        frame,
        k=float(prereg["target"]["barriers"]["k"]),
        horizon_bars=int(prereg["target"]["time_barrier"]["max_future_bars"]),
    )
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.labels_out, index=False)
    summary = summarize_labels(labels, EXPECTED_DATA_SHA256)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
