"""Causal OHLCV and timestamp-aligned microstructure feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from services.ml_core.adaptive_targets import calculate_atr

REQUIRED_OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
REQUIRED_MICROSTRUCTURE_COLUMNS = (
    "date",
    "order_book_imbalance",
    "taker_buy_volume",
    "taker_sell_volume",
    "funding_rate",
    "microstructure_timestamp",
    "funding_timestamp",
)
FEATURE_COLUMNS = (
    "return_1",
    "return_3",
    "log_return_1",
    "realized_volatility_24",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_14",
    "atr_fraction",
    "bollinger_percent_b",
    "bollinger_width",
    "volume_zscore_24",
    "range_fraction",
    "order_book_imbalance",
    "taker_buy_sell_log_ratio",
    "funding_rate",
)
REGIME_OBSERVATION_COLUMNS = (
    "log_return_1",
    "realized_volatility_24",
    "range_fraction",
)
MICROSTRUCTURE_INTERVAL = pd.Timedelta(hours=1)


def _timezone_series(values: Series, field: str) -> Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any() or parsed.dt.tz is None:
        raise ValueError(f"{field}_must_be_valid_timezone_aware_timestamps")
    return parsed.dt.tz_convert("UTC")


def _validate_ohlcv(ohlcv: DataFrame, interval: pd.Timedelta) -> DataFrame:
    missing = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(ohlcv.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if ohlcv.empty:
        raise ValueError("ohlcv_empty")
    data = ohlcv.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy().reset_index(drop=True)
    data["date"] = _timezone_series(data["date"], "ohlcv_date")
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("ohlcv_timestamps_must_be_unique_and_increasing")
    if len(data) > 1 and not bool((data["date"].diff().iloc[1:] == interval).all()):
        raise ValueError("ohlcv_interval_must_be_contiguous")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not bool(np.isfinite(data[column].to_numpy(dtype=float)).all()):
            raise ValueError(f"ohlcv_non_finite:{column}")
    prices = data.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    if bool((prices <= 0).any()) or bool((data["volume"].to_numpy(dtype=float) < 0).any()):
        raise ValueError("ohlcv_values_out_of_range")
    opened = data["open"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    if bool((high < np.maximum(opened, close)).any()):
        raise ValueError("ohlcv_high_below_open_or_close")
    if bool((low > np.minimum(opened, close)).any()) or bool((low > high).any()):
        raise ValueError("ohlcv_low_above_open_close_or_high")
    return data


def merge_market_archives(
    ohlcv: DataFrame,
    microstructure: DataFrame,
    *,
    interval: pd.Timedelta = MICROSTRUCTURE_INTERVAL,
) -> DataFrame:
    """Exact-join archived market observations; never interpolate missing inputs.

    The microstructure timestamp must fall inside the associated completed
    candle. Funding timestamps may be older, but can never be in the future.
    Funding is an external context series, not a native Nobitex Spot field.
    """
    if interval <= pd.Timedelta(0):
        raise ValueError("market_interval_must_be_positive")
    missing = sorted(set(REQUIRED_MICROSTRUCTURE_COLUMNS) - set(microstructure.columns))
    if missing:
        raise ValueError(f"microstructure_missing_columns:{','.join(missing)}")
    prices = _validate_ohlcv(ohlcv, interval)
    micro = microstructure.loc[:, list(REQUIRED_MICROSTRUCTURE_COLUMNS)].copy()
    micro = micro.reset_index(drop=True)
    micro["date"] = _timezone_series(micro["date"], "microstructure_date")
    if micro["date"].duplicated().any() or not micro["date"].is_monotonic_increasing:
        raise ValueError("microstructure_timestamps_must_be_unique_and_increasing")
    if not prices["date"].equals(micro["date"]):
        raise ValueError("microstructure_archive_must_exactly_match_ohlcv_timestamps")
    micro["microstructure_timestamp"] = _timezone_series(
        micro["microstructure_timestamp"], "microstructure_timestamp"
    )
    micro["funding_timestamp"] = _timezone_series(
        micro["funding_timestamp"], "funding_timestamp"
    )
    candle_end = prices["date"] + interval
    if bool((micro["microstructure_timestamp"] < prices["date"]).any()) or bool(
        (micro["microstructure_timestamp"] > candle_end).any()
    ):
        raise ValueError("microstructure_timestamp_outside_completed_candle")
    if bool((micro["funding_timestamp"] > candle_end).any()):
        raise ValueError("funding_timestamp_is_from_the_future")
    for column in (
        "order_book_imbalance",
        "taker_buy_volume",
        "taker_sell_volume",
        "funding_rate",
    ):
        micro[column] = pd.to_numeric(micro[column], errors="coerce")
        if not bool(np.isfinite(micro[column].to_numpy(dtype=float)).all()):
            raise ValueError(f"microstructure_non_finite:{column}")
    imbalance = micro["order_book_imbalance"].to_numpy(dtype=float)
    if bool(((imbalance < -1.0) | (imbalance > 1.0)).any()):
        raise ValueError("order_book_imbalance_out_of_range")
    buy = micro["taker_buy_volume"].to_numpy(dtype=float)
    sell = micro["taker_sell_volume"].to_numpy(dtype=float)
    if bool((buy < 0).any()) or bool((sell < 0).any()) or bool(((buy + sell) <= 0).any()):
        raise ValueError("taker_volume_must_be_positive_and_nonnegative_by_side")
    merged = prices.merge(micro, on="date", how="inner", validate="one_to_one")
    if len(merged) != len(prices):
        raise ValueError("microstructure_merge_row_count_mismatch")
    return merged


def _rsi(close: Series, period: int = 14) -> Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    losses = (-delta.clip(upper=0.0)).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    relative_strength = gains / losses.replace(0.0, np.nan)
    values = 100.0 - (100.0 / (1.0 + relative_strength))
    values = values.mask((losses == 0.0) & (gains > 0.0), 100.0)
    values = values.mask((losses == 0.0) & (gains == 0.0), 50.0)
    return values.rename("rsi_14")


def build_causal_features(
    market_data: DataFrame,
    *,
    interval: pd.Timedelta = MICROSTRUCTURE_INTERVAL,
) -> DataFrame:
    """Build a finite, causal feature matrix from completed candles only.

    The first rows remain NaN until each rolling indicator has enough history.
    Callers must drop those rows explicitly; this function never fills them.
    """
    required = set(REQUIRED_OHLCV_COLUMNS) | set(REQUIRED_MICROSTRUCTURE_COLUMNS[1:])
    missing = sorted(required - set(market_data.columns))
    if missing:
        raise ValueError(f"market_data_missing_columns:{','.join(missing)}")
    data = _validate_ohlcv(market_data, interval)
    for column in REQUIRED_MICROSTRUCTURE_COLUMNS[1:]:
        if column not in data.columns:
            raise ValueError(f"market_data_missing_columns:{column}")
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    log_return = np.log(close).diff()
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    upper = middle + 2.0 * deviation
    lower = middle - 2.0 * deviation
    band_width = upper - lower
    atr = calculate_atr(data, period=14, expected_interval=interval)
    volume_mean = volume.rolling(24, min_periods=24).mean()
    volume_std = volume.rolling(24, min_periods=24).std(ddof=0)
    buy = data["taker_buy_volume"].astype(float)
    sell = data["taker_sell_volume"].astype(float)
    ratio = np.log((buy + 1e-12) / (sell + 1e-12))
    features = DataFrame(
        {
            "return_1": close.pct_change(fill_method=None),
            "return_3": close.pct_change(periods=3, fill_method=None),
            "log_return_1": log_return,
            "realized_volatility_24": log_return.rolling(24, min_periods=24).std(ddof=0),
            "rsi_14": _rsi(close),
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd - macd_signal,
            "atr_14": atr,
            "atr_fraction": atr / close,
            "bollinger_percent_b": (close - lower) / band_width.replace(0.0, np.nan),
            "bollinger_width": band_width / middle.replace(0.0, np.nan),
            "volume_zscore_24": (volume - volume_mean) / volume_std.replace(0.0, np.nan),
            "range_fraction": (high - low) / close,
            "order_book_imbalance": data["order_book_imbalance"].astype(float),
            "taker_buy_sell_log_ratio": ratio,
            "funding_rate": data["funding_rate"].astype(float),
        },
        index=data.index,
    )
    result = features.loc[:, list(FEATURE_COLUMNS)].copy()
    if tuple(result.columns) != FEATURE_COLUMNS:
        raise ValueError("feature_column_contract_mismatch")
    if any(name.startswith("&-") for name in result.columns):
        raise ValueError("target_column_in_feature_matrix")
    return result
