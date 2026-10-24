"""Causal 4H feature engineering for the research-only AI core."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

AI_FEATURE_SCHEMA_VERSION: Final[str] = "alix.ai-core.features.v1"
SENTIMENT_FEATURE_SCHEMA_VERSION: Final[str] = "alix.ai-core.sentiment-features.v1"
AI_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "return_1",
    "return_2",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "log_return_1",
    "sma_ratio_6",
    "ema_ratio_6",
    "sma_ratio_12",
    "ema_ratio_12",
    "sma_ratio_24",
    "ema_ratio_24",
    "sma_ratio_48",
    "ema_ratio_48",
    "sma_ratio_72",
    "ema_ratio_72",
    "sma_slope_12",
    "sma_slope_24",
    "ema_crossover_6_24",
    "ema_crossover_12_48",
    "ema_crossover_24_72",
    "bollinger_width_12",
    "bollinger_position_12",
    "bollinger_width_24",
    "bollinger_position_24",
    "rsi_6",
    "rsi_14",
    "rsi_28",
    "true_range_pct",
    "atr_ratio_7",
    "atr_ratio_14",
    "atr_ratio_28",
    "realized_volatility_6",
    "realized_volatility_12",
    "realized_volatility_24",
    "realized_volatility_48",
    "downside_volatility_12",
    "downside_volatility_24",
    "volume_return_1",
    "volume_return_6",
    "volume_zscore_12",
    "relative_volume_12",
    "rolling_quote_turnover_ratio_12",
    "volume_zscore_24",
    "relative_volume_24",
    "rolling_quote_turnover_ratio_24",
    "rolling_vwap_distance_12",
    "volume_profile_poc_distance_12",
    "volume_profile_value_area_position_12",
    "rolling_vwap_distance_24",
    "volume_profile_poc_distance_24",
    "volume_profile_value_area_position_24",
    "breakout_distance_12",
    "breakout_distance_24",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "ethusdt_return_1",
    "ethusdt_return_6",
    "ethusdt_correlation_12",
    "ethusdt_correlation_24",
    "ethusdt_beta_24",
    "relative_strength_ethusdt_12",
)
BASE_AI_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "return_1",
    "return_2",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "log_return_1",
    "sma_ratio_6",
    "ema_ratio_6",
    "sma_ratio_12",
    "ema_ratio_12",
    "sma_ratio_24",
    "ema_ratio_24",
    "sma_ratio_48",
    "ema_ratio_48",
    "sma_ratio_72",
    "ema_ratio_72",
    "sma_slope_12",
    "sma_slope_24",
    "ema_crossover_6_24",
    "ema_crossover_12_48",
    "ema_crossover_24_72",
    "bollinger_width_12",
    "bollinger_position_12",
    "bollinger_width_24",
    "bollinger_position_24",
    "rsi_6",
    "rsi_14",
    "rsi_28",
    "true_range_pct",
    "atr_ratio_7",
    "atr_ratio_14",
    "atr_ratio_28",
    "realized_volatility_6",
    "realized_volatility_12",
    "realized_volatility_24",
    "realized_volatility_48",
    "downside_volatility_12",
    "downside_volatility_24",
    "volume_return_1",
    "volume_return_6",
    "volume_zscore_12",
    "relative_volume_12",
    "rolling_quote_turnover_ratio_12",
    "volume_zscore_24",
    "relative_volume_24",
    "rolling_quote_turnover_ratio_24",
    "rolling_vwap_distance_12",
    "volume_profile_poc_distance_12",
    "volume_profile_value_area_position_12",
    "rolling_vwap_distance_24",
    "volume_profile_poc_distance_24",
    "volume_profile_value_area_position_24",
    "breakout_distance_12",
    "breakout_distance_24",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
)
CROSS_ASSET_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "ethusdt_return_1",
    "ethusdt_return_6",
    "ethusdt_correlation_12",
    "ethusdt_correlation_24",
    "ethusdt_beta_24",
    "relative_strength_ethusdt_12",
)

VOLUME_DEPENDENT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "volume_return_1",
    "volume_return_6",
    "volume_zscore_12",
    "relative_volume_12",
    "rolling_quote_turnover_ratio_12",
    "volume_zscore_24",
    "relative_volume_24",
    "rolling_quote_turnover_ratio_24",
    "rolling_vwap_distance_12",
    "volume_profile_poc_distance_12",
    "volume_profile_value_area_position_12",
    "rolling_vwap_distance_24",
    "volume_profile_poc_distance_24",
    "volume_profile_value_area_position_24",
)
PRICE_ONLY_AI_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name
    for name in BASE_AI_FEATURE_COLUMNS
    if name not in VOLUME_DEPENDENT_FEATURE_COLUMNS
)
PRICE_ONLY_EXTENDED_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in AI_FEATURE_COLUMNS if name not in VOLUME_DEPENDENT_FEATURE_COLUMNS
)

OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
SENTIMENT_COLUMNS: Final[tuple[str, ...]] = (
    "sentiment_mean_24h",
    "sentiment_mean_72h",
    "sentiment_weighted_24h",
    "sentiment_event_count_24h",
    "sentiment_confidence_mean_24h",
    "sentiment_available",
)
_TREND_WINDOWS: Final[tuple[int, ...]] = (6, 12, 24, 48, 72)
_RETURN_WINDOWS: Final[tuple[int, ...]] = (1, 2, 3, 6, 12, 24)
_BOLLINGER_WINDOWS: Final[tuple[int, ...]] = (12, 24)
_RSI_WINDOWS: Final[tuple[int, ...]] = (6, 14, 28)
_ATR_WINDOWS: Final[tuple[int, ...]] = (7, 14, 28)
_VOLATILITY_WINDOWS: Final[tuple[int, ...]] = (6, 12, 24, 48)
_CROSS_WINDOWS: Final[tuple[int, ...]] = (12, 24)
_REQUIRED_CROSS_ASSETS: Final[tuple[str, ...]] = ("ETHUSDT",)


def select_model_feature_schema(
    candles_4h: DataFrame, *, cross_asset_available: bool
) -> tuple[tuple[str, ...], str]:
    """Select only feature groups supportable by observed data; never synthesize volume."""
    if "volume" not in candles_4h.columns or candles_4h.empty:
        raise ValueError("ai_core_feature_schema_volume_column_missing")
    volume = pd.to_numeric(candles_4h["volume"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("ai_core_feature_schema_volume_invalid")
    if np.any(volume > 0):
        selected = (
            AI_FEATURE_COLUMNS if cross_asset_available else BASE_AI_FEATURE_COLUMNS
        )
        return selected, "SOURCE_VOLUME_POSITIVE_OBSERVATIONS_PRESENT"
    selected = (
        PRICE_ONLY_EXTENDED_FEATURE_COLUMNS
        if cross_asset_available
        else PRICE_ONLY_AI_FEATURE_COLUMNS
    )
    return selected, "BLOCKED_NO_POSITIVE_SOURCE_VOLUME"



def _validate_required_ohlcv_columns(candles: DataFrame) -> None:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ai_core_ohlcv_columns_missing:{missing}")
    if candles.empty or candles.columns.duplicated().any():
        raise ValueError("ai_core_ohlcv_empty_or_duplicate_columns")


def _validate_timestamp_sequence(
    date_values: Series, interval_seconds: int
) -> Series:
    parsed_dates = pd.to_datetime(date_values, errors="raise")
    if parsed_dates.dt.tz is None:
        raise ValueError("ai_core_timestamps_must_be_timezone_aware")
    normalized_dates = parsed_dates.dt.tz_convert("UTC")
    if normalized_dates.duplicated().any():
        raise ValueError("ai_core_duplicate_timestamps")
    if not normalized_dates.is_monotonic_increasing:
        raise ValueError("ai_core_timestamps_not_chronological")
    gaps = normalized_dates.diff().dropna().dt.total_seconds()
    if not gaps.eq(float(interval_seconds)).all():
        raise ValueError("ai_core_timeframe_gap_or_mismatch")
    return normalized_dates


def _validate_ohlcv_numeric_ranges(prices: np.ndarray, volumes: np.ndarray) -> None:
    if not np.isfinite(prices).all() or not np.isfinite(volumes).all():
        raise ValueError("ai_core_ohlcv_non_finite")
    if (prices <= 0).any() or (volumes < 0).any():
        raise ValueError("ai_core_ohlcv_out_of_range")


def _validate_ohlc_invariants(candles: DataFrame) -> None:
    if (
        (candles["high"] < candles[["open", "close"]].max(axis=1)).any()
        or (candles["low"] > candles[["open", "close"]].min(axis=1)).any()
        or (candles["high"] < candles["low"]).any()
    ):
        raise ValueError("ai_core_ohlc_invariant_failed")


def _validate_ohlcv(candles: DataFrame, interval_seconds: int) -> DataFrame:
    _validate_required_ohlcv_columns(candles)
    result = candles.loc[:, list(OHLCV_COLUMNS)].copy()
    result["date"] = _validate_timestamp_sequence(result["date"], interval_seconds)
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    prices = result[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volumes = result["volume"].to_numpy(dtype=float)
    _validate_ohlcv_numeric_ranges(prices, volumes)
    _validate_ohlc_invariants(result)
    return result

def aggregate_hourly_to_4h(candles: DataFrame) -> DataFrame:
    """Aggregate exactly four contiguous hourly candles; never fill gaps or partial bars."""
    hourly = _validate_ohlcv(candles, 3600)
    if len(hourly) < 4 or len(hourly) % 4:
        raise ValueError("ai_core_hourly_rows_must_form_complete_4h_bars")
    hourly = hourly.assign(_bar_group=np.arange(len(hourly), dtype=int) // 4)
    aggregated = hourly.groupby("_bar_group", sort=True).agg(
        date=("date", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    result = aggregated.reset_index(drop=True)
    return _validate_ohlcv(result, 4 * 3600)


def _rsi(close: Series, period: int) -> Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    losses = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    denominator = losses.where(losses > 0)
    relative_strength = gains / denominator
    value = 100.0 - (100.0 / (1.0 + relative_strength))
    value = value.mask((losses == 0) & (gains > 0), 100.0)
    return value.mask((losses == 0) & (gains == 0), 50.0)


def _true_range(high: Series, low: Series, close: Series) -> Series:
    previous = close.shift(1)
    return pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)


def _volume_profile_proxy(
    typical_price: Series, volume: Series, close: Series, window: int
) -> tuple[Series, Series]:
    """Coarse OHLCV profile proxy; not a trade-level volume-at-price feed."""
    point_of_control = np.full(len(typical_price), np.nan, dtype=float)
    value_area_position = np.full(len(typical_price), np.nan, dtype=float)
    price_values = typical_price.to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    close_values = close.to_numpy(dtype=float)

    for end in range(window - 1, len(price_values)):
        prices = price_values[end - window + 1 : end + 1]
        volumes = volume_values[end - window + 1 : end + 1]
        total_volume = float(volumes.sum())
        if total_volume <= 0 or not np.isfinite(prices).all():
            continue
        if float(prices.max()) == float(prices.min()):
            point_of_control[end] = float(prices[0])
            value_area_position[end] = 0.5
            continue
        histogram, edges = np.histogram(prices, bins=8, weights=volumes)
        point_index = int(np.argmax(histogram))
        point_of_control[end] = float(
            (edges[point_index] + edges[point_index + 1]) / 2.0
        )
        ranked = np.argsort(histogram)[::-1]
        selected: list[int] = []
        selected_volume = 0.0
        for index in ranked:
            selected.append(int(index))
            selected_volume += float(histogram[index])
            if selected_volume >= 0.70 * total_volume:
                break
        lower = float(edges[min(selected)])
        upper = float(edges[max(selected) + 1])
        span = upper - lower
        value_area_position[end] = (
            float(np.clip((close_values[end] - lower) / span, 0.0, 1.0))
            if span > 0
            else 0.5
        )
    poc_distance = pd.Series(point_of_control, index=close.index) / close - 1.0
    return poc_distance, pd.Series(value_area_position, index=close.index)



def _trend_feature_values(close: Series, log_returns: Series) -> dict[str, Series]:
    values: dict[str, Series] = {}
    for window in _RETURN_WINDOWS:
        values[f"return_{window}"] = close.pct_change(window, fill_method=None)
    values["log_return_1"] = log_returns
    averages: dict[int, Series] = {}
    exponential: dict[int, Series] = {}
    for window in _TREND_WINDOWS:
        averages[window] = close.rolling(window, min_periods=window).mean()
        exponential[window] = close.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()
        values[f"sma_ratio_{window}"] = close / averages[window] - 1.0
        values[f"ema_ratio_{window}"] = close / exponential[window] - 1.0
    for window in (12, 24):
        values[f"sma_slope_{window}"] = averages[window].pct_change(3, fill_method=None)
    for fast, slow in ((6, 24), (12, 48), (24, 72)):
        values[f"ema_crossover_{fast}_{slow}"] = (
            exponential[fast] / exponential[slow] - 1.0
        )
    for window in _BOLLINGER_WINDOWS:
        mean = close.rolling(window, min_periods=window).mean()
        deviation = close.rolling(window, min_periods=window).std(ddof=0)
        upper = mean + 2.0 * deviation
        lower = mean - 2.0 * deviation
        width = (upper - lower) / mean
        values[f"bollinger_width_{window}"] = width
        values[f"bollinger_position_{window}"] = (
            (close - lower) / (upper - lower).where((upper - lower) > 0)
        ).mask((upper - lower) == 0, 0.5)
    return values


def _volatility_feature_values(
    close: Series, returns: Series, true_range: Series
) -> dict[str, Series]:
    values: dict[str, Series] = {}
    for window in _RSI_WINDOWS:
        values[f"rsi_{window}"] = _rsi(close, window)
    values["true_range_pct"] = true_range / close
    for window in _ATR_WINDOWS:
        values[f"atr_ratio_{window}"] = (
            true_range.rolling(window, min_periods=window).mean() / close
        )
    for window in _VOLATILITY_WINDOWS:
        values[f"realized_volatility_{window}"] = returns.rolling(
            window, min_periods=window
        ).std(ddof=0)
    for window in (12, 24):
        values[f"downside_volatility_{window}"] = (
            returns.clip(upper=0).rolling(window, min_periods=window).std(ddof=0)
        )
    return values


def _volume_feature_values(
    opened: Series,
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    typical_price: Series,
) -> dict[str, Series]:
    values: dict[str, Series] = {
        "volume_return_1": volume.pct_change(fill_method=None),
        "volume_return_6": volume.pct_change(6, fill_method=None),
    }
    for window in (12, 24):
        rolling_volume = volume.rolling(window, min_periods=window)
        mean = rolling_volume.mean()
        deviation = rolling_volume.std(ddof=0).where(rolling_volume.std(ddof=0) > 0)
        values[f"volume_zscore_{window}"] = (volume - mean) / deviation
        values[f"relative_volume_{window}"] = volume / mean.where(mean > 0)
        quote_turnover = close * volume
        rolling_turnover = quote_turnover.rolling(window, min_periods=window).sum()
        values[f"rolling_quote_turnover_ratio_{window}"] = (
            rolling_turnover
            / rolling_turnover.rolling(window, min_periods=window).median()
        ) - 1.0
    for window in (12, 24):
        vwap = (typical_price * volume).rolling(
            window, min_periods=window
        ).sum() / volume.rolling(window, min_periods=window).sum().where(
            volume.rolling(window, min_periods=window).sum() > 0
        )
        values[f"rolling_vwap_distance_{window}"] = close / vwap - 1.0
        poc, value_area = _volume_profile_proxy(typical_price, volume, close, window)
        values[f"volume_profile_poc_distance_{window}"] = poc
        values[f"volume_profile_value_area_position_{window}"] = value_area
    for window in (12, 24):
        previous_high = high.shift(1).rolling(window, min_periods=window).max()
        values[f"breakout_distance_{window}"] = close / previous_high - 1.0
    values["candle_body_ratio"] = (close - opened) / (high - low).where(high > low)
    values["upper_wick_ratio"] = (
        high - pd.concat([opened, close], axis=1).max(axis=1)
    ) / (high - low).where(high > low)
    values["lower_wick_ratio"] = (
        pd.concat([opened, close], axis=1).min(axis=1) - low
    ) / (high - low).where(high > low)
    return values


def _base_feature_values(candles: DataFrame) -> dict[str, Series]:
    opened = candles["open"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)
    volume = candles["volume"].astype(float)
    returns = close.pct_change(fill_method=None)
    log_returns = pd.Series(
        np.log(close.to_numpy(dtype=float)), index=close.index
    ).diff()
    typical_price = (high + low + close) / 3.0
    true_range = _true_range(high, low, close)
    values = _trend_feature_values(close, log_returns)
    values.update(_volatility_feature_values(close, returns, true_range))
    values.update(_volume_feature_values(opened, high, low, close, volume, typical_price))
    return values

def _cross_asset_features(
    base: DataFrame, cross_asset: dict[str, DataFrame] | None
) -> dict[str, Series]:
    output: dict[str, Series] = {}
    if cross_asset is None:
        return output
    missing_symbols = sorted(set(_REQUIRED_CROSS_ASSETS) - set(cross_asset))
    if missing_symbols:
        raise ValueError(f"ai_core_cross_asset_sources_missing:{missing_symbols}")
    base_close = base["close"].astype(float)
    base_returns = base_close.pct_change(fill_method=None)
    for symbol in _REQUIRED_CROSS_ASSETS:
        close = pd.Series(np.nan, index=base.index, dtype=float)
        if cross_asset is not None and symbol in cross_asset:
            asset = _validate_ohlcv(cross_asset[symbol], 4 * 3600)
            aligned = asset.set_index("date")["close"].reindex(base["date"])
            if aligned.isna().any():
                raise ValueError(f"ai_core_cross_asset_alignment_incomplete:{symbol}")
            close = pd.Series(aligned.to_numpy(dtype=float), index=base.index)
        asset_returns = close.pct_change(fill_method=None)
        key = symbol.lower()
        output[f"{key}_return_1"] = asset_returns
        output[f"{key}_return_6"] = close.pct_change(6, fill_method=None)
        for window in _CROSS_WINDOWS:
            output[f"{key}_correlation_{window}"] = base_returns.rolling(
                window, min_periods=window
            ).corr(asset_returns)
        rolling_variance = asset_returns.rolling(24, min_periods=24).var(ddof=0)
        rolling_covariance = base_returns.rolling(24, min_periods=24).cov(asset_returns)
        output[f"{key}_beta_24"] = rolling_covariance / rolling_variance.where(
            rolling_variance > 0
        )
        output[f"relative_strength_{key}_12"] = base_close.pct_change(
            12, fill_method=None
        ) - close.pct_change(12, fill_method=None)
    return output


def build_ai_features(
    candles_4h: DataFrame,
    *,
    cross_asset: dict[str, DataFrame] | None = None,
    sentiment_features: DataFrame | None = None,
) -> DataFrame:
    """Create causal features from validated 4H candles and point-in-time side inputs."""
    candles = _validate_ohlcv(candles_4h, 4 * 3600)
    base_values = _base_feature_values(candles)
    cross_values = _cross_asset_features(candles, cross_asset)
    values = {**base_values, **cross_values}
    features = DataFrame(values, index=candles.index).replace([np.inf, -np.inf], np.nan)
    result = pd.concat([candles, features], axis=1)
    if sentiment_features is not None:
        required = {"date", *SENTIMENT_COLUMNS}
        if not required.issubset(sentiment_features.columns):
            raise ValueError("ai_core_sentiment_feature_schema_invalid")
        sentiment = sentiment_features.loc[:, ["date", *SENTIMENT_COLUMNS]].copy()
        sentiment["date"] = pd.to_datetime(sentiment["date"], utc=True, errors="raise")
        if sentiment["date"].duplicated().any():
            raise ValueError("ai_core_sentiment_duplicate_timestamps")
        result = result.merge(
            sentiment, how="left", on="date", validate="one_to_one", sort=False
        )
    return result


__all__ = [
    "AI_FEATURE_COLUMNS",
    "AI_FEATURE_SCHEMA_VERSION",
    "BASE_AI_FEATURE_COLUMNS",
    "CROSS_ASSET_FEATURE_COLUMNS",
    "SENTIMENT_COLUMNS",
    "SENTIMENT_FEATURE_SCHEMA_VERSION",
    "VOLUME_DEPENDENT_FEATURE_COLUMNS",
    "PRICE_ONLY_AI_FEATURE_COLUMNS",
    "PRICE_ONLY_EXTENDED_FEATURE_COLUMNS",
    "select_model_feature_schema",
    "aggregate_hourly_to_4h",
    "build_ai_features",
]
