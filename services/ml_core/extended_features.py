"""Point-in-time Trend, Volatility, and Volume features for ML Core."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from services.freqai_baseline.features import (
    FEATURE_COLUMNS as BASELINE_FEATURE_COLUMNS,
)
from services.ml_core.causal_features import build_causal_feature_matrix

EXTENDED_FEATURE_SCHEMA_VERSION: Final[str] = "alix.ml-core.extended-features.v1"
OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

FEATURE_GROUP_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "Trend": (
        "%-trend_ema_distance_12",
        "%-trend_ema_distance_26",
        "%-trend_ema_spread_12_26",
        "%-trend_ema_slope_12_6",
        "%-trend_momentum_12",
        "%-trend_adx_14",
        "%-trend_plus_di_14",
        "%-trend_minus_di_14",
    ),
    "Volatility": (
        "%-volatility_realized_12",
        "%-volatility_realized_24",
        "%-volatility_bollinger_bandwidth_20",
    ),
    "Volume": (
        "%-volume_relative_20",
        "%-volume_zscore_20",
        "%-volume_obv_imbalance_20",
        "%-volume_chaikin_money_flow_20",
    ),
}

FEATURE_PROFILE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Baseline": ("Baseline",),
    "Baseline+Trend": ("Baseline", "Trend"),
    "Baseline+Volatility": ("Baseline", "Volatility"),
    "Baseline+Volume": ("Baseline", "Volume"),
    "Combined": ("Baseline", "Trend", "Volatility", "Volume"),
}
FEATURE_PROFILE_ORDER: Final[tuple[str, ...]] = tuple(FEATURE_PROFILE_GROUPS)


def feature_profile_groups(profile: str) -> tuple[str, ...]:
    """Return the immutable group membership for one approved profile."""
    try:
        return FEATURE_PROFILE_GROUPS[profile]
    except KeyError as error:
        raise ValueError("feature_profile_unknown") from error


def feature_profile_columns(profile: str) -> tuple[str, ...]:
    """Return ordered feature names for one approved profile."""
    groups = feature_profile_groups(profile)
    columns: list[str] = []
    for group in groups:
        if group == "Baseline":
            columns.extend(BASELINE_FEATURE_COLUMNS)
        else:
            columns.extend(FEATURE_GROUP_COLUMNS[group])
    return tuple(columns)


def feature_profile_manifest() -> list[dict[str, object]]:
    """Return the stable, ordered definitions used in reports and artifacts."""
    return [
        {
            "name": profile,
            "groups": list(feature_profile_groups(profile)),
            "columns": list(feature_profile_columns(profile)),
            "feature_count": len(feature_profile_columns(profile)),
        }
        for profile in FEATURE_PROFILE_ORDER
    ]


def _validated_utc_timestamps(data: DataFrame) -> Series:
    """Validate a causal, strictly increasing timestamp sequence."""
    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.dt.tz is None:
        raise ValueError("timezone_required")
    utc_dates = dates.dt.tz_convert("UTC")
    if utc_dates.duplicated().any() or not utc_dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = utc_dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=np.float64)
    if deltas.size and (
        not bool(np.isfinite(deltas).all())
        or bool(np.any(deltas <= 0.0))
        or not bool(np.all(deltas == deltas[0]))
    ):
        raise ValueError("timestamp_interval_inconsistent")
    return utc_dates


def _validate_price_volume_geometry(data: DataFrame) -> None:
    """Reject impossible prices, negative volume, and invalid candle geometry."""
    opened = data["open"].to_numpy(dtype=np.float64)
    high = data["high"].to_numpy(dtype=np.float64)
    low = data["low"].to_numpy(dtype=np.float64)
    close = data["close"].to_numpy(dtype=np.float64)
    volume = data["volume"].to_numpy(dtype=np.float64)
    if bool(np.any(np.column_stack((opened, high, low, close)) <= 0.0)):
        raise ValueError("ohlcv_price_must_be_positive")
    if bool(np.any(volume < 0.0)):
        raise ValueError("ohlcv_volume_negative")
    invalid_geometry = (
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    )
    if bool(invalid_geometry):
        raise ValueError("ohlcv_geometry_invalid")


def _validate_numeric_ohlcv(data: DataFrame) -> None:
    """Normalize OHLCV numbers and reject non-finite values."""
    for name in ("open", "high", "low", "close", "volume"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    values = data.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not bool(np.isfinite(values).all()):
        raise ValueError("ohlcv_value_not_finite")
    _validate_price_volume_geometry(data)


def _validated_ohlcv(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    data = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    data["date"] = _validated_utc_timestamps(data)
    _validate_numeric_ohlcv(data)
    return data


def _wilder_average(values: Series, period: int) -> Series:
    """Calculate a causal Wilder-style exponential average."""
    return values.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _trend_features(data: DataFrame) -> DataFrame:
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    upward_move = high.diff()
    downward_move = -low.diff()
    positive_dm = upward_move.where(
        (upward_move > downward_move) & (upward_move > 0.0), 0.0
    ).fillna(0.0)
    negative_dm = downward_move.where(
        (downward_move > upward_move) & (downward_move > 0.0), 0.0
    ).fillna(0.0)

    smoothed_range = _wilder_average(true_range, 14)
    smoothed_positive_dm = _wilder_average(positive_dm, 14)
    smoothed_negative_dm = _wilder_average(negative_dm, 14)
    positive_di = (
        100.0 * smoothed_positive_dm / smoothed_range.where(smoothed_range > 0.0)
    ).where(smoothed_range.notna())
    negative_di = (
        100.0 * smoothed_negative_dm / smoothed_range.where(smoothed_range > 0.0)
    ).where(smoothed_range.notna())
    positive_di = positive_di.mask(smoothed_range.eq(0.0), 0.0)
    negative_di = negative_di.mask(smoothed_range.eq(0.0), 0.0)
    directional_sum = positive_di + negative_di
    directional_index = (
        100.0
        * (positive_di - negative_di).abs()
        / directional_sum.where(directional_sum > 0.0)
    )
    directional_index = directional_index.mask(directional_sum.eq(0.0), 0.0)
    directional_index = directional_index.where(
        positive_di.notna() & negative_di.notna()
    )
    adx = _wilder_average(directional_index, 14)

    return DataFrame(
        {
            "%-trend_ema_distance_12": close / ema_12 - 1.0,
            "%-trend_ema_distance_26": close / ema_26 - 1.0,
            "%-trend_ema_spread_12_26": ema_12 / ema_26 - 1.0,
            "%-trend_ema_slope_12_6": ema_12 / ema_12.shift(6) - 1.0,
            "%-trend_momentum_12": close.pct_change(12, fill_method=None),
            "%-trend_adx_14": adx,
            "%-trend_plus_di_14": positive_di,
            "%-trend_minus_di_14": negative_di,
        },
        index=data.index,
    )


def _volatility_features(data: DataFrame) -> DataFrame:
    close = data["close"].astype(float)
    returns = close.pct_change(fill_method=None)
    rolling_mean = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
    return DataFrame(
        {
            "%-volatility_realized_12": returns.rolling(12, min_periods=12).std(ddof=0),
            "%-volatility_realized_24": returns.rolling(24, min_periods=24).std(ddof=0),
            "%-volatility_bollinger_bandwidth_20": (
                4.0 * rolling_std / rolling_mean.where(rolling_mean > 0.0)
            ),
        },
        index=data.index,
    )


def _volume_features(data: DataFrame) -> DataFrame:
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    rolling_mean = volume.rolling(20, min_periods=20).mean()
    rolling_std = volume.rolling(20, min_periods=20).std(ddof=0)
    relative_volume = volume / rolling_mean.where(rolling_mean > 0.0)
    volume_zscore = (volume - rolling_mean) / rolling_std.where(rolling_std > 0.0)
    volume_zscore = volume_zscore.mask(rolling_std.eq(0.0), 0.0)

    price_change = close.diff().fillna(0.0)
    direction = price_change.gt(0.0).astype(float) - price_change.lt(0.0).astype(float)
    signed_volume = (direction * volume).rolling(20, min_periods=20).sum()
    total_volume = volume.rolling(20, min_periods=20).sum()
    obv_imbalance = signed_volume / total_volume.where(total_volume > 0.0)

    high_low_range = high - low
    money_flow_multiplier = (2.0 * close - high - low) / high_low_range.where(
        high_low_range > 0.0
    )
    money_flow_multiplier = money_flow_multiplier.mask(high_low_range.eq(0.0), 0.0)
    money_flow_volume = money_flow_multiplier * volume
    chaikin_money_flow = money_flow_volume.rolling(
        20, min_periods=20
    ).sum() / total_volume.where(total_volume > 0.0)
    return DataFrame(
        {
            "%-volume_relative_20": relative_volume,
            "%-volume_zscore_20": volume_zscore,
            "%-volume_obv_imbalance_20": obv_imbalance,
            "%-volume_chaikin_money_flow_20": chaikin_money_flow,
        },
        index=data.index,
    )


def build_extended_feature_groups(candles: DataFrame) -> dict[str, DataFrame]:
    """Build causal Trend, Volatility, and OHLCV-proxy Volume feature groups.

    All calculations are right-aligned and use only the current and prior closed
    candles. Rolling warm-up rows intentionally remain null and are never backfilled.
    """
    data = _validated_ohlcv(candles)
    groups = {
        "Trend": _trend_features(data),
        "Volatility": _volatility_features(data),
        "Volume": _volume_features(data),
    }
    for group_name, frame in groups.items():
        expected_columns = FEATURE_GROUP_COLUMNS[group_name]
        if tuple(frame.columns) != expected_columns:
            raise ValueError("extended_feature_group_schema_mismatch")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        values = frame.to_numpy(dtype=np.float64)
        if bool(np.isinf(values).any()):
            raise ValueError("extended_feature_non_finite")
    return groups


def build_feature_profiles(candles: DataFrame) -> dict[str, DataFrame]:
    """Build all five approved profiles with identical row indices and a shared origin."""
    baseline = build_causal_feature_matrix(candles)
    extended_groups = build_extended_feature_groups(candles)
    profiles: dict[str, DataFrame] = {}
    for profile in FEATURE_PROFILE_ORDER:
        frames: list[DataFrame] = []
        for group in feature_profile_groups(profile):
            if group == "Baseline":
                frames.append(baseline)
            else:
                frames.append(extended_groups[group])
        matrix = pd.concat(frames, axis=1)
        if not matrix.columns.is_unique:
            raise ValueError("feature_profile_duplicate_columns")
        if tuple(matrix.columns) != feature_profile_columns(profile):
            raise ValueError("feature_profile_schema_mismatch")
        if any(name.startswith("&-") for name in matrix.columns):
            raise ValueError("target_column_in_feature_matrix")
        matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        if bool(np.isinf(matrix.to_numpy(dtype=np.float64)).any()):
            raise ValueError("feature_profile_non_finite")
        profiles[profile] = matrix
    return profiles
