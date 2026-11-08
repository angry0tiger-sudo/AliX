"""Causal point-in-time Funding Rate features for offline ML research."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

EXTERNAL_FEATURE_COLUMNS = ("funding_rate_zscore", "funding_rate_ma")
FUNDING_ROLLING_WINDOW = 30
FUNDING_MINIMUM_PERIODS = 20
FUNDING_MAX_AGE = pd.Timedelta(days=2)
FUNDING_MAX_EVENT_GAP = pd.Timedelta(days=2)
PUBLICATION_LAG = pd.Timedelta(minutes=5)
SUPPORTED_TIMEFRAMES = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def _utc_timestamp(value: object, field_name: str) -> pd.Timestamp:
    if not isinstance(value, (str, datetime, np.datetime64, pd.Timestamp)):
        raise ValueError(f"ai6_timestamp_invalid:{field_name}")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"ai6_timestamp_invalid:{field_name}") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"ai6_timestamp_timezone_required:{field_name}")
    return timestamp.tz_convert("UTC")


def _utc_series(values: Series, field_name: str) -> Series:
    return Series(
        [_utc_timestamp(value, field_name) for value in values.tolist()],
        index=values.index,
        dtype="datetime64[ns, UTC]",
    )


def _validate_ordered_timestamps(values: Series, reason: str) -> None:
    if values.empty or values.isna().any():
        raise ValueError(f"ai6_timestamp_sequence_invalid:{reason}")
    if values.duplicated().any() or not values.is_monotonic_increasing:
        raise ValueError(f"ai6_timestamp_sequence_not_unique_increasing:{reason}")


def _validate_bars(bars: DataFrame, timeframe: str) -> tuple[Series, Series]:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError("ai6_timeframe_not_supported")
    if bars.empty or "date" not in bars.columns:
        raise ValueError("ai6_bars_schema_invalid")
    dates = _utc_series(bars["date"], "bar_date").reset_index(drop=True)
    _validate_ordered_timestamps(dates, "bars")
    decisions = dates + SUPPORTED_TIMEFRAMES[timeframe]
    return dates, decisions


def _numeric_series(frame: DataFrame, name: str) -> Series:
    if name not in frame.columns:
        raise ValueError(f"ai6_external_column_missing:{name}")
    values = pd.to_numeric(frame[name], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"ai6_external_numeric_invalid:{name}")
    return values.reset_index(drop=True)


def _validate_funding(
    funding: DataFrame,
    *,
    symbol: str,
) -> tuple[Series, Series, Series]:
    required = {"symbol", "funding_time", "funding_rate", "available_at"}
    if not required.issubset(funding.columns) or funding.empty:
        raise ValueError("ai6_funding_schema_or_data_invalid")
    if not funding["symbol"].eq(symbol).all():
        raise ValueError("ai6_funding_symbol_mismatch")
    event_time = _utc_series(
        funding["funding_time"], "funding_time"
    ).reset_index(drop=True)
    available = _utc_series(
        funding["available_at"], "funding_available_at"
    ).reset_index(drop=True)
    rates = _numeric_series(funding, "funding_rate")
    _validate_ordered_timestamps(event_time, "funding")
    _validate_ordered_timestamps(available, "funding_available")
    if bool((available < event_time + PUBLICATION_LAG).any()):
        raise ValueError("ai6_funding_available_before_conservative_lag")
    return event_time, available, rates


def _segmented_funding_features(
    event_time: Series,
    rates: Series,
    *,
    window: int,
    minimum_periods: int,
) -> tuple[Series, Series]:
    """Reset rolling funding moments after a prolonged settlement gap."""
    groups = event_time.diff().gt(FUNDING_MAX_EVENT_GAP).cumsum()
    zscore = Series(np.nan, index=rates.index, dtype=float)
    moving_average = Series(np.nan, index=rates.index, dtype=float)
    for _, indexes in rates.groupby(groups).groups.items():
        selected = rates.loc[indexes]
        rolling = selected.rolling(window=window, min_periods=minimum_periods)
        mean = rolling.mean()
        standard_deviation = rolling.std(ddof=0)
        zscore.loc[indexes] = (
            (selected - mean) / standard_deviation.where(standard_deviation > 1e-12)
        )
        moving_average.loc[indexes] = mean
    return zscore, moving_average


def _asof_values(
    decisions: Series,
    available_at: Series,
    feature_values: Series,
    *,
    maximum_age: pd.Timedelta,
) -> Series:
    decision_ns = decisions.astype("int64").to_numpy()
    available_ns = available_at.astype("int64").to_numpy()
    source_values = feature_values.to_numpy(dtype=float)
    output = np.full(len(decisions), np.nan, dtype=float)
    selected = np.searchsorted(available_ns, decision_ns, side="right") - 1
    in_range = selected >= 0
    target_rows = np.flatnonzero(in_range)
    source_rows = selected[in_range]
    age_ns = decision_ns[target_rows] - available_ns[source_rows]
    fresh = (age_ns >= 0) & (age_ns <= maximum_age.value)
    output[target_rows[fresh]] = source_values[source_rows[fresh]]
    return Series(output, index=decisions.index, dtype=float)


def _feature_diagnostics(features: DataFrame) -> dict[str, int | float | str | bool]:
    valid_zscore = int(features["funding_rate_zscore"].notna().sum())
    valid_ma = int(features["funding_rate_ma"].notna().sum())
    complete = int(features.notna().all(axis=1).sum())
    return {
        "feature_schema": "alix.ml-core.ai6-funding-features.v2",
        "row_count": int(len(features)),
        "funding_zscore_valid_rows": valid_zscore,
        "funding_ma_valid_rows": valid_ma,
        "complete_funding_feature_rows": complete,
        "funding_missing_rows": int(len(features) - complete),
        "complete_feature_fraction": (
            float(complete / len(features)) if len(features) else 0.0
        ),
        "interpolation_used": False,
        "backfill_used": False,
        "future_observations_joined": 0,
    }


def build_external_market_features(
    bars: DataFrame,
    funding: DataFrame,
    *,
    symbol: str,
    timeframe: str,
    funding_max_age: pd.Timedelta = FUNDING_MAX_AGE,
) -> tuple[DataFrame, dict[str, int | float | str | bool]]:
    """Join funding z-score and mean using only settlements available by bar close.

    Funding publication is conservatively approximated as fundingTime + 5 minutes.
    This is a research availability proxy, not a provider-certified publication time.
    """
    if (
        not np.isfinite(funding_max_age.total_seconds())
        or funding_max_age <= timedelta(0)
    ):
        raise ValueError("ai6_funding_max_age_invalid")
    _, decisions = _validate_bars(bars, timeframe)
    event_time, available, rates = _validate_funding(funding, symbol=symbol)
    zscore, moving_average = _segmented_funding_features(
        event_time,
        rates,
        window=FUNDING_ROLLING_WINDOW,
        minimum_periods=FUNDING_MINIMUM_PERIODS,
    )
    output = DataFrame(
        {
            "funding_rate_zscore": _asof_values(
                decisions, available, zscore, maximum_age=funding_max_age
            ),
            "funding_rate_ma": _asof_values(
                decisions, available, moving_average, maximum_age=funding_max_age
            ),
        }
    )
    output.index = bars.index
    output = output.loc[:, list(EXTERNAL_FEATURE_COLUMNS)].astype(float)
    if bool(np.isinf(output.to_numpy(dtype=float)).any()):
        raise ValueError("ai6_external_feature_non_finite")
    return output, _feature_diagnostics(output)
