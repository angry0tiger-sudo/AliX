"""Causal point-in-time funding and OI features for offline ML research."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

EXTERNAL_FEATURE_COLUMNS = ("funding_rate_zscore", "oi_change")
FUNDING_ROLLING_WINDOW = 30
FUNDING_MINIMUM_PERIODS = 20
FUNDING_MAX_AGE = pd.Timedelta(days=2)
FUNDING_MAX_EVENT_GAP = pd.Timedelta(days=2)
PUBLICATION_LAG = pd.Timedelta(minutes=5)
SUPPORTED_TIMEFRAMES = {"4h": pd.Timedelta(hours=4)}


def _utc_timestamp(value: object, field_name: str) -> pd.Timestamp:
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


def _numeric_series(frame: DataFrame, name: str, *, positive: bool) -> Series:
    if name not in frame.columns:
        raise ValueError(f"ai6_external_column_missing:{name}")
    values = pd.to_numeric(frame[name], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"ai6_external_numeric_invalid:{name}")
    if positive and bool((values <= 0.0).any()):
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
    event_time = _utc_series(funding["funding_time"], "funding_time").reset_index(drop=True)
    available = _utc_series(funding["available_at"], "funding_available_at").reset_index(drop=True)
    rates = _numeric_series(funding, "funding_rate", positive=False)
    _validate_ordered_timestamps(event_time, "funding")
    _validate_ordered_timestamps(available, "funding_available")
    if bool((available < event_time + PUBLICATION_LAG).any()):
        raise ValueError("ai6_funding_available_before_conservative_lag")
    return event_time, available, rates


def _validate_open_interest(
    open_interest: DataFrame,
    *,
    symbol: str,
    period: str,
) -> tuple[Series, Series, Series]:
    required = {"symbol", "period", "timestamp", "sum_open_interest", "available_at"}
    if not required.issubset(open_interest.columns) or open_interest.empty:
        raise ValueError("ai6_oi_schema_or_data_invalid")
    if not open_interest["symbol"].eq(symbol).all():
        raise ValueError("ai6_oi_symbol_mismatch")
    if not open_interest["period"].eq(period).all() or period != "4h":
        raise ValueError("ai6_oi_period_mismatch")
    event_time = _utc_series(open_interest["timestamp"], "oi_timestamp").reset_index(drop=True)
    available = _utc_series(open_interest["available_at"], "oi_available_at").reset_index(drop=True)
    values = _numeric_series(open_interest, "sum_open_interest", positive=True)
    _validate_ordered_timestamps(event_time, "open_interest")
    _validate_ordered_timestamps(available, "open_interest_available")
    required_available = event_time + SUPPORTED_TIMEFRAMES[period] + PUBLICATION_LAG
    if bool((available < required_available).any()):
        raise ValueError("ai6_oi_available_before_period_close")
    return event_time, available, values


def _segmented_funding_zscore(
    event_time: Series,
    rates: Series,
    *,
    window: int,
    minimum_periods: int,
) -> Series:
    """Reset rolling statistics after a prolonged funding observation gap."""
    groups = event_time.diff().gt(FUNDING_MAX_EVENT_GAP).cumsum()
    result = Series(np.nan, index=rates.index, dtype=float)
    for _, indexes in rates.groupby(groups).groups.items():
        selected = rates.loc[indexes]
        mean = selected.rolling(window=window, min_periods=minimum_periods).mean()
        std = selected.rolling(window=window, min_periods=minimum_periods).std(ddof=0)
        zscore = (selected - mean) / std.where(std > 1e-12)
        result.loc[indexes] = zscore
    return result


def _open_interest_change(
    event_time: Series,
    values: Series,
    *,
    period: str,
) -> Series:
    """Use observed adjacent snapshots only; a missing interval breaks the change."""
    elapsed = event_time.diff()
    maximum_interval = SUPPORTED_TIMEFRAMES[period] * 2
    changes = values.pct_change(fill_method=None)
    return changes.where(elapsed.le(maximum_interval))


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


def _feature_diagnostics(features: DataFrame) -> dict[str, int | float | str]:
    funding_valid = int(features["funding_rate_zscore"].notna().sum())
    oi_valid = int(features["oi_change"].notna().sum())
    both_valid = int(features.notna().all(axis=1).sum())
    return {
        "feature_schema": "alix.ml-core.ai6-external-features.v1",
        "row_count": int(len(features)),
        "funding_valid_rows": funding_valid,
        "oi_change_valid_rows": oi_valid,
        "complete_external_feature_rows": both_valid,
        "funding_missing_rows": int(len(features) - funding_valid),
        "oi_missing_rows": int(len(features) - oi_valid),
        "complete_feature_fraction": (
            float(both_valid / len(features)) if len(features) else 0.0
        ),
        "interpolation_used": False,
        "backfill_used": False,
        "future_observations_joined": 0,
    }


def build_external_market_features(
    bars: DataFrame,
    funding: DataFrame,
    open_interest: DataFrame,
    *,
    symbol: str,
    timeframe: str = "4h",
    oi_period: str = "4h",
    funding_max_age: pd.Timedelta = FUNDING_MAX_AGE,
) -> tuple[DataFrame, dict[str, int | float | str]]:
    """Join funding and OI using only observations available by each bar close.

    Funding's historical availability is conservatively approximated as
    fundingTime + 5 minutes. OI is treated as a period aggregate and becomes
    available only after the period close + 5 minutes. These are explicit
    research proxies, not claims about Binance's original publication clock.
    """
    if not np.isfinite(funding_max_age.total_seconds()) or funding_max_age <= timedelta(0):
        raise ValueError("ai6_funding_max_age_invalid")
    _, decisions = _validate_bars(bars, timeframe)
    funding_time, funding_available, funding_rate = _validate_funding(
        funding, symbol=symbol
    )
    oi_time, oi_available, oi_values = _validate_open_interest(
        open_interest, symbol=symbol, period=oi_period
    )
    zscore = _segmented_funding_zscore(
        funding_time,
        funding_rate,
        window=FUNDING_ROLLING_WINDOW,
        minimum_periods=FUNDING_MINIMUM_PERIODS,
    )
    oi_change = _open_interest_change(oi_time, oi_values, period=oi_period)
    output = DataFrame(
        {
            "funding_rate_zscore": _asof_values(
                decisions,
                funding_available,
                zscore,
                maximum_age=funding_max_age,
            ),
            "oi_change": _asof_values(
                decisions,
                oi_available,
                oi_change,
                maximum_age=SUPPORTED_TIMEFRAMES[oi_period] * 3 + PUBLICATION_LAG,
            ),
        }
    )
    output.index = bars.index
    output = output.loc[:, list(EXTERNAL_FEATURE_COLUMNS)].astype(float)
    values = output.to_numpy(dtype=float)
    if bool(np.isinf(values).any()):
        raise ValueError("ai6_external_feature_non_finite")
    return output, _feature_diagnostics(output)
