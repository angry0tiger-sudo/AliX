"""Deterministic high-volatility regime selection using training data only."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
TIMESTAMP_COLUMN = 0
CLOSE_COLUMN = 4
DEFAULT_REGIME_WINDOW_CANDLES = 120_000
DEFAULT_CANDIDATE_STRIDE_CANDLES = 1_440
DEFAULT_VOLATILITY_QUANTILE = 0.75


def _validate_window_shape(
    training_data: Float64Array, window_candles: int
) -> None:
    if training_data.ndim != 2 or training_data.shape[1] != 6:
        raise ValueError("market_data_must_have_six_columns")
    if isinstance(window_candles, bool) or not isinstance(window_candles, int):
        raise ValueError("regime_window_candles_must_be_integer")
    if window_candles < 2 or window_candles > len(training_data):
        raise ValueError("regime_window_candles_out_of_range")


def _validate_candidate_stride(candidate_stride_candles: int) -> None:
    if isinstance(candidate_stride_candles, bool) or not isinstance(
        candidate_stride_candles, int
    ):
        raise ValueError("candidate_stride_candles_must_be_integer")
    if candidate_stride_candles < 1:
        raise ValueError("candidate_stride_candles_must_be_positive")


def _validate_volatility_quantile(minimum_volatility_quantile: float) -> None:
    if (
        isinstance(minimum_volatility_quantile, bool)
        or not math.isfinite(minimum_volatility_quantile)
        or not 0.0 <= minimum_volatility_quantile < 1.0
    ):
        raise ValueError("volatility_quantile_out_of_range")


def _validate_training_market_data(training_data: Float64Array) -> None:
    if not np.isfinite(training_data).all():
        raise ValueError("market_data_contains_non_finite_values")
    if not np.all(np.diff(training_data[:, TIMESTAMP_COLUMN]) > 0.0):
        raise ValueError("market_data_timestamps_must_increase")
    if np.any(training_data[:, CLOSE_COLUMN] <= 0.0):
        raise ValueError("close_prices_must_be_positive")


def _candidate_window_starts(
    data_length: int, window_candles: int, candidate_stride_candles: int
) -> NDArray[np.int64]:
    max_start = data_length - window_candles
    starts = np.arange(0, max_start + 1, candidate_stride_candles, dtype=np.int64)
    if int(starts[-1]) != max_start:
        starts = np.append(starts, np.int64(max_start))
    return starts


def _candidate_window_metrics(
    close: NDArray[np.float64],
    starts: NDArray[np.int64],
    window_candles: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    log_close = np.log(close)
    log_returns = np.diff(log_close)
    returns_per_window = window_candles - 1
    squared_return_prefix = np.concatenate(
        (np.asarray([0.0]), np.cumsum(np.square(log_returns), dtype=np.float64))
    )
    squared_return_sum = (
        squared_return_prefix[starts + returns_per_window]
        - squared_return_prefix[starts]
    )
    realized_volatility = np.sqrt(
        np.maximum(squared_return_sum, 0.0) / returns_per_window
    )
    signed_log_return = log_close[starts + returns_per_window] - log_close[starts]
    return realized_volatility, signed_log_return


def _select_candidate_index(
    starts: NDArray[np.int64],
    realized_volatility: NDArray[np.float64],
    signed_log_return: NDArray[np.float64],
    minimum_volatility_quantile: float,
) -> tuple[int, float, NDArray[np.intp]]:
    absolute_log_return = np.abs(signed_log_return)
    volatility_threshold = float(
        np.quantile(realized_volatility, minimum_volatility_quantile)
    )
    eligible = np.flatnonzero(realized_volatility >= volatility_threshold)
    ranking = np.lexsort(
        (
            -starts[eligible],
            realized_volatility[eligible],
            absolute_log_return[eligible],
        )
    )
    return int(eligible[int(ranking[-1])]), volatility_threshold, eligible


def _selected_window_metadata(
    training_data: Float64Array,
    starts: NDArray[np.int64],
    realized_volatility: NDArray[np.float64],
    signed_log_return: NDArray[np.float64],
    selected_candidate: int,
    volatility_threshold: float,
    eligible: NDArray[np.intp],
    window_candles: int,
    candidate_stride_candles: int,
    minimum_volatility_quantile: float,
) -> dict[str, int | float | str]:
    start_index = int(starts[selected_candidate])
    end_index_exclusive = start_index + window_candles
    selected_log_return = float(signed_log_return[selected_candidate])
    net_return_pct = float(np.expm1(selected_log_return) * 100.0)
    absolute_return_pct = float(np.expm1(abs(selected_log_return)) * 100.0)
    selected_volatility = float(realized_volatility[selected_candidate])
    if not all(
        math.isfinite(value)
        for value in (
            net_return_pct,
            absolute_return_pct,
            selected_volatility,
            volatility_threshold,
        )
    ):
        raise ValueError("regime_selection_metrics_non_finite")

    close = training_data[:, CLOSE_COLUMN]
    return {
        "selection_method": (
            "top_quartile_realized_volatility_then_max_absolute_log_return"
        ),
        "selection_split": "training_only",
        "window_candles": window_candles,
        "candidate_stride_candles": candidate_stride_candles,
        "minimum_volatility_quantile": float(minimum_volatility_quantile),
        "candidate_window_count": int(len(starts)),
        "high_volatility_candidate_count": int(len(eligible)),
        "volatility_threshold_per_candle": volatility_threshold,
        "selected_realized_volatility_per_candle": selected_volatility,
        "selected_absolute_log_return": float(abs(selected_log_return)),
        "selected_net_return_pct": net_return_pct,
        "selected_absolute_return_pct": absolute_return_pct,
        "selected_start_index": start_index,
        "selected_end_index_exclusive": end_index_exclusive,
        "selected_start_timestamp_ms": int(
            training_data[start_index, TIMESTAMP_COLUMN]
        ),
        "selected_end_timestamp_ms": int(
            training_data[end_index_exclusive - 1, TIMESTAMP_COLUMN]
        ),
        "selected_start_close": float(close[start_index]),
        "selected_end_close": float(close[end_index_exclusive - 1]),
    }


def select_high_volatility_window(
    training_data: Float64Array,
    window_candles: int = DEFAULT_REGIME_WINDOW_CANDLES,
    candidate_stride_candles: int = DEFAULT_CANDIDATE_STRIDE_CANDLES,
    minimum_volatility_quantile: float = DEFAULT_VOLATILITY_QUANTILE,
) -> tuple[Float64Array, dict[str, int | float | str]]:
    """Select a high-volatility window with the strongest directional return.

    Candidate windows are ranked only after filtering to the requested upper
    quantile of realized one-minute volatility. Ties prefer higher volatility,
    then the earliest window. Callers must pass only the training split.
    """
    _validate_window_shape(training_data, window_candles)
    _validate_candidate_stride(candidate_stride_candles)
    _validate_volatility_quantile(minimum_volatility_quantile)
    _validate_training_market_data(training_data)
    starts = _candidate_window_starts(
        len(training_data), window_candles, candidate_stride_candles
    )
    realized_volatility, signed_log_return = _candidate_window_metrics(
        training_data[:, CLOSE_COLUMN], starts, window_candles
    )
    selected_candidate, volatility_threshold, eligible = _select_candidate_index(
        starts,
        realized_volatility,
        signed_log_return,
        minimum_volatility_quantile,
    )
    start_index = int(starts[selected_candidate])
    end_index_exclusive = start_index + window_candles
    selected = np.ascontiguousarray(
        training_data[start_index:end_index_exclusive], dtype=np.float64
    )
    metadata = _selected_window_metadata(
        training_data,
        starts,
        realized_volatility,
        signed_log_return,
        selected_candidate,
        volatility_threshold,
        eligible,
        window_candles,
        candidate_stride_candles,
        minimum_volatility_quantile,
    )
    return selected, metadata
