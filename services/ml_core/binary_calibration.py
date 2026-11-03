"""Leakage-safe dynamic BUY probability threshold calibration."""

from __future__ import annotations

from math import ceil, isfinite
from typing import Any

import numpy as np

DEFAULT_TARGET_BUY_FRACTION = 0.10


def validate_target_buy_fraction(value: float) -> float:
    """Validate the requested share of highest-probability calibration rows."""
    if (
        isinstance(value, bool)
        or not isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError("binary_calibration_target_fraction_invalid")
    return float(value)


def select_dynamic_buy_threshold(
    probabilities: Any,
    *,
    target_buy_fraction: float = DEFAULT_TARGET_BUY_FRACTION,
) -> dict[str, float | int | str]:
    """Select a scalar cutoff for the top probability fraction.

    Ties at the cutoff are retained by the >= comparison. The achieved
    calibration fraction and tie expansion are reported explicitly.
    """
    fraction = validate_target_buy_fraction(target_buy_fraction)
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not bool(np.isfinite(values).all())
        or bool(np.any(values < 0.0))
        or bool(np.any(values > 1.0))
    ):
        raise ValueError("binary_calibration_probabilities_invalid")

    target_count = max(1, int(ceil(values.size * fraction)))
    cutoff_position = values.size - target_count
    threshold = float(np.partition(values, cutoff_position)[cutoff_position])
    selected_count = int(np.count_nonzero(values >= threshold))
    tie_count = int(np.count_nonzero(values == threshold))
    return {
        "method": "TOP_PERCENTILE_ORDER_STATISTIC",
        "comparison": "GREATER_THAN_OR_EQUAL",
        "threshold": threshold,
        "target_buy_fraction": fraction,
        "target_percentile": 100.0 * (1.0 - fraction),
        "calibration_rows": int(values.size),
        "target_selected_rows": target_count,
        "selected_calibration_rows": selected_count,
        "selected_calibration_fraction": selected_count / float(values.size),
        "threshold_tie_rows": tie_count,
        "tie_expansion_rows": max(0, selected_count - target_count),
    }
