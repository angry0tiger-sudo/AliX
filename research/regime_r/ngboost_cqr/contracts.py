"""Pure, mock-only data and probabilistic forecast contracts for Regime R."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Literal, Mapping

HORIZON_BARS: Final[dict[str, int]] = {"1h": 1, "4h": 4, "24h": 24}
SUPPORTED_HORIZONS: Final[tuple[str, ...]] = ("1h", "4h", "24h")
COST_SCENARIOS_BPS: Final[tuple[int, int]] = (50, 90)
ForecastOutputSemantics = Literal["distributional", "quantile_summary"]
CANDIDATES_PER_HORIZON: Final[int] = 10
N_BUDGET: Final[int] = len(SUPPORTED_HORIZONS) * CANDIDATES_PER_HORIZON
FEATURE_NAMES: Final[tuple[str, ...]] = (
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
TRIAL_IDS: Final[tuple[str, ...]] = tuple(
    f"NOBITEX-R1-NGBOOST-{horizon}-{candidate:02d}"
    for horizon in SUPPORTED_HORIZONS
    for candidate in range(1, CANDIDATES_PER_HORIZON + 1)
)


class ContractViolation(ValueError):
    """Raised when a mock data/model payload violates a frozen contract."""


@dataclass(frozen=True, slots=True)
class TargetWindow:
    """One executable-reference target window indexed from a closed decision bar."""

    horizon: str
    decision_index: int
    entry_index: int
    exit_index: int
    entry_open: float
    exit_close: float


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """A feature row plus the latest source index used by each feature."""

    decision_index: int
    values: Mapping[str, float]
    available_through_index: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class DistributionForecast:
    """Generic forecast envelope with explicit distributional/quantile-only semantics.

    Quantile-only models must leave mean and dispersion unset. They must not
    relabel a median as a mean or a quantile spread as a standard deviation.
    """

    horizon: str
    mean: float | None
    dispersion: float | None
    quantile_levels: tuple[float, float, float]
    quantile_values: tuple[float, float, float]
    output_semantics: ForecastOutputSemantics = "distributional"


@dataclass(frozen=True, slots=True)
class ConformalInterval:
    """CQR-adjusted interval; lower/upper corrections may be signed."""

    lower_bound: float
    upper_bound: float
    nominal_coverage: float
    lower_correction: float
    upper_correction: float
    calibration_size: int


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isfinite(float(value))


def _is_non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_trial_id(trial_id: str) -> None:
    """Require a canonical ID from the preregistered 30-slot family."""
    if trial_id not in TRIAL_IDS:
        raise ContractViolation("trial_id_not_in_preregistered_budget")


def validate_target_window(window: TargetWindow) -> None:
    """Require next-bar-open entry and horizon-close exit without look-ahead."""
    if window.horizon not in HORIZON_BARS:
        raise ContractViolation("target_horizon_unsupported")
    indices = (window.decision_index, window.entry_index, window.exit_index)
    if not all(_is_non_negative_integer(index) for index in indices):
        raise ContractViolation("target_indices_invalid")
    expected_entry = window.decision_index + 1
    expected_exit = window.decision_index + HORIZON_BARS[window.horizon]
    if window.entry_index != expected_entry:
        raise ContractViolation("target_entry_must_be_next_bar_open")
    if window.exit_index != expected_exit:
        raise ContractViolation("target_exit_must_match_horizon")
    if not _is_finite_number(window.entry_open) or not _is_finite_number(
        window.exit_close
    ):
        raise ContractViolation("target_prices_must_be_finite")
    if window.entry_open <= 0 or window.exit_close <= 0:
        raise ContractViolation("target_prices_must_be_positive")


def target_gross_return(window: TargetWindow) -> float:
    """Compute simple gross return after validating the temporal contract."""
    validate_target_window(window)
    return window.exit_close / window.entry_open - 1.0


def validate_feature_snapshot(snapshot: FeatureSnapshot) -> None:
    """Reject missing, extra, non-finite or future-derived feature values."""
    if not _is_non_negative_integer(snapshot.decision_index):
        raise ContractViolation("feature_decision_index_invalid")
    expected = set(FEATURE_NAMES)
    if set(snapshot.values) != expected:
        raise ContractViolation("feature_schema_mismatch")
    if set(snapshot.available_through_index) != expected:
        raise ContractViolation("feature_asof_schema_mismatch")
    for name in FEATURE_NAMES:
        value = snapshot.values[name]
        asof_index = snapshot.available_through_index[name]
        if not _is_finite_number(value):
            raise ContractViolation("feature_value_must_be_finite")
        if not _is_non_negative_integer(asof_index):
            raise ContractViolation("feature_asof_index_invalid")
        if asof_index > snapshot.decision_index:
            raise ContractViolation("feature_lookahead_detected")


def _validate_forecast_horizon(forecast: DistributionForecast) -> None:
    """Require a supported forecast horizon."""
    if forecast.horizon not in HORIZON_BARS:
        raise ContractViolation("forecast_horizon_unsupported")


def _validate_forecast_mean(forecast: DistributionForecast) -> None:
    """Require a finite forecast location estimate."""
    if not _is_finite_number(forecast.mean):
        raise ContractViolation("forecast_mean_must_be_finite")


def _validate_forecast_dispersion(forecast: DistributionForecast) -> None:
    """Require a finite, strictly positive forecast scale."""
    if (
        forecast.dispersion is None
        or not _is_finite_number(forecast.dispersion)
        or forecast.dispersion <= 0
    ):
        raise ContractViolation("forecast_dispersion_must_be_positive")


def _validate_forecast_quantile_shape(forecast: DistributionForecast) -> None:
    """Require exactly three quantile levels and values."""
    if len(forecast.quantile_levels) != 3 or len(forecast.quantile_values) != 3:
        raise ContractViolation("forecast_requires_three_quantiles")


def _validate_forecast_quantile_levels(forecast: DistributionForecast) -> None:
    """Require finite, ordered quantile levels with a median at one half."""
    lower_level, median_level, upper_level = forecast.quantile_levels
    if not all(_is_finite_number(level) for level in forecast.quantile_levels):
        raise ContractViolation("forecast_quantile_levels_must_be_finite")
    if not (0 < lower_level < median_level < upper_level < 1):
        raise ContractViolation("forecast_quantile_levels_not_ordered")
    if median_level != 0.5:
        raise ContractViolation("forecast_median_quantile_must_be_half")


def _validate_forecast_quantile_values(forecast: DistributionForecast) -> None:
    """Require finite and monotonically increasing quantile predictions."""
    if not all(_is_finite_number(value) for value in forecast.quantile_values):
        raise ContractViolation("forecast_quantiles_must_be_finite")
    lower_value, median_value, upper_value = forecast.quantile_values
    if not lower_value <= median_value <= upper_value:
        raise ContractViolation("forecast_quantile_values_not_monotonic")


def _validate_forecast_output_semantics(forecast: DistributionForecast) -> None:
    """Validate the statistical meaning carried by the generic forecast envelope."""
    if forecast.output_semantics == "distributional":
        _validate_forecast_mean(forecast)
        _validate_forecast_dispersion(forecast)
    elif forecast.output_semantics == "quantile_summary":
        if forecast.mean is not None or forecast.dispersion is not None:
            raise ContractViolation(
                "quantile_summary_must_not_claim_mean_or_dispersion"
            )
    else:
        raise ContractViolation("forecast_output_semantics_unsupported")


def validate_distribution_forecast(forecast: DistributionForecast) -> None:
    """Validate forecast semantics, finite quantiles and their monotonic order."""
    _validate_forecast_horizon(forecast)
    _validate_forecast_output_semantics(forecast)
    _validate_forecast_quantile_shape(forecast)
    _validate_forecast_quantile_levels(forecast)
    _validate_forecast_quantile_values(forecast)


def make_cqr_interval(
    lower_quantile: float,
    upper_quantile: float,
    lower_correction: float,
    upper_correction: float,
    nominal_coverage: float,
    calibration_size: int,
) -> ConformalInterval:
    """Apply separate lower/upper CQR conformity corrections to base quantiles."""
    values = (
        lower_quantile,
        upper_quantile,
        lower_correction,
        upper_correction,
        nominal_coverage,
    )
    if not all(_is_finite_number(value) for value in values):
        raise ContractViolation("cqr_inputs_must_be_finite")
    if lower_quantile > upper_quantile:
        raise ContractViolation("cqr_base_interval_inverted")
    if not 0 < nominal_coverage < 1:
        raise ContractViolation("cqr_nominal_coverage_invalid")
    if not _is_non_negative_integer(calibration_size) or calibration_size < 1:
        raise ContractViolation("cqr_calibration_size_invalid")
    calibrated_lower = lower_quantile - lower_correction
    calibrated_upper = upper_quantile + upper_correction
    if calibrated_lower > calibrated_upper:
        raise ContractViolation("cqr_calibrated_interval_inverted")
    return ConformalInterval(
        lower_bound=calibrated_lower,
        upper_bound=calibrated_upper,
        nominal_coverage=nominal_coverage,
        lower_correction=lower_correction,
        upper_correction=upper_correction,
        calibration_size=calibration_size,
    )


def validate_conformal_interval(interval: ConformalInterval) -> None:
    """Reject an invalid conformal interval before it reaches a cost decision."""
    values = (
        interval.lower_bound,
        interval.upper_bound,
        interval.nominal_coverage,
        interval.lower_correction,
        interval.upper_correction,
    )
    if not all(_is_finite_number(value) for value in values):
        raise ContractViolation("conformal_interval_values_must_be_finite")
    if interval.lower_bound > interval.upper_bound:
        raise ContractViolation("conformal_interval_bounds_inverted")
    if not 0 < interval.nominal_coverage < 1:
        raise ContractViolation("conformal_interval_coverage_invalid")
    if (
        not _is_non_negative_integer(interval.calibration_size)
        or interval.calibration_size < 1
    ):
        raise ContractViolation("conformal_interval_calibration_size_invalid")


def passes_cost_gate(calibrated_lower_bound: float, cost_bps: int) -> bool:
    """Apply the Owner's strict lower-bound > round-trip cost gate."""
    if cost_bps not in COST_SCENARIOS_BPS:
        raise ContractViolation("cost_scenario_not_preregistered")
    if not _is_finite_number(calibrated_lower_bound):
        raise ContractViolation("lower_bound_must_be_finite")
    return calibrated_lower_bound > cost_bps / 10_000.0
