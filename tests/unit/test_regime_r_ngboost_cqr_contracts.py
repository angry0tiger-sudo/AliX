"""Mock-only tests for Regime R NGBoost + CQR contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import pytest

from research.regime_r.ngboost_cqr.contracts import (
    COST_SCENARIOS_BPS,
    FEATURE_NAMES,
    HORIZON_BARS,
    N_BUDGET,
    TRIAL_IDS,
    ContractViolation,
    DistributionForecast,
    FeatureSnapshot,
    TargetWindow,
    make_cqr_interval,
    passes_cost_gate,
    target_gross_return,
    validate_distribution_forecast,
    validate_feature_snapshot,
    validate_target_window,
    validate_trial_id,
)


def _feature_snapshot(
    *, decision_index: int = 100, future_asof: int | None = None
) -> FeatureSnapshot:
    values = {name: 0.01 for name in FEATURE_NAMES}
    asof = {name: decision_index for name in FEATURE_NAMES}
    if future_asof is not None:
        asof[FEATURE_NAMES[0]] = future_asof
    return FeatureSnapshot(
        decision_index=decision_index,
        values=values,
        available_through_index=asof,
    )


def _forecast() -> DistributionForecast:
    return DistributionForecast(
        horizon="4h",
        mean=0.002,
        dispersion=0.01,
        quantile_levels=(0.1, 0.5, 0.9),
        quantile_values=(-0.01, 0.002, 0.02),
    )


def test_budget_has_exactly_thirty_unique_trial_ids() -> None:
    assert N_BUDGET == 30
    assert len(TRIAL_IDS) == N_BUDGET
    assert len(set(TRIAL_IDS)) == N_BUDGET
    for horizon in HORIZON_BARS:
        assert sum(f"-{horizon}-" in trial_id for trial_id in TRIAL_IDS) == 10


def test_trial_id_rejects_unregistered_candidate() -> None:
    validate_trial_id("NOBITEX-R1-NGBOOST-1h-01")
    with pytest.raises(ContractViolation, match="trial_id_not_in_preregistered_budget"):
        validate_trial_id("NOBITEX-R1-NGBOOST-1h-11")


@pytest.mark.parametrize(
    ("horizon", "exit_index"),
    (("1h", 101), ("4h", 104), ("24h", 124)),
)
def test_target_window_matches_next_open_and_horizon_close(
    horizon: str, exit_index: int
) -> None:
    window = TargetWindow(
        horizon=horizon,
        decision_index=100,
        entry_index=101,
        exit_index=exit_index,
        entry_open=100.0,
        exit_close=101.0,
    )
    validate_target_window(window)
    assert target_gross_return(window) == pytest.approx(0.01)


def test_target_window_rejects_unsupported_horizon_and_invalid_indices() -> None:
    with pytest.raises(ContractViolation, match="target_horizon_unsupported"):
        validate_target_window(TargetWindow("2h", 10, 11, 12, 100.0, 101.0))
    with pytest.raises(ContractViolation, match="target_indices_invalid"):
        validate_target_window(TargetWindow("1h", True, 11, 11, 100.0, 101.0))


def test_target_window_rejects_wrong_entry_and_exit_indices() -> None:
    with pytest.raises(ContractViolation, match="target_entry_must_be_next_bar_open"):
        validate_target_window(TargetWindow("1h", 10, 12, 11, 100.0, 101.0))
    with pytest.raises(ContractViolation, match="target_exit_must_match_horizon"):
        validate_target_window(TargetWindow("4h", 10, 11, 13, 100.0, 101.0))


def test_target_window_rejects_non_finite_and_non_positive_prices() -> None:
    with pytest.raises(ContractViolation, match="target_prices_must_be_finite"):
        validate_target_window(TargetWindow("1h", 10, 11, 11, 100.0, math.nan))
    with pytest.raises(ContractViolation, match="target_prices_must_be_positive"):
        validate_target_window(TargetWindow("1h", 10, 11, 11, 0.0, 101.0))


def test_feature_snapshot_accepts_finite_causal_features() -> None:
    validate_feature_snapshot(_feature_snapshot())


def test_feature_snapshot_rejects_invalid_decision_and_asof_indices() -> None:
    valid = _feature_snapshot()
    with pytest.raises(ContractViolation, match="feature_decision_index_invalid"):
        validate_feature_snapshot(
            FeatureSnapshot(-1, valid.values, valid.available_through_index)
        )
    invalid_asof = dict(valid.available_through_index)
    invalid_asof[FEATURE_NAMES[0]] = -1
    with pytest.raises(ContractViolation, match="feature_asof_index_invalid"):
        validate_feature_snapshot(
            FeatureSnapshot(valid.decision_index, valid.values, invalid_asof)
        )


def test_feature_snapshot_rejects_missing_and_extra_features() -> None:
    valid = _feature_snapshot()
    missing_values = dict(valid.values)
    missing_values.pop(FEATURE_NAMES[0])
    with pytest.raises(ContractViolation, match="feature_schema_mismatch"):
        validate_feature_snapshot(
            FeatureSnapshot(
                valid.decision_index, missing_values, valid.available_through_index
            )
        )
    extra_values = dict(valid.values)
    extra_values["unregistered_feature"] = 0.0
    with pytest.raises(ContractViolation, match="feature_schema_mismatch"):
        validate_feature_snapshot(
            FeatureSnapshot(
                valid.decision_index, extra_values, valid.available_through_index
            )
        )


def test_feature_snapshot_rejects_future_derived_and_non_finite_values() -> None:
    with pytest.raises(ContractViolation, match="feature_lookahead_detected"):
        validate_feature_snapshot(_feature_snapshot(future_asof=101))
    valid = _feature_snapshot()
    non_finite = dict(valid.values)
    non_finite[FEATURE_NAMES[0]] = math.inf
    with pytest.raises(ContractViolation, match="feature_value_must_be_finite"):
        validate_feature_snapshot(
            FeatureSnapshot(
                valid.decision_index, non_finite, valid.available_through_index
            )
        )
    invalid_type = dict(valid.values)
    invalid_type[FEATURE_NAMES[0]] = cast(float, "not-a-number")
    with pytest.raises(ContractViolation, match="feature_value_must_be_finite"):
        validate_feature_snapshot(
            FeatureSnapshot(
                valid.decision_index, invalid_type, valid.available_through_index
            )
        )


def test_feature_snapshot_rejects_asof_schema_mismatch() -> None:
    valid = _feature_snapshot()
    asof = dict(valid.available_through_index)
    asof.pop(FEATURE_NAMES[0])
    with pytest.raises(ContractViolation, match="feature_asof_schema_mismatch"):
        validate_feature_snapshot(
            FeatureSnapshot(valid.decision_index, valid.values, asof)
        )


def test_distribution_forecast_accepts_ordered_conditional_quantiles() -> None:
    validate_distribution_forecast(_forecast())


def test_distribution_forecast_rejects_unsupported_horizon_and_invalid_mean() -> None:
    valid = _forecast()
    with pytest.raises(ContractViolation, match="forecast_horizon_unsupported"):
        validate_distribution_forecast(replace(valid, horizon="2h"))
    with pytest.raises(ContractViolation, match="forecast_mean_must_be_finite"):
        validate_distribution_forecast(replace(valid, mean=math.nan))


def test_distribution_forecast_rejects_invalid_dispersion_and_quantile_shapes() -> None:
    valid = _forecast()
    with pytest.raises(ContractViolation, match="forecast_dispersion_must_be_positive"):
        validate_distribution_forecast(replace(valid, dispersion=0.0))
    with pytest.raises(ContractViolation, match="forecast_requires_three_quantiles"):
        validate_distribution_forecast(
            replace(valid, quantile_levels=cast(tuple[float, float, float], (0.1, 0.9)))
        )


def test_distribution_forecast_rejects_invalid_quantile_levels_and_values() -> None:
    valid = _forecast()
    with pytest.raises(ContractViolation, match="forecast_quantile_levels_must_be_finite"):
        validate_distribution_forecast(
            replace(valid, quantile_levels=(math.nan, 0.5, 0.9))
        )
    with pytest.raises(ContractViolation, match="forecast_quantile_levels_not_ordered"):
        validate_distribution_forecast(
            replace(valid, quantile_levels=(0.9, 0.5, 0.1))
        )
    with pytest.raises(ContractViolation, match="forecast_median_quantile_must_be_half"):
        validate_distribution_forecast(
            replace(valid, quantile_levels=(0.1, 0.4, 0.9))
        )
    with pytest.raises(ContractViolation, match="forecast_quantiles_must_be_finite"):
        validate_distribution_forecast(
            replace(valid, quantile_values=(math.nan, 0.0, 0.1))
        )
    with pytest.raises(ContractViolation, match="forecast_quantile_values_not_monotonic"):
        validate_distribution_forecast(
            replace(valid, quantile_values=(0.01, 0.0, -0.01))
        )


def test_cqr_interval_applies_separate_conformal_corrections() -> None:
    interval = make_cqr_interval(
        lower_quantile=-0.01,
        upper_quantile=0.02,
        lower_correction=0.002,
        upper_correction=0.003,
        nominal_coverage=0.9,
        calibration_size=50,
    )
    assert interval.lower_bound == pytest.approx(-0.012)
    assert interval.upper_bound == pytest.approx(0.023)
    assert interval.calibration_size == 50


def test_cqr_interval_rejects_invalid_inputs_and_coverage() -> None:
    with pytest.raises(ContractViolation, match="cqr_inputs_must_be_finite"):
        make_cqr_interval(-0.01, 0.02, math.nan, 0.0, 0.9, 50)
    with pytest.raises(ContractViolation, match="cqr_base_interval_inverted"):
        make_cqr_interval(0.02, -0.01, 0.0, 0.0, 0.9, 50)
    with pytest.raises(ContractViolation, match="cqr_nominal_coverage_invalid"):
        make_cqr_interval(-0.01, 0.02, 0.0, 0.0, 1.0, 50)
    with pytest.raises(ContractViolation, match="cqr_calibration_size_invalid"):
        make_cqr_interval(-0.01, 0.02, 0.0, 0.0, 0.9, 0)


def test_cqr_interval_rejects_inverted_calibrated_interval() -> None:
    with pytest.raises(ContractViolation, match="cqr_calibrated_interval_inverted"):
        make_cqr_interval(-0.001, 0.001, 0.01, 0.0, 0.9, 50)


@pytest.mark.parametrize(
    ("lower_bound", "cost_bps", "expected"),
    (
        (0.005, 50, False),
        (0.0051, 50, True),
        (0.009, 90, False),
        (0.0091, 90, True),
    ),
)
def test_cost_gate_is_strict_for_both_round_trip_scenarios(
    lower_bound: float, cost_bps: int, expected: bool
) -> None:
    assert passes_cost_gate(lower_bound, cost_bps) is expected


def test_cost_gate_rejects_unregistered_cost_and_non_finite_bound() -> None:
    with pytest.raises(ContractViolation, match="cost_scenario_not_preregistered"):
        passes_cost_gate(0.01, 75)
    with pytest.raises(ContractViolation, match="lower_bound_must_be_finite"):
        passes_cost_gate(math.nan, 50)


def test_cost_scenarios_are_the_owner_registered_pair() -> None:
    assert COST_SCENARIOS_BPS == (50, 90)
