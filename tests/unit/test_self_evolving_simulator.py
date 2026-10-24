from __future__ import annotations

import math

import pytest

from services.ai_core.self_evolving_simulator import (
    SimBar,
    SimulationConfig,
    simulate,
    validate_bars,
)


def _bars() -> list[SimBar]:
    return [
        SimBar(0, 59_999, 100.0, 101.0, 99.0, 100.0, 10.0),
        SimBar(60_000, 119_999, 100.0, 112.0, 99.0, 110.0, 12.0),
        SimBar(120_000, 179_999, 110.0, 111.0, 109.0, 110.0, 8.0),
    ]


def test_round_trip_costs_are_applied_per_executed_side() -> None:
    result = simulate(_bars(), [0.5, 0.0, 0.0])
    assert [fill.side for fill in result.fills] == ["BUY", "SELL"]
    assert result.fills[0].timestamp_ms == 60_000
    assert result.fills[0].execution_price == pytest.approx(100.2)
    assert result.fills[1].execution_price == pytest.approx(109.78)
    assert result.total_fees == pytest.approx(26.2475)
    assert result.total_slippage == pytest.approx(21.0)
    assert result.final_equity == pytest.approx(10_452.7525)
    assert result.net_return == pytest.approx(0.04527525)
    assert result.max_drawdown >= 0.0


def test_signal_is_not_filled_until_the_next_bar() -> None:
    result = simulate(_bars(), [0.0, 0.0, 0.5])
    assert result.fills == ()
    assert result.final_equity == pytest.approx(10_000.0)


def test_open_position_is_liquidated_at_final_close_with_costs() -> None:
    result = simulate(_bars(), [0.0, 0.5, 0.5])
    assert [fill.side for fill in result.fills] == ["BUY", "SELL"]
    assert result.fills[-1].timestamp_ms == 179_999
    assert result.equity_curve[-1].quantity == 0.0
    assert result.final_equity < result.equity_curve[-2].equity


def test_default_costs_and_exposure_envelope_are_explicit() -> None:
    config = SimulationConfig()
    assert config.fee_bps_per_side == 25.0
    assert config.slippage_bps_per_side == 20.0
    assert config.max_abs_exposure == 0.95


@pytest.mark.parametrize(
    "bars",
    [
        [],
        [_bars()[0], _bars()[0]],
        [SimBar(0, 59_999, 100.0, 99.0, 99.5, 100.0, 1.0)],
        [SimBar(0, 60_000, 100.0, 101.0, 99.0, 100.0, math.nan)],
        [SimBar(0, 59_999, -1.0, 1.0, 0.5, 1.0, 1.0)],
        [
            SimBar(0, 60_000, 100.0, 101.0, 99.0, 100.0, 1.0),
            SimBar(60_000, 119_999, 100.0, 101.0, 99.0, 100.0, 1.0),
        ],
    ],
)
def test_invalid_bar_sets_fail_closed(bars: list[SimBar]) -> None:
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_mismatched_non_finite_and_negative_exposures_fail_closed() -> None:
    with pytest.raises(ValueError, match="count_mismatch"):
        simulate(_bars(), [0.0])
    with pytest.raises(ValueError, match="must_be_finite"):
        simulate(_bars(), [0.0, math.nan, 0.0])
    with pytest.raises(ValueError, match="outside_spot_envelope"):
        simulate(_bars(), [-0.1, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside_spot_envelope"):
        simulate(_bars(), [1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fee_bps_per_side": -1.0},
        {"fee_bps_per_side": 10_000.0},
        {"slippage_bps_per_side": 10_000.0},
        {"initial_equity": 0.0},
        {"max_abs_exposure": 1.1},
        {"initial_equity": math.inf},
    ],
)
def test_invalid_simulation_config_fails_closed(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)
