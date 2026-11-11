from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.deap.engine import (
    ALLOWED_TIMEFRAMES_MINUTES,
    BAR_INTERVAL_MS,
    MAX_DRAWDOWN,
    MAX_TREE_HEIGHT,
    MAX_TREE_NODES,
    MIN_TRADES_PER_YEAR,
    BooleanVector,
    MarketArrays,
    NumericVector,
    SimulationMetrics,
    _fitness_from_metrics,
    _oos_economic_checks,
    _numeric,
    add,
    build_causal_features,
    build_primitive_set,
    div,
    gt,
    logical_and,
    logical_or,
    lt,
    mul,
    simulate_segment,
    sub,
)


def test_timeframe_allowlist_and_tree_limits_are_exact() -> None:
    assert ALLOWED_TIMEFRAMES_MINUTES == (15, 30, 60, 240)
    assert 1440 not in ALLOWED_TIMEFRAMES_MINUTES
    assert 10080 not in ALLOWED_TIMEFRAMES_MINUTES
    assert MAX_TREE_HEIGHT == 5
    assert MAX_TREE_NODES == 31


def test_numeric_primitives_are_finite_bounded_and_vectorized() -> None:
    left = _numeric(np.asarray([1.0, -2.0, 0.5]))
    right = _numeric(np.asarray([2.0, 0.0, 0.25]))
    assert isinstance(add(left, right), NumericVector)
    assert np.allclose(add(left, right), [3.0, -2.0, 0.75])
    assert np.allclose(sub(left, right), [-1.0, -2.0, 0.25])
    assert np.allclose(mul(left, right), [2.0, 0.0, 0.125])
    quotient = div(left, right)
    assert np.isfinite(quotient).all()
    assert np.allclose(quotient, [0.5, 0.0, 2.0])


def test_boolean_primitives_return_boolean_vectors() -> None:
    left = _numeric(np.asarray([1.0, -1.0, 0.5]))
    right = _numeric(np.asarray([0.0, 0.0, 1.0]))
    greater = gt(left, right)
    lesser = lt(left, right)
    assert isinstance(greater, BooleanVector)
    assert isinstance(lesser, BooleanVector)
    assert np.array_equal(greater, [True, False, False])
    assert np.array_equal(lesser, [False, True, True])
    assert np.array_equal(logical_and(greater, lesser), [False, False, False])
    assert np.array_equal(logical_or(greater, lesser), [True, True, True])


def test_primitive_set_is_typed_with_boolean_buy_output() -> None:
    primitive_set = build_primitive_set()
    assert primitive_set.ret is BooleanVector
    assert len(primitive_set.arguments) == 7
    names = {primitive.name for primitive in primitive_set.primitives[BooleanVector]}
    assert {"gt", "lt", "and_", "or_"} <= names


def synthetic_ohlcv(rows: int = 2_400) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    close = 100.0 + 0.05 * index + 2.0 * np.sin(index / 8.0)
    open_price = close * (1.0 - 0.0005 * np.sin(index / 5.0))
    high = np.maximum(open_price, close) * 1.002
    low = np.minimum(open_price, close) * 0.998
    volume = 100.0 + 10.0 * np.cos(index / 7.0)
    dates = pd.date_range(
        "2022-01-01",
        periods=rows,
        freq="h",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_features_are_bounded_finite_and_strictly_causal() -> None:
    data = synthetic_ohlcv()
    features = build_causal_features(data)
    selected = features.values.loc[features.valid_mask]
    assert features.valid_mask.dtype == np.bool_
    assert np.isfinite(selected.to_numpy(dtype=np.float64)).all()
    assert (selected.abs().to_numpy() <= 1.0000001).all()
    assert ((selected["ATR_14"] >= 0.0) & (selected["ATR_14"] <= 1.0)).all()

    changed_future = data.copy()
    changed_future.loc[200:, "close"] *= 1.25
    changed_future.loc[200:, "high"] *= 1.25
    changed_future.loc[200:, "low"] *= 1.25
    changed_future.loc[200:, "open"] *= 1.25
    changed = build_causal_features(changed_future)
    assert np.allclose(
        features.values.iloc[:200].to_numpy(dtype=np.float64),
        changed.values.iloc[:200].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def make_market(
    closes: tuple[float, ...],
    lows: tuple[float, ...] | None = None,
) -> MarketArrays:
    close = np.asarray(closes, dtype=np.float64)
    opening = close.copy()
    high = close * 1.001
    low = close * 0.999
    if lows is not None:
        low = np.asarray(lows, dtype=np.float64)
    timestamps = (
        1_700_000_000_000
        + np.arange(len(close), dtype=np.int64) * BAR_INTERVAL_MS
        + BAR_INTERVAL_MS
        - 1
    )
    return MarketArrays(
        open=opening,
        high=high,
        low=low,
        close=close,
        close_time_ms=timestamps,
    )


def test_simulator_charges_two_sides_and_exits_after_signal_off() -> None:
    market = make_market((100.0, 100.0, 100.0, 100.0, 100.0, 100.0))
    signal = np.asarray(
        [True, True, False, False, False, False],
        dtype=np.bool_,
    )
    metrics = simulate_segment(market, signal)
    assert metrics.trade_count == 1
    assert metrics.net_profit_factor == 0.0
    assert metrics.win_rate == 0.0
    assert metrics.net_return == pytest.approx(-0.004987531172, abs=1e-8)
    assert metrics.total_fees_bps_of_initial_equity > 0.0


def test_simulator_marks_intrabar_low_for_conservative_drawdown() -> None:
    market = make_market(
        (100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        (99.9, 99.9, 70.0, 99.9, 99.9, 99.9),
    )
    signal = np.ones(6, dtype=np.bool_)
    metrics = simulate_segment(market, signal)
    assert metrics.trade_count == 1
    assert metrics.max_drawdown >= 0.29


def test_simulator_does_not_fabricate_pf_for_no_trades() -> None:
    market = make_market((100.0, 101.0, 99.0, 102.0))
    signal = np.zeros(4, dtype=np.bool_)
    metrics = simulate_segment(market, signal)
    assert metrics.trade_count == 0
    assert metrics.net_profit_factor is None
    assert metrics.win_rate is None
    assert metrics.net_return == pytest.approx(0.0)


def test_fitness_severely_penalizes_weak_economic_metrics() -> None:
    weak = SimulationMetrics(
        trade_count=80,
        net_profit_factor=0.7,
        win_rate=0.35,
        max_drawdown=0.40,
        net_return=-0.20,
        gross_profit=0.1,
        gross_loss=0.14,
        trades_per_year=80.0,
        duration_years=1.0,
        average_trade_return=-0.001,
        total_fees_bps_of_initial_equity=50.0,
        max_holding_bars=48,
        trade_returns=(),
    )
    strong = SimulationMetrics(
        trade_count=350,
        net_profit_factor=1.8,
        win_rate=0.58,
        max_drawdown=0.10,
        net_return=0.30,
        gross_profit=0.9,
        gross_loss=0.5,
        trades_per_year=300.0,
        duration_years=1.1,
        average_trade_return=0.001,
        total_fees_bps_of_initial_equity=50.0,
        max_holding_bars=48,
        trade_returns=(),
    )
    assert _fitness_from_metrics(
        weak,
        tree_nodes=5,
        tree_height=2,
    ) < _fitness_from_metrics(
        strong,
        tree_nodes=5,
        tree_height=2,
    )
    assert MAX_DRAWDOWN == 0.15
    assert MIN_TRADES_PER_YEAR == 200.0


def test_oos_economic_gates_match_owner_directive_exactly() -> None:
    candidate = SimulationMetrics(
        trade_count=240,
        net_profit_factor=1.1,
        win_rate=0.55,
        max_drawdown=0.149,
        net_return=0.12,
        gross_profit=0.22,
        gross_loss=0.20,
        trades_per_year=200.01,
        duration_years=1.2,
        average_trade_return=0.001,
        total_fees_bps_of_initial_equity=1200.0,
        max_holding_bars=48,
        trade_returns=(),
    )
    accepted = _oos_economic_checks(candidate)
    assert accepted["oos_net_profit_factor_at_least_1_1"] is True
    assert accepted["oos_max_drawdown_below_15_percent"] is True
    assert accepted["oos_trade_frequency_over_200_per_year"] is True

    assert (
        _oos_economic_checks(replace(candidate, net_profit_factor=1.0999))[
            "oos_net_profit_factor_at_least_1_1"
        ]
        is False
    )
    assert (
        _oos_economic_checks(replace(candidate, max_drawdown=0.15))[
            "oos_max_drawdown_below_15_percent"
        ]
        is False
    )
    assert (
        _oos_economic_checks(replace(candidate, trades_per_year=200.0))[
            "oos_trade_frequency_over_200_per_year"
        ]
        is False
    )


def test_simulator_uses_pre_segment_signal_as_episode_context() -> None:
    market = make_market((100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0))
    signal = np.asarray(
        [True, False, True, True, False, False, False],
        dtype=np.bool_,
    )
    metrics = simulate_segment(
        market,
        signal,
        initial_previous_signal=True,
    )
    assert metrics.trade_count == 1
