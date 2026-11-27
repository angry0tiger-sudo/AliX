from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from scripts.run_m6_triple_barrier_3class import (
    CLASS_NAMES,
    FEATURE_NAMES,
    Candle,
    Event,
    _resolve_event,
    aggregate_four_hour,
    build_events,
    build_features,
    build_fold_plans,
    build_model,
    is_long_signal,
    is_stable_fold,
    load_preregistration,
    net_profit_factor,
    simulate_portfolio,
)


def make_hourly(count: int = 60) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp=start + timedelta(hours=index),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=10.0 + index,
            base_start_index=index,
            base_end_index=index,
        )
        for index in range(count)
    ]


def replace_candle(candles: list[Candle], index: int, **changes: Any) -> list[Candle]:
    updated = list(candles)
    values = {
        "timestamp": updated[index].timestamp,
        "open": updated[index].open,
        "high": updated[index].high,
        "low": updated[index].low,
        "close": updated[index].close,
        "volume": updated[index].volume,
        "base_start_index": updated[index].base_start_index,
        "base_end_index": updated[index].base_end_index,
    }
    values.update(changes)
    updated[index] = Candle(**values)
    return updated


def test_feature_builder_uses_exact_eight_ohlcv_features() -> None:
    candles = make_hourly(80)
    features, atr_fraction = build_features(candles)

    assert features.shape == (80, 8)
    assert tuple(FEATURE_NAMES) == (
        "log_return_1",
        "log_return_4",
        "log_return_24",
        "realized_volatility_6",
        "realized_volatility_24",
        "range_fraction_1",
        "atr_fraction_14",
        "log_volume_change_1",
    )
    assert np.isnan(features[23]).any()
    assert np.isfinite(features[24]).all()
    assert np.isfinite(atr_fraction[24])


def test_four_hour_aggregation_preserves_ohlcv_and_base_hour_mapping() -> None:
    hourly = make_hourly(9)
    hourly = replace_candle(hourly, 1, high=105.0, low=98.0, close=103.0)
    hourly = replace_candle(hourly, 2, high=104.0, low=97.0, close=101.0, volume=20.0)
    four_hour = aggregate_four_hour(hourly)

    assert len(four_hour) == 2
    assert four_hour[0].open == 100.0
    assert four_hour[0].high == 105.0
    assert four_hour[0].low == 97.0
    assert four_hour[0].close == 100.0
    assert four_hour[0].volume == pytest.approx(sum(item.volume for item in hourly[:4]))
    assert four_hour[0].base_start_index == 0
    assert four_hour[0].base_end_index == 3
    assert four_hour[1].base_start_index == 4
    assert four_hour[1].base_end_index == 7


def test_three_class_target_resolves_tp_first_sl_first_and_timeout() -> None:
    baseline = make_hourly(30)
    tp_candles = replace_candle(baseline, 1, high=103.0)
    sl_candles = replace_candle(baseline, 1, low=97.0)

    tp = _resolve_event(tp_candles, 1, 0.01)
    sl = _resolve_event(sl_candles, 1, 0.01)
    timeout = _resolve_event(baseline, 1, 0.01)

    assert tp is not None and (tp.outcome, tp.label, tp.exit_price) == ("TP_FIRST", 0, 102.0)
    assert sl is not None and (sl.outcome, sl.label, sl.exit_price) == ("SL_FIRST", 1, 98.0)
    assert timeout is not None
    assert timeout.outcome == "VERTICAL_TIMEOUT"
    assert timeout.label == 2
    assert timeout.exit_base_index == 24
    assert timeout.max_end_base_index == 24
    assert CLASS_NAMES[timeout.label] == "VERTICAL_TIMEOUT"


def test_same_hour_dual_touch_is_censored_but_conservatively_exits_at_stop() -> None:
    candles = replace_candle(make_hourly(30), 1, high=103.0, low=97.0)

    event = _resolve_event(candles, 1, 0.01)

    assert event is not None
    assert event.outcome == "AMBIGUOUS"
    assert event.label is None
    assert event.exit_price == pytest.approx(event.stop_loss_price)


def test_opening_gap_uses_open_price_for_adverse_stop_fill() -> None:
    candles = replace_candle(make_hourly(30), 2, open=97.5, high=99.0, low=97.0)

    event = _resolve_event(candles, 1, 0.01)

    assert event is not None
    assert event.outcome == "SL_FIRST"
    assert event.exit_price == pytest.approx(97.5)


def test_incomplete_future_window_is_unavailable() -> None:
    event = _resolve_event(make_hourly(20), 1, 0.01)

    assert event is None


def test_four_hour_events_enter_at_next_four_hour_group_open() -> None:
    hourly = make_hourly(80)
    four_hour = aggregate_four_hour(hourly)
    atr = np.full(len(four_hour), 0.01, dtype=np.float64)

    events = build_events(hourly, four_hour, atr)

    assert events[0] is not None
    assert events[0].entry_base_index == 4
    assert events[0].max_end_base_index == 27


def test_fold_plans_purge_label_windows_and_apply_embargo() -> None:
    bars = [
        Candle(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4 * index),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
            base_start_index=4 * index, base_end_index=4 * index + 3,
        )
        for index in range(1000)
    ]
    events = [
        Event("TP_FIRST", 0, 4 * (index + 1), 4 * (index + 1) + 23,
              4 * (index + 1) + 1, 100.0, 102.0, 102.0, 98.0)
        for index in range(len(bars))
    ]
    eligible = list(range(len(bars) - 1))

    plans = build_fold_plans(eligible, bars, events, embargo_hours=24)

    assert len(plans) == 4
    for plan in plans:
        assert plan.train_indices
        assert plan.test_indices
        assert max(plan.train_indices) < min(plan.test_indices)
        assert all(
            events[index].max_end_base_index < plan.first_test_entry_base_index - 24
            for index in plan.train_indices
        )
        assert all(
            events[index].max_end_base_index <= plan.last_test_base_index
            for index in plan.test_indices
        )


def test_ambiguous_and_unavailable_events_are_not_training_labels() -> None:
    hourly = make_hourly(80)
    bars = hourly
    atr = np.full(len(bars), 0.01, dtype=np.float64)
    hourly = replace_candle(hourly, 1, high=103.0, low=97.0)
    events = build_events(hourly, bars, atr)

    assert events[0] is not None and events[0].outcome == "AMBIGUOUS"
    assert events[0].label is None
    assert events[-2] is None


def test_random_forest_configuration_is_single_fixed_multiclass_model() -> None:
    config = {
        "model": {
            "parameters": {
                "n_estimators": 300,
                "criterion": "gini",
                "max_depth": 5,
                "min_samples_split": 40,
                "min_samples_leaf": 20,
                "max_features": 0.8,
                "bootstrap": True,
                "max_samples": 0.8,
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": 1,
            }
        }
    }
    model = build_model(config)

    assert model.n_estimators == 300
    assert model.max_depth == 5
    assert model.min_samples_split == 40
    assert model.min_samples_leaf == 20
    assert model.max_features == 0.8
    assert model.class_weight == "balanced_subsample"
    assert model.random_state == 42
    assert model.n_jobs == 1


def test_tp_probability_threshold_is_strictly_greater_than_point_six() -> None:
    assert not is_long_signal(0.60)
    assert not is_long_signal(0.59)
    assert is_long_signal(0.600001)


def test_costed_portfolio_charges_round_trip_cost_once_and_marks_nav() -> None:
    hourly = make_hourly(40)
    event = Event("TP_FIRST", 0, 1, 24, 1, 100.0, 102.0, 102.0, 98.0)
    events: list[Event | None] = [event]

    result = simulate_portfolio(
        hourly,
        test_indices=[0],
        events=events,
        probabilities=[0.61],
        decision_interval_hours=1,
        cost=0.005,
        mark_indices=list(range(0, 30)),
    )

    expected_notional = 1.0 / 25.0
    assert result.metrics["completed_oos_trades"] == 1
    assert result.metrics["position_notional_fraction"] == pytest.approx(expected_notional)
    assert result.trades[0].pnl == pytest.approx(expected_notional * (0.02 - 0.005))
    assert result.pnl == pytest.approx(result.trades[0].pnl)
    assert result.nav[-1] == pytest.approx(1.0 + result.pnl)
    assert result.metrics["ambiguous_trades_conservatively_charged_as_sl"] == 0


def test_ambiguous_oos_trade_is_counted_and_charged_as_stop_loss() -> None:
    hourly = make_hourly(40)
    event = Event("AMBIGUOUS", None, 1, 24, 2, 100.0, 98.0, 102.0, 98.0)

    result = simulate_portfolio(
        hourly,
        test_indices=[0],
        events=[event],
        probabilities=[0.61],
        decision_interval_hours=4,
        cost=0.009,
        mark_indices=list(range(0, 30)),
    )

    assert result.metrics["completed_oos_trades"] == 1
    assert result.metrics["ambiguous_trades_conservatively_charged_as_sl"] == 1
    assert result.trades[0].pnl == pytest.approx((1.0 / 7.0) * (-0.02 - 0.009))


def test_profit_factor_is_undefined_without_negative_trades() -> None:
    event = Event("TP_FIRST", 0, 1, 24, 1, 100.0, 102.0, 102.0, 98.0)
    result = simulate_portfolio(
        make_hourly(40), [0], [event], [0.61], 1, 0.005, list(range(30))
    )
    assert net_profit_factor(result.trades) is None


def test_stable_fold_requires_twenty_trades_positive_pnl_and_pf_above_one() -> None:
    from scripts.run_m6_triple_barrier_3class import PortfolioOutcome, Trade

    trades = tuple(
        [Trade(i, i + 1, i + 1, 0.04, 0.001, "TP_FIRST") for i in range(20)]
    )
    stable = PortfolioOutcome(
        metrics={"completed_oos_trades": 20, "net_profit_factor": 2.0},
        nav=(1.0,), pnl=0.02, trades=trades
    )
    assert is_stable_fold(stable)

    unstable = PortfolioOutcome(
        metrics={"completed_oos_trades": 20, "net_profit_factor": 0.9},
        nav=(1.0,), pnl=-0.01, trades=trades
    )
    assert not is_stable_fold(unstable)


def test_preregistration_is_byte_locked_and_contains_two_trials() -> None:
    config, digest = load_preregistration()

    assert digest == "f94af9eb6f6003818beb88fcdbfbebe2ac2d7ac8a42281529db2781b44962e42"
    assert config["trial_family"]["logical_trial_count"] == 2
    assert config["scope"]["market"] == "BTCUSDT"
    assert config["acceptance_gates"]["minimum_stable_folds"] == 3
