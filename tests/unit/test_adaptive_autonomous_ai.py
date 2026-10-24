from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ai_core.adaptive_autonomous import (
    POLICY_SCHEMA,
    AdaptiveConfig,
    AdaptiveParameterState,
    analyze_post_trade_failure,
    build_multitimeframe_contexts,
    initial_parameter_state,
    load_adaptive_config,
    plan_dynamic_stop,
    select_adaptive_timeframe,
    size_adaptive_position,
    summarize_adaptive_metrics,
    update_intelligent_trailing_stop,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "ai_core"
    / "adaptive_autonomous.py"
)


def hourly_candles(rows: int = 1_440) -> DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 + index * 0.018 + 0.8 * np.sin(index / 7.0)
    opened = np.r_[close[0] - 0.05, close[:-1]]
    high = np.maximum(opened, close) + 0.35
    low = np.minimum(opened, close) - 0.35
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01T00:00:00Z", periods=rows, freq="1h"
            ),
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.zeros(rows),
        }
    )


def context_row(
    *, trend_strength: float = 0.8, volatility_score: float = 0.1
) -> dict[str, object]:
    row: dict[str, object] = {}
    for timeframe in ("1h", "4h", "1d"):
        row.update(
            {
                f"{timeframe}_close": 110.0,
                f"{timeframe}_ema_fast": 109.0,
                f"{timeframe}_ema_slow": 108.0,
                f"{timeframe}_atr": 2.0,
                f"{timeframe}_atr_ratio": 0.02,
                f"{timeframe}_trend_strength": trend_strength,
                f"{timeframe}_volatility_score": volatility_score,
                f"{timeframe}_macd_histogram": 1.0,
                f"{timeframe}_support": 100.0,
            }
        )
    return row


def closed_trade(
    *,
    pnl: float = -10.0,
    exit_reason: str = "trailing_stop",
    entry_regime: str = "trend",
    exit_regime: str = "trend",
    stop_atr: float = 1.0,
    favorable_atr: float = 1.2,
) -> dict[str, Any]:
    return {
        "net_pnl_quote": pnl,
        "exit_reason": exit_reason,
        "entry_regime": entry_regime,
        "exit_regime": exit_regime,
        "initial_stop_distance_atr": stop_atr,
        "max_favorable_excursion_atr": favorable_atr,
    }


def test_default_config_has_bounded_adaptive_risk() -> None:
    config = AdaptiveConfig()
    assert config.minimum_risk_fraction < config.maximum_risk_fraction
    assert config.maximum_risk_fraction < 0.01
    assert config.maximum_position_fraction <= 1.0
    assert initial_parameter_state(config).entry_confidence_floor == pytest.approx(0.55)


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_selection_confidence": 0.0},
        {"minimum_entry_trend_strength": 1.1},
        {"minimum_risk_fraction": 0.01, "maximum_risk_fraction": 0.005},
        {"maximum_position_fraction": 1.1},
        {"maximum_stop_distance_fraction": 1.0},
        {"support_buffer_atr": -0.1},
        {"initial_stop_atr_multiple": 5.0},
        {"initial_trailing_atr_multiple": 6.0},
        {"maximum_confidence_floor": 0.5},
        {"weak_trailing_atr_multiple": 9.0},
        {"stop_adaptation_step": 0.0},
        {"fee_bps_per_side": -1.0},
    ],
)
def test_config_rejects_unsafe_bounds(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AdaptiveConfig(**cast(Any, changes))


def test_policy_config_load_checks_schema_and_module_hash(tmp_path: Path) -> None:
    payload = {
        "schema_version": POLICY_SCHEMA,
        "revision": "test-v1",
        "code_sha256": hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        "parameters": asdict(AdaptiveConfig()),
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_adaptive_config(path, MODULE_PATH) == AdaptiveConfig()

    payload["code_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="code_hash_mismatch"):
        load_adaptive_config(path, MODULE_PATH)

    payload["schema_version"] = "unsupported"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_unsupported"):
        load_adaptive_config(path, MODULE_PATH)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "columns_missing"),
        ("empty", "candles_empty"),
        ("naive", "timezone_aware"),
        ("duplicate", "timestamps_invalid"),
        ("descending", "timestamps_not_monotonic"),
        ("gap", "not_contiguous_hourly"),
        ("non_finite", "non_finite"),
        ("negative_volume", "out_of_range"),
        ("bad_ohlc", "ohlc_invariant_invalid"),
    ],
)
def test_multitimeframe_builder_rejects_invalid_hourly_data(
    mutation: str, reason: str
) -> None:
    candles = hourly_candles(100)
    if mutation == "missing":
        candles = candles.drop(columns=["volume"])
    elif mutation == "empty":
        candles = candles.iloc[:0]
    elif mutation == "naive":
        candles["date"] = candles["date"].dt.tz_localize(None)
    elif mutation == "duplicate":
        candles.loc[1, "date"] = candles.loc[0, "date"]
    elif mutation == "descending":
        candles = candles.iloc[::-1].reset_index(drop=True)
    elif mutation == "gap":
        candles.loc[20, "date"] = candles.loc[20, "date"] + pd.Timedelta(1, unit="m")
    elif mutation == "non_finite":
        candles.loc[20, "close"] = np.inf
    elif mutation == "negative_volume":
        candles.loc[20, "volume"] = -1.0
    elif mutation == "bad_ohlc":
        candles.loc[20, "low"] = candles.loc[20, "high"] + 1.0
    with pytest.raises(ValueError, match=reason):
        build_multitimeframe_contexts(candles)


def test_timeframe_contexts_are_complete_and_causal() -> None:
    candles = hourly_candles()
    contexts = build_multitimeframe_contexts(candles)
    assert len(contexts) == len(candles)
    assert {"1h_atr", "4h_atr", "1d_atr"}.issubset(contexts.columns)
    decision_times = pd.to_datetime(contexts["decision_time"], utc=True)
    assert (decision_times <= candles["date"] + pd.Timedelta(1, unit="h")).all()
    last_closed_day = 24 * 30 - 1
    assert contexts.loc[last_closed_day, "1d_close"] == pytest.approx(
        candles.loc[last_closed_day, "close"]
    )

    changed_future = candles.copy()
    changed_future.loc[24 * 35 :, ["open", "high", "low", "close"]] += 50.0
    altered_contexts = build_multitimeframe_contexts(changed_future)
    cutoff = 24 * 35 - 2
    pd.testing.assert_series_equal(
        contexts.loc[cutoff, contexts.columns != "decision_time"],
        altered_contexts.loc[cutoff, altered_contexts.columns != "decision_time"],
        check_names=False,
    )


def test_timeframe_selector_prefers_shorter_horizon_for_fast_volatility() -> None:
    state = initial_parameter_state(AdaptiveConfig())
    row = context_row(trend_strength=0.1, volatility_score=1.0)
    decision = select_adaptive_timeframe(row, state)
    assert decision.timeframe == "1h"
    assert decision.confidence >= state.entry_confidence_floor


def test_timeframe_selector_prefers_daily_context_for_strong_calm_trend() -> None:
    state = initial_parameter_state(AdaptiveConfig())
    decision = select_adaptive_timeframe(
        context_row(trend_strength=0.95, volatility_score=0.1), state
    )
    assert decision.timeframe == "1d"
    assert decision.confidence > state.entry_confidence_floor


def test_timeframe_selector_abstains_when_context_is_ambiguous_or_missing() -> None:
    state = initial_parameter_state(AdaptiveConfig())
    row = context_row(trend_strength=0.5, volatility_score=0.5)
    decision = select_adaptive_timeframe(row, state)
    assert decision.timeframe in {"1h", "4h", "1d"}
    row = {"1h_close": float("nan")}
    missing = select_adaptive_timeframe(row, state)
    assert missing.timeframe is None
    assert missing.reason == "no_valid_timeframe_context"

    one_context = {
        key: value for key, value in context_row().items() if key.startswith("1h_")
    }
    single = select_adaptive_timeframe(one_context, state)
    assert single.confidence == pytest.approx(0.5)
    assert single.reason == "timeframe_confidence_below_floor"


def test_dynamic_stop_uses_atr_and_prior_support_or_abstains() -> None:
    config = AdaptiveConfig()
    state = initial_parameter_state(config)
    plan = plan_dynamic_stop(
        entry_price=100.0, atr=2.0, support_level=95.0, state=state, config=config
    )
    assert plan is not None
    assert plan.stop_loss < 95.0
    assert plan.stop_loss < 100.0
    assert plan.stop_distance == pytest.approx(100.0 - plan.stop_loss)

    assert plan_dynamic_stop(
        entry_price=100.0, atr=10.0, support_level=90.0, state=state, config=config
    ) is None
    assert plan_dynamic_stop(
        entry_price=100.0, atr=2.0, support_level=110.0, state=state, config=config
    ) is None
    assert plan_dynamic_stop(
        entry_price=float("nan"), atr=2.0, support_level=95.0, state=state, config=config
    ) is None
    assert plan_dynamic_stop(
        entry_price=100.0,
        atr=2.0,
        support_level=95.0,
        state=replace(state, atr_stop_multiple=99.0),
        config=config,
    ) is None


def test_adaptive_sizing_responds_to_confidence_trend_and_volatility() -> None:
    config = AdaptiveConfig()
    high = size_adaptive_position(
        equity=10_000.0, entry_price=100.0, stop_loss=95.0,
        confidence=0.95, trend_strength=0.9, volatility_score=0.1, config=config,
    )
    low = size_adaptive_position(
        equity=10_000.0, entry_price=100.0, stop_loss=95.0,
        confidence=0.4, trend_strength=0.3, volatility_score=0.9, config=config,
    )
    assert high is not None and low is not None
    assert high.risk_fraction > low.risk_fraction
    assert high.quantity > low.quantity
    assert high.notional <= 10_000.0 * config.maximum_position_fraction
    assert high.risk_fraction <= config.maximum_risk_fraction
    assert size_adaptive_position(
        equity=0, entry_price=100, stop_loss=95, confidence=0.9,
        trend_strength=0.9, volatility_score=0.1, config=config,
    ) is None
    assert size_adaptive_position(
        equity=1000, entry_price=100, stop_loss=101, confidence=0.9,
        trend_strength=0.9, volatility_score=0.1, config=config,
    ) is None
    assert size_adaptive_position(
        equity=1000, entry_price=100, stop_loss=90, confidence=float("nan"),
        trend_strength=0.9, volatility_score=0.1, config=config,
    ) is None


def test_intelligent_trailing_stop_gives_winners_room_and_never_loosens() -> None:
    config = AdaptiveConfig()
    state = initial_parameter_state(config)
    strong = update_intelligent_trailing_stop(
        current_stop=90, high_watermark=120, current_close=118, atr=2,
        support_level=110, trend_strength=0.9, macd_histogram=1.0,
        state=state, config=config,
    )
    weak = update_intelligent_trailing_stop(
        current_stop=90, high_watermark=120, current_close=118, atr=2,
        support_level=110, trend_strength=0.3, macd_histogram=-0.1,
        state=state, config=config,
    )
    assert strong is not None and weak is not None
    assert strong[1] == "TREND_RUNNER"
    assert weak[1] == "TIGHTEN_TRAIL"
    assert strong[0] >= 90 and weak[0] >= 90
    assert weak[0] > strong[0]
    assert update_intelligent_trailing_stop(
        current_stop=119, high_watermark=120, current_close=118, atr=2,
        support_level=110, trend_strength=0.9, macd_histogram=1,
        state=state, config=config,
    ) is None
    assert update_intelligent_trailing_stop(
        current_stop=float("nan"), high_watermark=120, current_close=118, atr=2,
        support_level=110, trend_strength=0.9, macd_histogram=1,
        state=state, config=config,
    ) is None


def test_failure_analysis_widens_a_tight_stop_for_future_trades() -> None:
    config = AdaptiveConfig()
    state = initial_parameter_state(config)
    result, updated = analyze_post_trade_failure(closed_trade(), state, config)
    assert result["classification"] == "stop_too_tight"
    assert result["parameter_update_applied"] is True
    assert updated.atr_stop_multiple == pytest.approx(state.atr_stop_multiple + 0.25)
    assert updated.trailing_atr_multiple == pytest.approx(state.trailing_atr_multiple + 0.25)
    assert updated.revision == 1
    assert updated.trades_observed == 1
    reachable = closed_trade(stop_atr=2.2, favorable_atr=1.1)
    reachable_analysis, _ = analyze_post_trade_failure(reachable, state, config)
    assert reachable_analysis["classification"] == "stop_too_tight"


def test_failure_analysis_increases_selectivity_after_regime_shift_or_signal_failure() -> None:
    config = AdaptiveConfig()
    state = initial_parameter_state(config)
    shifted, shifted_state = analyze_post_trade_failure(
        closed_trade(entry_regime="trend", exit_regime="range"), state, config
    )
    assert shifted["classification"] == "market_state_shift"
    assert shifted_state.entry_confidence_floor > state.entry_confidence_floor

    reversed_trade, reversed_state = analyze_post_trade_failure(
        closed_trade(exit_reason="trend_reversal"), state, config
    )
    assert reversed_trade["classification"] == "signal_reversal"
    assert reversed_state.entry_confidence_floor > state.entry_confidence_floor

    generic, generic_state = analyze_post_trade_failure(
        closed_trade(exit_reason="end_of_data", stop_atr=2.0, favorable_atr=0.1),
        state, config,
    )
    assert generic["classification"] == "signal_or_execution_assumption_failure"
    assert generic_state.entry_confidence_floor > state.entry_confidence_floor


def test_three_consecutive_wins_adjust_future_trailing_and_confidence() -> None:
    config = AdaptiveConfig()
    state = initial_parameter_state(config)
    for _ in range(3):
        _, state = analyze_post_trade_failure(
            closed_trade(pnl=20.0, exit_reason="trailing_stop"), state, config
        )
    assert state.trades_observed == 3
    assert state.revision == 1
    assert state.trailing_atr_multiple > config.initial_trailing_atr_multiple
    assert state.entry_confidence_floor < config.minimum_selection_confidence
    assert state.consecutive_wins == 0


def test_failure_analysis_rejects_unverifiable_trade_and_respects_parameter_caps() -> None:
    config = AdaptiveConfig()
    state = replace(
        initial_parameter_state(config),
        atr_stop_multiple=config.maximum_stop_atr_multiple,
        trailing_atr_multiple=config.maximum_trailing_atr_multiple,
    )
    _, capped = analyze_post_trade_failure(closed_trade(), state, config)
    assert capped.atr_stop_multiple == config.maximum_stop_atr_multiple
    assert capped.trailing_atr_multiple == config.maximum_trailing_atr_multiple
    assert capped.revision == state.revision

    max_floor = replace(state, entry_confidence_floor=config.maximum_confidence_floor)
    _, unchanged = analyze_post_trade_failure(
        closed_trade(entry_regime="trend", exit_regime="range"), max_floor, config
    )
    assert unchanged.entry_confidence_floor == config.maximum_confidence_floor
    with pytest.raises(ValueError, match="numeric_fields_invalid"):
        analyze_post_trade_failure({"net_pnl_quote": float("nan")}, state, config)
    with pytest.raises(ValueError, match="categorical_fields_invalid"):
        analyze_post_trade_failure(
            {"net_pnl_quote": -1, "initial_stop_distance_atr": 1,
             "max_favorable_excursion_atr": 0.1, "exit_reason": 1,
             "entry_regime": "trend", "exit_regime": "trend"},
            state, config,
        )


def test_metrics_keep_empty_samples_null_and_report_closed_trade_drawdown() -> None:
    config = AdaptiveConfig()
    empty = summarize_adaptive_metrics([], initial_equity=config.initial_equity)
    assert empty["status"] == "NO_TRADES"
    assert empty["win_rate_pct"] is None
    assert empty["expectancy_bps_per_trade"] is None
    trades = [
        {"net_return_bps": 100.0, "net_pnl_quote": 10.0},
        {"net_return_bps": -200.0, "net_pnl_quote": -20.0},
    ]
    metrics = summarize_adaptive_metrics(trades, initial_equity=1000.0)
    assert metrics["trade_count"] == 2
    assert metrics["win_rate_pct"] == 50.0
    assert metrics["expectancy_bps_per_trade"] == -50.0
    assert metrics["total_return_pct"] == pytest.approx(-1.0)
    assert metrics["max_drawdown_pct"] > 0
    with pytest.raises(ValueError, match="initial_equity_invalid"):
        summarize_adaptive_metrics([], initial_equity=0.0)
    with pytest.raises(ValueError, match="non_finite_trade"):
        summarize_adaptive_metrics(
            [{"net_return_bps": float("nan"), "net_pnl_quote": 1.0}],
            initial_equity=1000.0,
        )
