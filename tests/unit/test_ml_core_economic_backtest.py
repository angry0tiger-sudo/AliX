from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.economic_backtest import (
    NO_ENTRY_SIGNAL,
    ExecutionConfig,
    TradeManagementConfig,
    build_forward_signals,
    extract_qualifying_candidates,
    max_drawdown_from_equity,
    probability_to_signal,
    resample_ohlcv,
    simulate_signals,
    validate_execution_config,
    validate_trade_management_config,
    validate_market_manifest,
    validate_source_report,
    _validate_drawdown_accounting,
)


def _bars(rows: int = 5, prices: list[float] | None = None) -> DataFrame:
    values = prices or [100.0] * rows
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="15min")
    return DataFrame(
        {
            "date": dates,
            "open": values,
            "close": values,
            "high": values,
            "low": values,
            "volume": [10.0] * rows,
        }
    )


def _simulate(
    bars: DataFrame,
    signals: list[dict[str, Any]],
    config: ExecutionConfig | None = None,
    *,
    management: TradeManagementConfig | None = None,
    atr_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run small deterministic fixtures with explicit point-in-time ATR."""
    selected_management = management or TradeManagementConfig(atr_period=1)
    values = atr_values or [10.0] * len(bars)
    return simulate_signals(
        bars,
        signals,
        config or ExecutionConfig(),
        trade_management_config=selected_management,
        atr_values=pd.Series(values, dtype=float),
    )


def test_execution_config_rejects_invalid_and_leveraged_assumptions() -> None:
    validate_execution_config(ExecutionConfig())
    with pytest.raises(ValueError, match="initial_equity"):
        validate_execution_config(ExecutionConfig(initial_equity=0))
    with pytest.raises(ValueError, match="execution_costs"):
        validate_execution_config(ExecutionConfig(fee_bps=-1))
    with pytest.raises(ValueError, match="position_fraction"):
        validate_execution_config(ExecutionConfig(position_fraction=1.01))
    with pytest.raises(ValueError, match="maximum_drawdown"):
        validate_execution_config(ExecutionConfig(maximum_drawdown=1.0))
    with pytest.raises(ValueError, match="finite"):
        validate_execution_config(ExecutionConfig(slippage_bps=float("nan")))


def test_source_report_and_market_manifest_fail_closed() -> None:
    report = {
        "schema_version": "alix.ml-core.v4h-tuning-geometry.v1",
        "commit_sha": "e34861c4873b7cc2170aaaf6c79ed9b464df0a9a",
        "diagnostic_scope": "CLASSIFICATION_ONLY",
        "economic_evaluation_performed": False,
        "trained_models": 96,
        "experiment_count": 48,
        "acceptance_policy": {"status": "MET", "strict_acceptance_count": 13},
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }
    validate_source_report(report)
    bad = {**report, "commit_sha": "bad"}
    with pytest.raises(ValueError, match="source_report"):
        validate_source_report(bad)
    source = {
        "symbol": "BTCUSDT",
        "requested_days": 1,
        "row_count": 1440,
        "start_utc": "2026-01-01T00:00:00+00:00",
        "end_utc": "2026-01-01T23:59:00+00:00",
        "csv_sha256": "abc",
    }
    current = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "requested_days": 1,
        "row_count": 1440,
        "start_utc": "2026-01-01T00:00:00Z",
        "end_utc": "2026-01-01T23:59:00Z",
        "csv_sha256": "abc",
        "commit_sha": "new-sha",
        "public_api_only": True,
        "credentials_used": False,
        "missing_minute_intervals": 0,
        "private_api_calls": 0,
        "real_order_attempts": 0,
    }
    validate_market_manifest(source, current, current_commit_sha="new-sha")
    current["csv_sha256"] = "different"
    with pytest.raises(ValueError, match="integrity"):
        validate_market_manifest(source, current, current_commit_sha="new-sha")


def test_candidate_extraction_requires_exact_thirteen_forward_passes() -> None:
    variants: dict[str, Any] = {}
    records = [
        {
            "row_index": 10,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "probabilities": {"BUY": 0.6, "SELL": 0.2, "HOLD": 0.2},
        },
        {
            "row_index": 11,
            "timestamp": "2026-01-01T00:15:00+00:00",
            "probabilities": {"BUY": 0.2, "SELL": 0.2, "HOLD": 0.6},
        },
    ]
    for index in range(13):
        variants[str(index)] = {
            "threshold_optimization": {
                "selected_threshold": 0.5,
                "post_calibration_validation": {
                    "hold_metrics": {"precision": 0.41, "recall": 0.21},
                    "start_row_index": 10,
                    "row_count": 2,
                },
            },
            "hyperparameter_tuning": {"selected_parameters": {"learning_rate": 0.05}},
            "validation_probabilities": records,
        }
    report = {
        "schema_version": "alix.ml-core.v4h-tuning-geometry.v1",
        "commit_sha": "e34861c4873b7cc2170aaaf6c79ed9b464df0a9a",
        "diagnostic_scope": "CLASSIFICATION_ONLY",
        "economic_evaluation_performed": False,
        "trained_models": 96,
        "experiment_count": 48,
        "acceptance_policy": {"status": "MET", "strict_acceptance_count": 13},
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
        "experiments": [
            {
                "scenario": "V4H_Tighter_SL",
                "pair": "BTC/USDT",
                "timeframe": "15m",
                "feature_profile": "Baseline",
                "model_variants": variants,
            }
        ],
    }
    assert len(extract_qualifying_candidates(report)) == 13
    experiment = cast(dict[str, Any], cast(list[Any], report["experiments"])[0])
    variants = cast(dict[str, Any], experiment["model_variants"])
    variants.pop("12")
    with pytest.raises(ValueError, match="candidate_count"):
        extract_qualifying_candidates(report)


def test_probability_signal_matches_v4h_hold_and_direction_policy() -> None:
    assert probability_to_signal({"BUY": 0.2, "SELL": 0.3, "HOLD": 0.5}, 0.5) == "HOLD"
    assert probability_to_signal({"BUY": 0.5, "SELL": 0.5, "HOLD": 0.0}, 0.5) == "BUY"
    assert probability_to_signal({"BUY": 0.2, "SELL": 0.7, "HOLD": 0.1}, 0.5) == "SELL"
    with pytest.raises(ValueError, match="threshold"):
        probability_to_signal({"BUY": 0.2, "SELL": 0.3, "HOLD": 0.5}, 1.1)
    with pytest.raises(ValueError, match="probabilities"):
        probability_to_signal({"BUY": 0.2, "SELL": 0.3, "HOLD": 0.4}, 0.5)


def test_forward_signals_use_only_ordered_persisted_forward_rows() -> None:
    candidate = {
        "timeframe": "15m",
        "selected_hold_threshold": 0.5,
        "forward_row_count": 2,
        "probability_records": [
            {
                "row_index": 10,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "probabilities": {"BUY": 0.7, "SELL": 0.2, "HOLD": 0.1},
            },
            {
                "row_index": 11,
                "timestamp": "2026-01-01T00:15:00+00:00",
                "probabilities": {"BUY": 0.1, "SELL": 0.2, "HOLD": 0.7},
            },
        ],
    }
    signals = build_forward_signals(candidate)
    assert [item["signal"] for item in signals] == ["BUY", "HOLD"]
    assert signals[0]["execution_timestamp"] == pd.Timestamp("2026-01-01T00:15:00Z")
    probability_records = cast(list[dict[str, Any]], candidate["probability_records"])
    probability_records[1]["row_index"] = 10
    with pytest.raises(ValueError, match="order"):
        build_forward_signals(candidate)


def test_resampling_requires_complete_bars_and_preserves_ohlcv() -> None:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="1min")
    minute = DataFrame(
        {
            "date": dates,
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.5] * 30,
            "volume": [1.0] * 30,
        }
    )
    bars = resample_ohlcv(minute, "15m")
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 100.0
    assert bars.iloc[0]["high"] == 101.0
    with pytest.raises(ValueError, match="timeframe"):
        resample_ohlcv(minute, "1m")
    incomplete_first_bar = resample_ohlcv(minute.drop(index=5), "15m")
    assert len(incomplete_first_bar) == 1
    assert incomplete_first_bar.iloc[0]["date"] == pd.Timestamp("2026-01-01T00:15:00Z")
    with pytest.raises(ValueError, match="resampled_bars_empty"):
        resample_ohlcv(minute.drop(index=[5, 20]), "15m")


def test_long_round_trip_charges_fee_and_slippage_on_both_sides() -> None:
    bars = _bars(3, [100.0, 100.0, 110.0])
    signals = [
        {"execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"), "signal": "BUY"},
        {"execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"), "signal": "SELL"},
    ]
    result = _simulate(bars, signals, ExecutionConfig())
    trade = result["trades"][0]
    assert result["metrics"]["trade_count"] == 1
    assert trade["entry_fill_price"] == pytest.approx(100.2)
    assert trade["exit_fill_price"] == pytest.approx(109.78)
    assert trade["fees_usd"] > 0
    assert trade["slippage_cost_usd"] == pytest.approx(42.0)
    assert trade["net_pnl_usd"] < trade["gross_market_pnl_usd"]
    assert trade["exit_reason"] == "SELL"


def test_hold_never_opens_and_keeps_existing_long_until_window_end() -> None:
    bars = _bars(4, [100.0, 100.0, 110.0, 110.0])
    hold_only = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "HOLD",
            }
        ],
    )
    assert hold_only["trades"] == []
    assert hold_only["metrics"]["trade_count"] == 0
    keep_existing = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "BUY",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"),
                "signal": "HOLD",
            },
        ],
    )
    assert len(keep_existing["trades"]) == 1
    assert keep_existing["trades"][0]["exit_reason"] == "END_OF_FORWARD_WINDOW"


def test_spot_sell_closes_long_and_is_ignored_when_flat() -> None:
    bars = _bars(4, [100.0, 100.0, 105.0, 95.0])
    result = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "BUY",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"),
                "signal": "SELL",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:45:00Z"),
                "signal": "SELL",
            },
        ],
        ExecutionConfig(),
    )
    assert len(result["trades"]) == 1
    assert result["trades"][0]["direction"] == "LONG"
    assert result["trades"][0]["exit_reason"] == "SELL"
    assert all(item["direction"] != "SHORT" for item in result["trades"])

    flat_sell = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "SELL",
            }
        ],
        ExecutionConfig(),
    )
    assert flat_sell["trades"] == []
    assert flat_sell["metrics"]["trade_count"] == 0


def test_zero_cost_and_fee_only_execution_accounting() -> None:
    bars = _bars(3, [100.0, 100.0, 110.0])
    signals = [
        {
            "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
            "signal": "BUY",
        },
        {
            "execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"),
            "signal": "HOLD",
        },
    ]
    zero_cost = _simulate(
        bars, signals, ExecutionConfig(fee_bps=0.0, slippage_bps=0.0)
    )
    fee_only = _simulate(
        bars, signals, ExecutionConfig(fee_bps=25.0, slippage_bps=0.0)
    )
    assert zero_cost["total_fees_usd"] == 0.0
    assert zero_cost["total_slippage_cost_usd"] == 0.0
    assert fee_only["total_fees_usd"] > 0.0
    assert fee_only["total_slippage_cost_usd"] == 0.0
    assert fee_only["trades"][0]["entry_fill_price"] == pytest.approx(100.0)
    assert fee_only["trades"][0]["net_pnl_usd"] < zero_cost["trades"][0]["net_pnl_usd"]


def test_undefined_profit_factor_does_not_pass_strict_profit_factor_gate() -> None:
    bars = _bars(3, [100.0, 100.0, 110.0])
    result = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "BUY",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"),
                "signal": "HOLD",
            },
        ],
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
    )
    assert result["metrics"]["profit_factor_infinite_no_losses"] is True
    assert result["metrics"]["profit_factor_net_of_costs"] is None
    assert result["metrics"]["profit_factor_gate_passed"] is False
    assert result["metrics"]["economic_candidate_gate_passed"] is False


def test_profit_factor_and_drawdown_are_net_of_costs_and_peak_to_trough() -> None:
    assert max_drawdown_from_equity([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="curve"):
        max_drawdown_from_equity([])
    with pytest.raises(ValueError, match="non_positive"):
        max_drawdown_from_equity([100.0, 0.0])
    bars = _bars(5, [100.0, 100.0, 120.0, 80.0, 70.0])
    result = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "BUY",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"),
                "signal": "SELL",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:45:00Z"),
                "signal": "BUY",
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T01:00:00Z"),
                "signal": "SELL",
            },
        ],
    )
    assert result["metrics"]["trade_count"] == 2
    assert all(trade["direction"] == "LONG" for trade in result["trades"])
    assert result["metrics"]["profit_factor_net_of_costs"] is not None
    assert result["metrics"]["max_drawdown_fraction"] >= 0
    assert result["total_fees_usd"] > 0
    assert result["total_slippage_cost_usd"] > 0


def test_confidence_filter_rejects_weak_buy_and_preserves_sell_exit() -> None:
    candidate = {
        "timeframe": "15m",
        "selected_hold_threshold": 0.5,
        "forward_row_count": 3,
        "probability_records": [
            {
                "row_index": index,
                "timestamp": f"2026-01-01T00:{index * 15:02d}:00+00:00",
                "probabilities": probabilities,
            }
            for index, probabilities in enumerate(
                [
                    {"BUY": 0.54, "SELL": 0.26, "HOLD": 0.20},
                    {"BUY": 0.56, "SELL": 0.24, "HOLD": 0.20},
                    {"BUY": 0.20, "SELL": 0.60, "HOLD": 0.20},
                ]
            )
        ],
    }
    signals = build_forward_signals(candidate, confidence_threshold=0.55)
    assert [item["raw_signal"] for item in signals] == ["BUY", "BUY", "SELL"]
    assert [item["signal"] for item in signals] == [
        NO_ENTRY_SIGNAL,
        "BUY",
        "SELL",
    ]
    assert [item["confidence_passed"] for item in signals] == [False, True, True]
    assert signals[0]["confidence_threshold"] == pytest.approx(0.55)


def test_default_signal_builder_preserves_v4j_unfiltered_behavior() -> None:
    candidate = {
        "timeframe": "15m",
        "selected_hold_threshold": 0.5,
        "forward_row_count": 1,
        "probability_records": [
            {
                "row_index": 10,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "probabilities": {"BUY": 0.40, "SELL": 0.35, "HOLD": 0.25},
            }
        ],
    }
    assert build_forward_signals(candidate)[0]["signal"] == "BUY"
    assert (
        build_forward_signals(candidate, confidence_threshold=0.55)[0]["signal"]
        == NO_ENTRY_SIGNAL
    )


def test_trade_management_config_rejects_invalid_assumptions() -> None:
    validate_trade_management_config(TradeManagementConfig())
    with pytest.raises(ValueError, match="atr_period"):
        validate_trade_management_config(TradeManagementConfig(atr_period=0))
    with pytest.raises(ValueError, match="horizon_candles"):
        validate_trade_management_config(TradeManagementConfig(horizon_candles=0))
    with pytest.raises(ValueError, match="multipliers"):
        validate_trade_management_config(
            TradeManagementConfig(take_profit_atr_multiplier=float("nan"))
        )
    with pytest.raises(ValueError, match="confidence_threshold"):
        validate_trade_management_config(
            TradeManagementConfig(confidence_threshold=0.0)
        )


def test_atr_take_profit_stop_loss_and_same_candle_collision() -> None:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min")
    base = DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 101.0, 101.0],
            "high": [100.5, 100.5, 103.0, 101.0],
            "low": [99.5, 99.5, 100.5, 99.0],
            "close": [100.0, 100.0, 102.5, 100.0],
            "volume": [10.0] * 4,
        }
    )
    signals = [
        {
            "signal_timestamp": dates[0],
            "execution_timestamp": dates[1],
            "signal": "BUY",
            "buy_probability": 0.8,
        }
    ]
    management = TradeManagementConfig(atr_period=1)
    take_profit = _simulate(
        base,
        signals,
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        management=management,
        atr_values=[1.0] * 4,
    )
    assert take_profit["trades"][0]["exit_reason"] == "TP_HIT"
    assert take_profit["trades"][0]["exit_reference_price"] == pytest.approx(102.0)
    assert take_profit["trades"][0]["take_profit_price"] == pytest.approx(102.0)
    assert take_profit["trades"][0]["stop_loss_price"] == pytest.approx(98.5)
    assert take_profit["trades"][0]["exit_trigger_type"] == "INTRABAR_TAKE"

    stop_frame = base.copy()
    stop_frame.loc[2, ["high", "low", "close"]] = [101.0, 98.0, 99.0]
    stop_loss = _simulate(
        stop_frame,
        signals,
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        management=management,
        atr_values=[1.0] * 4,
    )
    assert stop_loss["trades"][0]["exit_reason"] == "SL_HIT"
    assert stop_loss["trades"][0]["exit_reference_price"] == pytest.approx(98.5)

    collision_frame = base.copy()
    collision_frame.loc[2, ["high", "low", "close"]] = [103.0, 98.0, 100.0]
    collision = _simulate(
        collision_frame,
        signals,
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        management=management,
        atr_values=[1.0] * 4,
    )
    assert collision["trades"][0]["exit_reason"] == "COLLISION"
    assert collision["trades"][0]["exit_reference_price"] == pytest.approx(98.5)
    assert collision["trades"][0]["exit_trigger_type"] == "INTRABAR_COLLISION"


def test_gap_stop_uses_open_price_and_precedes_holding_signal() -> None:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min")
    bars = DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 97.0, 97.0],
            "high": [100.5, 100.5, 98.0, 98.0],
            "low": [99.5, 99.5, 96.0, 96.0],
            "close": [100.0, 100.0, 97.0, 97.0],
            "volume": [10.0] * 4,
        }
    )
    result = _simulate(
        bars,
        [
            {
                "signal_timestamp": dates[0],
                "execution_timestamp": dates[1],
                "signal": "BUY",
                "buy_probability": 0.75,
            },
            {
                "signal_timestamp": dates[1],
                "execution_timestamp": dates[2],
                "signal": "HOLD",
                "buy_probability": 0.2,
            },
        ],
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        atr_values=[1.0] * 4,
    )
    trade = result["trades"][0]
    assert trade["exit_reason"] == "SL_HIT"
    assert trade["exit_trigger_type"] == "OPEN_GAP"
    assert trade["exit_reference_price"] == pytest.approx(97.0)


def test_hold_keeps_position_until_twelve_candle_vertical_expiry() -> None:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=13, freq="15min")
    bars = DataFrame(
        {
            "date": dates,
            "open": [100.0] * 13,
            "high": [101.0] * 13,
            "low": [99.0] * 13,
            "close": [100.0] * 13,
            "volume": [10.0] * 13,
        }
    )
    signals = [
        {
            "signal_timestamp": dates[index - 1],
            "execution_timestamp": dates[index],
            "signal": "BUY" if index == 1 else "HOLD",
            "buy_probability": 0.8 if index == 1 else 0.2,
        }
        for index in range(1, 13)
    ]
    result = _simulate(
        bars,
        signals,
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        management=TradeManagementConfig(atr_period=1, horizon_candles=12),
        atr_values=[1.0] * 13,
    )
    trade = result["trades"][0]
    assert trade["exit_reason"] == "HORIZON_EXPIRY"
    assert trade["exit_trigger_type"] == "VERTICAL_BARRIER"
    assert trade["bars_held"] == 12
    assert trade["exit_timestamp"] == dates[12].isoformat()


def test_rejected_buy_does_not_close_active_long_or_create_short() -> None:
    bars = _bars(4, [100.0, 100.0, 101.0, 101.0])
    result = _simulate(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "BUY",
                "buy_probability": 0.8,
            },
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"),
                "signal": NO_ENTRY_SIGNAL,
                "buy_probability": 0.4,
            },
        ],
        atr_values=[10.0] * 4,
    )
    assert len(result["trades"]) == 1
    assert result["trades"][0]["direction"] == "LONG"
    assert result["trades"][0]["exit_reason"] == "END_OF_FORWARD_WINDOW"
    assert result["trades"][0]["bars_held"] == 3


def test_drawdown_accounting_allows_only_equity_rounding_tolerance() -> None:
    _validate_drawdown_accounting(0.2, [100.0, 80.0])
    _validate_drawdown_accounting(0.20000000005, [100.0, 80.0])
    with pytest.raises(RuntimeError, match="economic_drawdown_accounting_inconsistent"):
        _validate_drawdown_accounting(0.200001, [100.0, 80.0])


def test_execution_rejects_misaligned_or_invalid_signals() -> None:
    bars = _bars()
    with pytest.raises(ValueError, match="timestamp_mismatch"):
        _simulate(
            bars,
            [
                {
                    "execution_timestamp": pd.Timestamp("2026-01-02T00:00:00Z"),
                    "signal": "BUY",
                }
            ],
            ExecutionConfig(),
        )
    with pytest.raises(ValueError, match="contract"):
        _simulate(
            bars,
            [
                {
                    "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                    "signal": "WAIT",
                }
            ],
            ExecutionConfig(),
        )
    with pytest.raises(ValueError, match="bars_or_signals"):
        _simulate(bars, [], ExecutionConfig())
