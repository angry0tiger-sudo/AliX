from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.economic_backtest import (
    ExecutionConfig,
    build_forward_signals,
    extract_qualifying_candidates,
    max_drawdown_from_equity,
    probability_to_signal,
    resample_ohlcv,
    simulate_signals,
    validate_execution_config,
    validate_market_manifest,
    validate_source_report,
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
        "commit_sha": "e34861c4873b7cc2170aaaf6c79ed9b464df0a9",
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
        "commit_sha": "e34861c4873b7cc2170aaaf6c79ed9b464df0a9",
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
    assert incomplete_first_bar.iloc[0]["date"] == pd.Timestamp(
        "2026-01-01T00:15:00Z"
    )
    with pytest.raises(ValueError, match="resampled_bars_empty"):
        resample_ohlcv(minute.drop(index=[5, 20]), "15m")


def test_long_round_trip_charges_fee_and_slippage_on_both_sides() -> None:
    bars = _bars(3, [100.0, 100.0, 110.0])
    signals = [
        {"execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"), "signal": "BUY"},
        {"execution_timestamp": pd.Timestamp("2026-01-01T00:30:00Z"), "signal": "HOLD"},
    ]
    result = simulate_signals(bars, signals, ExecutionConfig())
    trade = result["trades"][0]
    assert result["metrics"]["trade_count"] == 1
    assert trade["entry_fill_price"] == pytest.approx(100.2)
    assert trade["exit_fill_price"] == pytest.approx(109.78)
    assert trade["fees_usd"] > 0
    assert trade["slippage_cost_usd"] == pytest.approx(42.0)
    assert trade["net_pnl_usd"] < trade["gross_market_pnl_usd"]
    assert trade["exit_reason"] == "HOLD"


def test_hold_never_opens_and_closes_an_existing_position() -> None:
    bars = _bars(4, [100.0, 100.0, 110.0, 110.0])
    hold_only = simulate_signals(
        bars,
        [
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:15:00Z"),
                "signal": "HOLD",
            }
        ],
        ExecutionConfig(),
    )
    assert hold_only["trades"] == []
    assert hold_only["metrics"]["trade_count"] == 0
    close_existing = simulate_signals(
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
        ExecutionConfig(),
    )
    assert len(close_existing["trades"]) == 1
    assert close_existing["trades"][0]["exit_reason"] == "HOLD"


def test_signal_reversal_closes_and_opens_a_new_direction() -> None:
    bars = _bars(4, [100.0, 100.0, 105.0, 95.0])
    result = simulate_signals(
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
        ],
        ExecutionConfig(),
    )
    assert len(result["trades"]) == 2
    assert result["trades"][0]["direction"] == "LONG"
    assert result["trades"][0]["exit_reason"] == "SIGNAL_REVERSAL"
    assert result["trades"][1]["direction"] == "SHORT"
    assert result["trades"][1]["exit_reason"] == "END_OF_FORWARD_WINDOW"


def test_profit_factor_and_drawdown_are_net_of_costs_and_peak_to_trough() -> None:
    assert max_drawdown_from_equity([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="curve"):
        max_drawdown_from_equity([])
    with pytest.raises(ValueError, match="non_positive"):
        max_drawdown_from_equity([100.0, 0.0])
    bars = _bars(4, [100.0, 100.0, 120.0, 80.0])
    result = simulate_signals(
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
            {
                "execution_timestamp": pd.Timestamp("2026-01-01T00:45:00Z"),
                "signal": "SELL",
            },
        ],
        ExecutionConfig(),
    )
    assert result["metrics"]["profit_factor_net_of_costs"] is not None
    assert result["metrics"]["max_drawdown_fraction"] >= 0
    assert result["total_fees_usd"] > 0
    assert result["total_slippage_cost_usd"] > 0


def test_execution_rejects_misaligned_or_invalid_signals() -> None:
    bars = _bars()
    with pytest.raises(ValueError, match="timestamp_mismatch"):
        simulate_signals(
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
        simulate_signals(
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
        simulate_signals(bars, [], ExecutionConfig())
