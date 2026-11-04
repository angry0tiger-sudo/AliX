"""Unit contracts for the V4-AB Primary cost/frequency diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4ab_cost_frequency as v4ab


def _bars(count: int = 1_000) -> DataFrame:
    close = np.linspace(100.0, 150.0, count)
    return DataFrame(
        {
            "date": pd.date_range("2024-01-01T00:00:00Z", periods=count, freq="h"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(count, 1_000.0),
        }
    )


def _cell(
    gross_pf: float | None,
    net_pf: float | None,
    count: int = 40,
    symbol: str = "BTCUSDT",
    horizon: int = 24,
    policy: str = "TB_1.5_1.0",
) -> dict[str, object]:
    return {
        "candidate_id": f"{symbol}|1h|{horizon}|{policy}",
        "symbol": symbol,
        "pair": f"{symbol.removesuffix('USDT')}/USDT",
        "timeframe": "1h",
        "horizon_bars": horizon,
        "exit_policy": policy,
        "primary_signal_count": 50,
        "eligible_primary_signal_count": 40,
        "trade_count": count,
        "gross_profit_factor": gross_pf,
        "gross_profit_factor_infinite_no_losses": False,
        "net_profit_factor": net_pf,
        "net_profit_factor_infinite_no_losses": False,
        "win_rate": 0.25,
        "gross_win_rate": 0.50,
        "median_bars_held": 24.0,
        "max_drawdown_percent": 15.0,
        "gross_pnl_total_usd": 100.0,
        "net_pnl_total_usd": -50.0,
        "fees_total_usd": 25.0,
        "slippage_total_usd": 25.0,
    }


def test_split_and_request_contracts_fail_closed() -> None:
    assert v4ab._outer_boundaries(1_000) == (700, 850)
    with pytest.raises(ValueError, match="v4ab_resampled_dataset_too_short"):
        v4ab._outer_boundaries(999)
    valid_sha = "a" * 40
    v4ab._validate_run_request(730, valid_sha)
    with pytest.raises(ValueError, match="v4ab_dataset_window_must_be_730_days"):
        v4ab._validate_run_request(365, valid_sha)
    with pytest.raises(ValueError, match="v4ab_commit_sha_invalid"):
        v4ab._validate_run_request(730, "bad")


def test_primary_rule_is_the_exact_v4y_closed_candle_signal() -> None:
    close = np.concatenate((np.full(220, 100.0), np.full(40, 130.0)))
    bars = _bars(len(close))
    bars["open"] = close
    bars["high"] = close * 1.01
    bars["low"] = close * 0.99
    bars["close"] = close
    assert np.flatnonzero(
        v4ab._primary_signal_series(bars).to_numpy(dtype=bool)
    ).tolist() == [220]


def test_maturity_cutoff_and_forward_action_timestamps() -> None:
    signals = Series(False, index=range(1_000), dtype=bool)
    signals.iloc[[700, 728, 729]] = True
    assert v4ab._forward_primary_signal_indices(signals, 700, 850) == [700, 728]
    with pytest.raises(ValueError, match="v4ab_primary_signal_window_invalid"):
        v4ab._forward_primary_signal_indices(signals.iloc[:100], 700, 850)

    bars = _bars()
    actions = v4ab._build_forward_actions(bars, [700, 728], 850)
    assert [item["signal"] for item in actions] == ["BUY", "BUY", "NO_ENTRY"]
    assert actions[0]["row_index"] == 700
    assert actions[0]["execution_timestamp"] == bars["date"].iloc[701].isoformat()
    assert actions[-1]["row_index"] == 848
    assert actions[-1]["execution_timestamp"] == bars["date"].iloc[849].isoformat()


def test_forward_action_guards_and_signal_timezone() -> None:
    bars = _bars()
    with pytest.raises(
        ValueError, match="v4ab_primary_events_must_be_unique_and_ordered"
    ):
        v4ab._build_forward_actions(bars, [701, 700], 850)
    with pytest.raises(ValueError, match="v4ab_primary_event_outside_forward_window"):
        v4ab._build_forward_actions(bars, [849], 850)
    with pytest.raises(
        ValueError, match="v4ab_forward_sentinel_collides_with_primary_event"
    ):
        v4ab._build_forward_actions(bars, [848], 850)
    with pytest.raises(ValueError, match="v4ab_signal_execution_index_invalid"):
        v4ab._signal_record(bars, 999, "BUY")
    naive = bars.copy()
    naive["date"] = pd.date_range("2024-01-01", periods=len(bars), freq="h")
    with pytest.raises(ValueError, match="v4ab_signal_timestamp_timezone_required"):
        v4ab._signal_record(naive, 10, "BUY")


def test_registered_exit_structures_and_invalid_parameters() -> None:
    tb = v4ab._trade_management_config("TB_1.5_1.0", 72)
    assert (tb.take_profit_atr_multiplier, tb.stop_loss_atr_multiplier) == (1.5, 1.0)
    assert tb.horizon_candles == 72
    time = v4ab._trade_management_config("TIME_N", 24)
    assert time.time_exit_candles == 24
    assert time.horizon_candles == 25
    assert time.take_profit_atr_multiplier == 10.0
    trail = v4ab._trade_management_config("TRAIL_2.0", 120)
    assert trail.stop_loss_atr_multiplier == 2.0
    assert trail.take_profit_atr_multiplier == 4.0
    assert trail.trailing_stop_enabled is True
    assert (trail.trailing_activation_atr, trail.trailing_step_atr) == (2.0, 1.0)
    with pytest.raises(ValueError, match="v4ab_horizon_out_of_policy"):
        v4ab._trade_management_config("TB_1.5_1.0", 12)
    with pytest.raises(ValueError, match="v4ab_exit_policy_out_of_policy"):
        v4ab._trade_management_config("OTHER", 24)


def test_profit_factor_and_infinite_profit_factor_rules() -> None:
    assert v4ab._profit_factor([10.0, -5.0]) == (2.0, False)
    assert v4ab._profit_factor([10.0, 2.0]) == (None, True)
    assert v4ab._profit_factor([]) == (None, False)
    assert v4ab._pf_above(None, True, 1.3)
    assert not v4ab._pf_above(None, False, 1.0)


def test_summarize_replay_reports_gross_net_and_true_time_horizon() -> None:
    trades = [
        {
            "direction": "LONG",
            "gross_market_pnl_usd": 120.0,
            "net_pnl_usd": 30.0,
            "fees_usd": 50.0,
            "slippage_cost_usd": 40.0,
            "bars_held": 25,
            "exit_reason": "TIME_EXIT",
            "time_exit_triggered": True,
        },
        {
            "direction": "LONG",
            "gross_market_pnl_usd": -60.0,
            "net_pnl_usd": -150.0,
            "fees_usd": 50.0,
            "slippage_cost_usd": 40.0,
            "bars_held": 25,
            "exit_reason": "TIME_EXIT",
            "time_exit_triggered": True,
        },
    ]
    report = v4ab._summarize_replay(
        {"trades": trades, "metrics": {"max_drawdown_percent": 12.5}},
        "TIME_N",
        24,
    )
    assert report["gross_profit_factor"] == 2.0
    assert report["net_profit_factor"] == 0.2
    assert report["median_bars_held"] == 24.0
    assert report["gross_pnl_total_usd"] == 60.0
    assert report["net_pnl_total_usd"] == -120.0
    assert report["max_drawdown_percent"] == 12.5


def test_summarize_replay_rejects_invalid_trade_shapes() -> None:
    trade = {
        "direction": "LONG",
        "gross_market_pnl_usd": 1.0,
        "net_pnl_usd": 1.0,
        "fees_usd": 0.0,
        "slippage_cost_usd": 0.0,
        "bars_held": 25,
        "exit_reason": "TIME_EXIT",
        "time_exit_triggered": True,
    }
    metrics = {"max_drawdown_percent": 0.0}
    with pytest.raises(
        ValueError, match="v4ab_unconditional_time_exit_contract_failed"
    ):
        v4ab._summarize_replay(
            {"trades": [{**trade, "bars_held": 24}], "metrics": metrics},
            "TIME_N",
            24,
        )
    with pytest.raises(ValueError, match="v4ab_non_spot_trade_detected"):
        v4ab._summarize_replay(
            {"trades": [{**trade, "direction": "SHORT"}], "metrics": metrics},
            "TB_1.5_1.0",
            24,
        )
    with pytest.raises(ValueError, match="v4ab_trade_exceeded_registered_horizon"):
        v4ab._summarize_replay(
            {"trades": [{**trade, "bars_held": 25}], "metrics": metrics},
            "TB_1.5_1.0",
            24,
        )
    with pytest.raises(ValueError, match="v4ab_trade_pnl_non_finite"):
        v4ab._summarize_replay(
            {
                "trades": [
                    {
                        **trade,
                        "gross_market_pnl_usd": float("nan"),
                        "bars_held": 24,
                    }
                ],
                "metrics": metrics,
            },
            "TB_1.5_1.0",
            24,
        )
    with pytest.raises(ValueError, match="v4ab_candidate_drawdown_invalid"):
        v4ab._summarize_replay(
            {"trades": [], "metrics": {"max_drawdown_percent": -1.0}},
            "TB_1.5_1.0",
            24,
        )


def test_decision_rule_implements_all_three_branches_and_intermediate_case() -> None:
    assert (
        v4ab._decision_rule([_cell(0.9, 0.8)])["decision_branch"]
        == "NO_GROSS_PF_ABOVE_1_HALT_META_AND_PIVOT_PRIMARY_GENERATOR"
    )
    assert (
        v4ab._decision_rule([_cell(1.2, 0.8)])["decision_branch"]
        == "GROSS_EDGE_BUT_NET_PF_BELOW_1_COST_FREQUENCY_OPTIMIZATION"
    )
    assert (
        v4ab._decision_rule([_cell(1.4, 1.35)])["decision_branch"]
        == "NET_PF_GT_1_3_WITH_AT_LEAST_30_TRADES_FLAG_FOR_INDEPENDENT_OOS"
    )
    assert (
        v4ab._decision_rule([_cell(1.2, 1.1, count=20)])["decision_branch"]
        == "INTERMEDIATE_RESULT_NOT_EXPLICITLY_COVERED_BY_REGISTERED_RULE"
    )
    infinite = _cell(None, None)
    infinite["gross_profit_factor_infinite_no_losses"] = True
    infinite["net_profit_factor_infinite_no_losses"] = True
    assert v4ab._decision_rule([infinite])["decision_branch"].startswith("NET_PF")


def test_report_requires_exact_18_cell_matrix_and_sets_safety() -> None:
    cells = [
        _cell(
            1.2,
            0.8,
            symbol=symbol,
            horizon=horizon,
            policy=policy,
        )
        for symbol in ("BTCUSDT", "ETHUSDT")
        for horizon in (24, 72, 120)
        for policy in v4ab.EXIT_POLICIES
    ]
    report = v4ab._build_report("a" * 40, 730, [], cells, [])
    assert len(report["cells"]) == 18
    assert report["execution_cost_contract"]["round_trip_cost_bps"] == 90.0
    assert report["primary_model"]["meta_model_used"] is False
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    with pytest.raises(ValueError, match="v4ab_candidate_matrix_incomplete"):
        v4ab._build_report("a" * 40, 730, [], cells[:-1], [])


def test_summary_formats_pf_and_contains_every_cell_and_required_metrics() -> None:
    cells = [
        _cell(
            1.2,
            0.8,
            symbol=symbol,
            horizon=horizon,
            policy=policy,
        )
        for symbol in ("BTCUSDT", "ETHUSDT")
        for horizon in (24, 72, 120)
        for policy in v4ab.EXIT_POLICIES
    ]
    cells[0]["gross_profit_factor_infinite_no_losses"] = True
    cells[1]["net_profit_factor"] = None
    decision = v4ab._decision_rule(cells)
    report = {
        "schema_version": v4ab.SCHEMA,
        "commit_sha": "a" * 40,
        "cells": cells,
        "decision_rule": decision,
    }
    text = v4ab.render_step_summary(report)
    assert text.count("| BTCUSDT |") + text.count("| ETHUSDT |") == 18
    assert "| INF |" in text
    assert "| N/A |" in text
    assert "Gross PnL USD" in text and "Net PnL USD" in text
    assert decision["decision_branch"] in text
    with pytest.raises(ValueError, match="v4ab_step_summary_report_schema_invalid"):
        v4ab.render_step_summary({"schema_version": "wrong"})


def test_time_exit_violations_capture_observed_early_exit_reasons() -> None:
    trades = [
        {
            "direction": "LONG",
            "exit_reason": "INITIAL_STOP",
            "exit_trigger_type": "INTRABAR_STOP",
            "time_exit_triggered": False,
            "bars_held": 7,
            "entry_timestamp": "2025-01-01T00:00:00+00:00",
            "exit_timestamp": "2025-01-01T07:00:00+00:00",
        },
        {
            "direction": "LONG",
            "exit_reason": "TIME_EXIT",
            "exit_trigger_type": "TIME_EXIT_OPEN",
            "time_exit_triggered": True,
            "bars_held": 25,
        },
    ]
    violations = v4ab._time_exit_contract_violations(trades, 24)
    assert len(violations) == 1
    details = v4ab._time_exit_violation_details(violations, 24)
    assert details["invalid_trade_count"] == 1
    assert details["exit_reason_counts"] == {"INITIAL_STOP": 1}
    sample = details["sample_invalid_trades"][0]
    assert sample["exit_trigger_type"] == "INTRABAR_STOP"
    assert sample["bars_held"] == 7
    assert sample["expected_bars_held"] == 25


def test_run_cell_skips_only_time_exit_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_trade = {
        "direction": "LONG",
        "exit_reason": "TP_HIT",
        "exit_trigger_type": "INTRABAR_TAKE",
        "time_exit_triggered": False,
        "bars_held": 9,
        "entry_timestamp": "2025-01-01T00:00:00+00:00",
        "exit_timestamp": "2025-01-01T09:00:00+00:00",
    }
    replay = {"trades": [invalid_trade], "metrics": {"max_drawdown_percent": 1.0}}
    monkeypatch.setattr(v4ab, "simulate_signals", lambda *args, **kwargs: replay)
    cell, trade_rows = v4ab._run_cell("BTCUSDT", _bars(), [], 12, 10, 24, "TIME_N")
    assert cell["status"] == "SKIPPED_WARNING"
    assert cell["trade_count"] is None
    assert cell["gross_profit_factor"] is None
    assert cell["warning_code"] == "v4ab_unconditional_time_exit_contract_failed"
    assert cell["warning_details"]["exit_reason_counts"] == {"TP_HIT": 1}
    assert trade_rows == []


def test_run_cell_does_not_suppress_other_policy_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = {
        "trades": [
            {
                "direction": "LONG",
                "gross_market_pnl_usd": 1.0,
                "net_pnl_usd": 0.0,
                "fees_usd": 0.5,
                "slippage_cost_usd": 0.5,
                "bars_held": 25,
                "exit_reason": "HORIZON_EXPIRY",
            }
        ],
        "metrics": {"max_drawdown_percent": 0.0},
    }
    monkeypatch.setattr(v4ab, "simulate_signals", lambda *args, **kwargs: replay)
    with pytest.raises(ValueError, match="v4ab_trade_exceeded_registered_horizon"):
        v4ab._run_cell("BTCUSDT", _bars(), [], 12, 10, 24, "TB_1.5_1.0")


def test_skipped_time_exit_cell_makes_matrix_decision_incomplete() -> None:
    cells = [_cell(0.9, 0.8)]
    skipped = v4ab._skipped_time_exit_cell(
        "ETHUSDT",
        20,
        15,
        24,
        [
            {
                "exit_reason": "INITIAL_STOP",
                "exit_trigger_type": "INTRABAR_STOP",
                "bars_held": 3,
                "entry_timestamp": "2025-01-01T00:00:00+00:00",
                "exit_timestamp": "2025-01-01T03:00:00+00:00",
            }
        ],
    )
    cells.append(skipped)
    result = v4ab._decision_rule(cells)
    assert (
        result["decision_branch"]
        == "INCOMPLETE_MATRIX_DUE_TO_SKIPPED_CELL_NO_FINAL_DECISION"
    )
    assert result["matrix_complete"] is False
    assert result["completed_cell_count"] == 1
    assert result["skipped_cell_ids"] == ["ETHUSDT|1h|24|TIME_N"]


def test_oos_qualifier_remains_identifiable_when_another_cell_is_skipped() -> None:
    qualified = _cell(1.5, 1.35, count=35)
    skipped = v4ab._skipped_time_exit_cell(
        "ETHUSDT",
        20,
        15,
        24,
        [{"exit_reason": "INITIAL_STOP", "bars_held": 2}],
    )
    result = v4ab._decision_rule([qualified, skipped])
    assert (
        result["decision_branch"]
        == "NET_PF_GT_1_3_WITH_AT_LEAST_30_TRADES_FLAG_FOR_INDEPENDENT_OOS"
    )
    assert result["net_oos_confirmation_cell_ids"] == [qualified["candidate_id"]]
    assert result["matrix_complete"] is False


def test_summary_marks_skipped_cell_n_a_and_does_not_call_it_zero_trades() -> None:
    completed = _cell(1.2, 0.8)
    skipped = v4ab._skipped_time_exit_cell(
        "ETHUSDT",
        20,
        15,
        24,
        [{"exit_reason": "INITIAL_STOP", "bars_held": 2}],
    )
    cells = [completed, skipped]
    report = {
        "schema_version": v4ab.SCHEMA,
        "commit_sha": "a" * 40,
        "cells": cells,
        "decision_rule": v4ab._decision_rule(cells),
    }
    summary = v4ab.render_step_summary(report)
    assert "SKIPPED_WARNING" in summary
    assert "| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |" in summary
    assert "N/A is not zero trades" in summary


def test_cell_status_predicates_reject_non_string_values() -> None:
    assert v4ab._cell_is_completed({})
    assert v4ab._cell_is_completed({"status": "COMPLETED"})
    assert not v4ab._cell_is_completed({"status": None})
    assert not v4ab._cell_is_completed({"status": 1})
    assert v4ab._cell_is_skipped({"status": "SKIPPED_WARNING"})
    assert not v4ab._cell_is_skipped({"status": []})
    assert not v4ab._cell_is_skipped({})


def test_csv_writer_handles_rows_and_empty_result(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    path = root / "nested" / "trades.csv"
    v4ab._write_trade_csv(path, [{"trade_index": 1, "net_pnl_usd": 2.0}])
    assert path.read_text(encoding="utf-8").splitlines()[0] == "trade_index,net_pnl_usd"
    empty = root / "empty.csv"
    v4ab._write_trade_csv(empty, [])
    assert empty.read_text(encoding="utf-8") == ""
