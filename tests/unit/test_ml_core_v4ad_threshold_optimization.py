"""Contract and matrix tests for the V4-AD threshold sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4ac_cost_optimization as v4ac
from scripts import run_ml_core_v4ad_threshold_optimization as v4ad


def _synthetic_matrix() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for symbol in v4ad.SYMBOLS:
        for timeframe in v4ad.TIMEFRAMES:
            for policy in v4ad.EXIT_POLICIES_BY_TIMEFRAME[timeframe]:
                for mode in v4ad.MODES:
                    target = (
                        None
                        if mode == "PRIMARY_ONLY"
                        else float(mode.removeprefix("META_TOP_")) / 100.0
                    )
                    candidate_id = v4ad._candidate_id(
                        symbol, timeframe, policy, mode
                    )
                    cell = {
                        "candidate_id": candidate_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "horizon_bars": 24,
                        "exit_policy": policy,
                        "mode": mode,
                        "status": "COMPLETED",
                        "meta_target_accept_fraction": target,
                        "meta_threshold": None if target is None else 0.5,
                        "accepted_signal_count": 35,
                        "trade_count": 31,
                        "gross_profit_factor": 1.8,
                        "gross_profit_factor_infinite_no_losses": False,
                        "net_profit_factor": 1.6,
                        "net_profit_factor_infinite_no_losses": False,
                        "max_drawdown_percent": 5.0,
                        "net_pnl_total_usd": 50.0,
                        "acceptance_target_met": mode != "PRIMARY_ONLY",
                    }
                    cells.append(cell)
                    if mode != "PRIMARY_ONLY":
                        scenarios.append(
                            {
                                "candidate_id": candidate_id,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "exit_policy": policy,
                                "mode": mode,
                                "target_accept_fraction": target,
                                "selected_threshold": 0.5,
                                "status": "TRAINED",
                                "skip_reason": None,
                                "diagnostics": {},
                            }
                        )
    return cells, scenarios


def test_scope_threshold_modes_and_refined_trailing_contract() -> None:
    assert v4ad.TIMEFRAMES == ("1h", "4h")
    assert v4ad.TARGET_ACCEPT_FRACTIONS == (0.30, 0.50, 0.70)
    assert v4ad.MODES == (
        "PRIMARY_ONLY",
        "META_TOP_30",
        "META_TOP_50",
        "META_TOP_70",
    )
    assert v4ad.EXIT_POLICIES_BY_TIMEFRAME["4h"] == (
        "TB_1.5_1.0",
        "TRAIL_2.0",
    )
    assert v4ad.REFINED_TRAIL_POLICY in v4ad.EXIT_POLICIES_BY_TIMEFRAME["1h"]
    for timeframe in ("1h", "4h"):
        v4ad._validate_timeframe(timeframe)
    for timeframe in ("15m", "30m", "1d", "1w"):
        with pytest.raises(ValueError, match="v4ad_timeframe_out_of_scope"):
            v4ad._validate_timeframe(timeframe)

    refined = v4ac._trade_management_config(v4ad.REFINED_TRAIL_POLICY)
    baseline = v4ac._trade_management_config("TRAIL_2.0")
    assert refined.trailing_stop_enabled and baseline.trailing_stop_enabled
    assert refined.stop_loss_atr_multiplier == baseline.stop_loss_atr_multiplier == 2.0
    assert refined.trailing_activation_atr == 1.5
    assert baseline.trailing_activation_atr == 2.0
    assert refined.trailing_step_atr == baseline.trailing_step_atr == 1.0


def test_target_modes_and_acceptance_gate_are_strict() -> None:
    assert [v4ad._mode_for_fraction(value) for value in (0.3, 0.5, 0.7)] == [
        "META_TOP_30",
        "META_TOP_50",
        "META_TOP_70",
    ]
    with pytest.raises(ValueError, match="v4ad_meta_target_accept_fraction_out_of_policy"):
        v4ad._mode_for_fraction(0.6)

    cell = {
        "trade_count": 31,
        "net_profit_factor": 1.5001,
        "net_profit_factor_infinite_no_losses": False,
        "max_drawdown_percent": 15.0,
    }
    assert v4ad._acceptance_target_met(cell)
    assert not v4ad._acceptance_target_met({**cell, "trade_count": 30})
    assert not v4ad._acceptance_target_met({**cell, "net_profit_factor": 1.5})
    assert not v4ad._acceptance_target_met({**cell, "max_drawdown_percent": 15.01})
    assert v4ad._acceptance_target_met(
        {**cell, "net_profit_factor": None, "net_profit_factor_infinite_no_losses": True}
    )


def test_meta_threshold_sweep_fits_one_model_and_uses_calibration_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = DataFrame({"close": np.arange(1_000, dtype=float) + 100.0})
    features = DataFrame({"feature": np.arange(1_000, dtype=float)})
    signals = Series(False, index=range(1_000), dtype=bool)
    training = DataFrame(
        {
            "event_index": [220, 240, 260],
            "label_end_idx": [225, 245, 250],
            "meta_label": [0, 1, 1],
            "primary_signal": ["BUY"] * 3,
            "primary_oof": [True] * 3,
            "feature": [1.0, 2.0, 3.0],
        }
    )
    monkeypatch.setattr(
        v4ac,
        "evaluate_primary_train_sample_gate",
        lambda *args, **kwargs: {"passed": True, "observed_primary_buy_count": 60},
    )
    monkeypatch.setattr(
        v4ac,
        "_meta_training_rows",
        lambda *args, **kwargs: (
            training,
            {
                "labels_matured_before_outer_train_end": True,
                "all_primary_events_are_buy": True,
                "all_primary_events_are_oof": True,
            },
        ),
    )
    monkeypatch.setattr(
        v4ac,
        "split_meta_fit_calibration",
        lambda data, **kwargs: (data.iloc[:2].copy(), data.iloc[2:].copy()),
    )
    fit_calls: list[dict[str, Any]] = []

    def fake_fit(*args: Any, **kwargs: Any) -> tuple[object, dict[str, Any]]:
        fit_calls.append(kwargs)
        return object(), {"model_family": "fake-lightgbm", "calibration_labels_used": False}

    monkeypatch.setattr(v4ac, "fit_lightgbm_meta_filter", fake_fit)
    monkeypatch.setattr(
        v4ac,
        "predict_meta_probabilities",
        lambda model, data: np.asarray([0.1, 0.4, 0.6, 0.9], dtype=float),
    )
    observed_targets: list[float] = []

    def fake_select(
        probabilities: np.ndarray, *, target_buy_fraction: float
    ) -> dict[str, float]:
        observed_targets.append(target_buy_fraction)
        return {"threshold": target_buy_fraction}

    monkeypatch.setattr(v4ad, "select_dynamic_buy_threshold", fake_select)
    model, diagnostics, status = v4ad._train_meta_model_threshold_sweep(
        bars, features, signals, 700, "TB_1.5_1.0"
    )

    assert model is not None and status == "TRAINED"
    assert len(fit_calls) == 1
    assert fit_calls[0]["target_accept_fraction"] == 0.5
    assert observed_targets == [0.3, 0.5, 0.7]
    assert diagnostics["fit_labels_purged_before_calibration"] is True
    assert diagnostics["validation_labels_used_for_fit"] is False
    assert diagnostics["validation_labels_used_for_threshold"] is False
    assert diagnostics["meta_model"]["single_model_reused_across_thresholds"] is True
    assert diagnostics["meta_model"]["thresholds"]["70"]["selected_threshold"] == 0.7


def test_meta_sweep_skips_closed_when_primary_sample_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v4ac,
        "evaluate_primary_train_sample_gate",
        lambda *args, **kwargs: {"passed": False, "observed_primary_buy_count": 49},
    )

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model_fit_must_not_run_before_primary_sample_gate")

    monkeypatch.setattr(v4ac, "_meta_training_rows", unexpected)
    model, diagnostics, status = v4ad._train_meta_model_threshold_sweep(
        DataFrame({"close": [1.0]}),
        DataFrame({"feature": [1.0]}),
        Series([False]),
        1,
        "TB_1.5_1.0",
    )
    assert model is None
    assert diagnostics["sample_gate"]["observed_primary_buy_count"] == 49
    assert status == "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"


def test_symbol_runner_builds_complete_1h_4h_threshold_and_exit_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=1_000, freq="h", tz="UTC"),
            "close": np.linspace(100.0, 200.0, 1_000),
        }
    )
    features = DataFrame({"feature": np.ones(len(bars))})
    signals = Series(False, index=range(len(bars)), dtype=bool)
    signals.iloc[[700, 720]] = True

    monkeypatch.setattr(v4ac, "resample_ohlcv", lambda source, timeframe: bars)
    monkeypatch.setattr(
        v4ac, "_primary_features_and_signals", lambda data: (features, signals)
    )
    monkeypatch.setattr(v4ac, "_outer_boundaries", lambda count: (700, 850))
    monkeypatch.setattr(
        v4ac, "_forward_primary_signal_indices", lambda *args: [700, 720]
    )
    threshold_diagnostics = {
        "meta_model": {
            "thresholds": {
                "30": {"selected_threshold": 0.3},
                "50": {"selected_threshold": 0.5},
                "70": {"selected_threshold": 0.7},
            }
        }
    }
    monkeypatch.setattr(
        v4ad,
        "_train_meta_model_threshold_sweep",
        lambda *args: (object(), threshold_diagnostics, "TRAINED"),
    )
    monkeypatch.setattr(
        v4ac,
        "_forward_meta_probabilities",
        lambda *args: {700: 0.4, 720: 0.8},
    )

    def fake_economic_cell(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        candidate_id = (
            f"{kwargs['symbol']}|{kwargs['timeframe']}|24|"
            f"{kwargs['policy']}|{kwargs['mode']}"
        )
        cell = {
            "candidate_id": candidate_id,
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "exit_policy": kwargs["policy"],
            "mode": kwargs["mode"],
            "status": "COMPLETED",
            "accepted_signal_count": len(kwargs["signal_indices"]),
            "trade_count": 31,
            "gross_profit_factor": 1.8,
            "gross_profit_factor_infinite_no_losses": False,
            "net_profit_factor": 1.6,
            "net_profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": 5.0,
            "net_pnl_total_usd": 50.0,
        }
        return cell, [{"candidate_id": candidate_id, "mode": kwargs["mode"]}]

    monkeypatch.setattr(v4ac, "_run_economic_cell", fake_economic_cell)
    cells, trades, decisions, windows, scenarios = v4ad._run_symbol_timeframe(
        "BTCUSDT", bars
    )
    assert len(cells) == 20
    assert len({item["candidate_id"] for item in cells}) == 20
    assert sum(item["mode"] == "PRIMARY_ONLY" for item in cells) == 5
    assert sum(item["mode"].startswith("META_TOP_") for item in cells) == 15
    assert len(scenarios) == 15
    assert {item["timeframe"] for item in cells} == {"1h", "4h"}
    assert {item["exit_policy"] for item in cells if item["timeframe"] == "4h"} == {
        "TB_1.5_1.0",
        "TRAIL_2.0",
    }
    assert len(windows) == 2
    assert all(item["acceptance_target_met"] for item in cells if item["mode"] != "PRIMARY_ONLY")
    assert len(trades) == 20
    assert decisions


def test_meta_variant_skip_preserves_null_economics_and_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v4ac,
        "_meta_skip_cell",
        lambda **kwargs: {
            "candidate_id": "old-id",
            "mode": "META_FILTERED",
            "status": kwargs["reason"],
            "trade_count": None,
            "net_profit_factor": None,
            "net_pnl_total_usd": None,
        },
    )
    cell, trades, decisions, scenario = v4ad._run_meta_variant(
        symbol="ETHUSDT",
        timeframe="4h",
        policy="TRAIL_2.0",
        bars=DataFrame({"date": [pd.Timestamp("2025-01-01", tz="UTC")]}),
        primary_indices=[10],
        raw_count=2,
        forward_end=20,
        model=None,
        diagnostics={"skip_reason": "insufficient_support"},
        meta_status="SKIPPED_INSUFFICIENT_META_SUPPORT",
        probabilities={},
        target_accept_fraction=0.7,
    )
    assert cell["candidate_id"] == "ETHUSDT|4h|24|TRAIL_2.0|META_TOP_70"
    assert cell["mode"] == "META_TOP_70"
    assert cell["trade_count"] is None and cell["net_profit_factor"] is None
    assert cell["acceptance_target_met"] is False
    assert trades == [] and decisions == []
    assert scenario["status"] == "SKIPPED_INSUFFICIENT_META_SUPPORT"


def test_report_matrix_and_safety_invariants_fail_closed() -> None:
    cells, scenarios = _synthetic_matrix()
    report = v4ad._build_report("a" * 40, 730, [], cells, [], scenarios)
    assert len(report["cells"]) == 40
    assert len(report["meta_model"]["scenarios"]) == 30
    assert report["timeframes"] == ["1h", "4h"]
    assert report["horizon_hours_by_timeframe"] == {"1h": 24.0, "4h": 96.0}
    assert report["meta_model"]["validation_labels_used_for_fit_or_threshold"] is False
    assert report["execution_cost_contract"]["round_trip_cost_bps"] == 30.0
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    assert report["safety"]["private_api_calls"] == 0
    assert report["safety"]["real_order_attempts"] == 0

    with pytest.raises(ValueError, match="v4ad_candidate_matrix_incomplete"):
        v4ad._build_report("a" * 40, 730, [], cells[:-1], [], scenarios)

    invalid = [dict(item) for item in cells]
    invalid[0]["timeframe"] = "1d"
    with pytest.raises(ValueError, match="v4ad_report_timeframe_out_of_policy"):
        v4ad._build_report("a" * 40, 730, [], invalid, [], scenarios)

    with pytest.raises(ValueError, match="v4ad_meta_scenario_matrix_incomplete"):
        v4ad._build_report("a" * 40, 730, [], cells, [], scenarios[:-1])


def test_step_summary_contains_all_rows_targets_metrics_and_safety() -> None:
    cells, scenarios = _synthetic_matrix()
    report = {
        "schema_version": v4ad.SCHEMA,
        "commit_sha": "a" * 40,
        "cells": cells,
        "meta_model": {"scenarios": scenarios},
    }
    summary = v4ad.render_step_summary(report)
    assert summary.count("| BTCUSDT |") + summary.count("| ETHUSDT |") == 40
    assert "Top 30%, Top 50%, and Top 70%" in summary
    assert "same model and calibration probabilities are reused" in summary
    assert "Trades >30, Net PF >1.5, and Max DD <=15%" in summary
    assert "Champion=null" in summary
    assert "trading_authorization=NO-GO" in summary
    with pytest.raises(ValueError, match="v4ad_step_summary_report_schema_invalid"):
        v4ad.render_step_summary({"schema_version": "incorrect"})


def test_run_v4ad_writes_report_and_csv_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cells, scenarios = _synthetic_matrix()

    def fake_runner(
        symbol: str, source_bars: DataFrame
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        return (
            [item for item in cells if item["symbol"] == symbol],
            [],
            [],
            [{"symbol": symbol, "timeframe": timeframe} for timeframe in v4ad.TIMEFRAMES],
            [item for item in scenarios if item["symbol"] == symbol],
        )

    monkeypatch.setattr(
        v4ac,
        "_load_diagnostic_datasets",
        lambda *args: (
            {"BTCUSDT": DataFrame(), "ETHUSDT": DataFrame()},
            [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}],
        ),
    )
    monkeypatch.setattr(v4ad, "_run_symbol_timeframe", fake_runner)
    report_path = tmp_path / "nested" / "report.json"
    trades_path = tmp_path / "trades.csv"
    decisions_path = tmp_path / "decisions.csv"
    report = v4ad.run_v4ad(
        tmp_path,
        report_path,
        trades_path,
        decisions_path,
        days=730,
        commit_sha="b" * 40,
    )
    assert report_path.is_file()
    assert trades_path.is_file() and decisions_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["schema_version"] == v4ad.SCHEMA
    assert len(report["cells"]) == 40
