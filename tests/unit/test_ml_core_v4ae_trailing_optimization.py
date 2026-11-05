"""Contract and matrix tests for the V4-AE primary-only trailing grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4ae_trailing_optimization as v4ae


def _synthetic_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for symbol in v4ae.SYMBOLS:
        for config in v4ae.GRID:
            policy = v4ae._policy_name(config)
            cells.append(
                {
                    "candidate_id": v4ae._candidate_id(symbol, config),
                    "symbol": symbol,
                    "timeframe": "1h",
                    "mode": "PRIMARY_ONLY",
                    "status": "COMPLETED",
                    "exit_policy": policy,
                    "initial_stop_atr": config.initial_stop_atr,
                    "trailing_activation_atr": config.trailing_activation_atr,
                    "trailing_step_atr": config.trailing_step_atr,
                    "trade_count": 35,
                    "gross_profit_factor": 1.8,
                    "gross_profit_factor_infinite_no_losses": False,
                    "net_profit_factor": 1.6,
                    "net_profit_factor_infinite_no_losses": False,
                    "max_drawdown_percent": 5.0,
                    "net_pnl_total_usd": 50.0,
                    "acceptance_target_met": True,
                }
            )
    return cells


def test_grid_is_exact_27_configurations_per_symbol_and_contains_v4ad_baseline() -> (
    None
):
    assert len(v4ae.GRID) == 27
    assert len({v4ae._policy_name(item) for item in v4ae.GRID}) == 27
    assert {item.initial_stop_atr for item in v4ae.GRID} == {1.0, 1.5, 2.0}
    assert {item.trailing_activation_atr for item in v4ae.GRID} == {0.5, 1.0, 1.5}
    assert {item.trailing_step_atr for item in v4ae.GRID} == {1.0, 1.5, 2.0}
    assert v4ae.TAKE_PROFIT_ATR == 4.0
    baseline = v4ae.TrailingGridConfig(2.0, 1.5, 1.0)
    assert baseline in v4ae.GRID
    assert v4ae._policy_name(baseline) == "TRAIL_SL2.0_ACT1.5_STEP1.0"


def test_trade_management_config_maps_all_three_grid_axes() -> None:
    selected = v4ae.TrailingGridConfig(1.5, 1.0, 2.0)
    management = v4ae._trade_management_config(selected)
    assert management.trailing_stop_enabled is True
    assert management.stop_loss_atr_multiplier == 1.5
    assert management.trailing_activation_atr == 1.0
    assert management.trailing_step_atr == 2.0
    assert management.take_profit_atr_multiplier == 4.0
    assert management.horizon_candles == 24


@pytest.mark.parametrize(
    "config",
    [
        v4ae.TrailingGridConfig(0.0, 1.0, 1.0),
        v4ae.TrailingGridConfig(1.0, 2.0, 1.0),
        v4ae.TrailingGridConfig(1.0, 1.0, 0.5),
        v4ae.TrailingGridConfig(float("nan"), 1.0, 1.0),
    ],
)
def test_invalid_trailing_grid_configuration_fails_closed(
    config: v4ae.TrailingGridConfig,
) -> None:
    with pytest.raises(ValueError):
        v4ae._validate_grid_config(config)


def test_acceptance_target_keeps_trade_pf_and_drawdown_constraints_strict() -> None:
    cell = {
        "trade_count": 31,
        "net_profit_factor": 1.5001,
        "net_profit_factor_infinite_no_losses": False,
        "max_drawdown_percent": 15.0,
    }
    assert v4ae._candidate_acceptance_met(cell)
    assert not v4ae._candidate_acceptance_met({**cell, "trade_count": 30})
    assert not v4ae._candidate_acceptance_met({**cell, "net_profit_factor": 1.5})
    assert not v4ae._candidate_acceptance_met({**cell, "max_drawdown_percent": 15.01})
    assert v4ae._candidate_acceptance_met(
        {
            **cell,
            "net_profit_factor": None,
            "net_profit_factor_infinite_no_losses": True,
        }
    )


def test_symbol_runner_executes_all_grid_rows_as_primary_only(
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
    monkeypatch.setattr(v4ae.v4ac, "resample_ohlcv", lambda source, timeframe: bars)
    monkeypatch.setattr(
        v4ae.v4ac,
        "_primary_features_and_signals",
        lambda source: (features, signals),
    )
    monkeypatch.setattr(v4ae.v4ac, "_outer_boundaries", lambda count: (700, 850))
    monkeypatch.setattr(
        v4ae.v4ac, "_forward_primary_signal_indices", lambda *args: [700, 720]
    )
    observed: list[str] = []

    def fake_candidate(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        config = kwargs["config"]
        candidate_id = v4ae._candidate_id(kwargs["symbol"], config)
        observed.append(candidate_id)
        return (
            {
                "candidate_id": candidate_id,
                "symbol": kwargs["symbol"],
                "timeframe": "1h",
                "mode": "PRIMARY_ONLY",
                "status": "COMPLETED",
                "initial_stop_atr": config.initial_stop_atr,
                "trailing_activation_atr": config.trailing_activation_atr,
                "trailing_step_atr": config.trailing_step_atr,
            },
            [],
        )

    monkeypatch.setattr(v4ae, "_run_candidate", fake_candidate)
    cells, trades, window = v4ae._run_symbol("BTCUSDT", bars)
    assert len(cells) == 27
    assert len(observed) == 27
    assert len(set(observed)) == 27
    assert trades == []
    assert window["eligible_forward_primary_signal_count"] == 2
    assert all(item["mode"] == "PRIMARY_ONLY" for item in cells)


def test_candidate_matrix_rejects_missing_or_out_of_scope_rows() -> None:
    cells = _synthetic_cells()
    v4ae._validate_candidate_matrix(cells)
    with pytest.raises(ValueError, match="v4ae_candidate_matrix_incomplete"):
        v4ae._validate_candidate_matrix(cells[:-1])
    invalid = [dict(item) for item in cells]
    invalid[0]["timeframe"] = "4h"
    with pytest.raises(ValueError, match="v4ae_candidate_scope_invalid"):
        v4ae._validate_candidate_matrix(invalid)


def test_report_and_step_summary_preserve_scope_costs_baseline_and_safety() -> None:
    cells = _synthetic_cells()
    report = v4ae._build_report(
        "a" * 40,
        730,
        [],
        cells,
        [
            {"symbol": "BTCUSDT", "timeframe": "1h"},
            {"symbol": "ETHUSDT", "timeframe": "1h"},
        ],
    )
    assert report["schema_version"] == v4ae.SCHEMA
    assert report["mode"] == "PRIMARY_ONLY"
    assert report["timeframes"] == ["1h"]
    assert report["grid"]["total_candidate_configurations"] == 54
    assert report["meta_model"]["enabled"] is False
    assert report["meta_model"]["model_fit_performed"] is False
    assert report["execution_cost_contract"]["round_trip_cost_bps"] == 30.0
    assert report["v4ad_reference"]["trade_count"] == 42
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    assert report["safety"]["private_api_calls"] == 0
    assert report["safety"]["real_order_attempts"] == 0
    summary = v4ae.render_step_summary(report)
    assert summary.count("| BTCUSDT |") + summary.count("| ETHUSDT |") == 54
    assert "Trades >30, Net PF >1.5, and Max DD <=15%" in summary
    assert "trailing_step_atr" in summary
    assert "Champion=null" in summary
    assert "trading_authorization=NO-GO" in summary


def test_run_v4ae_writes_report_and_trade_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cells = _synthetic_cells()

    def fake_runner(
        symbol: str, source_bars: DataFrame
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        return (
            [item for item in cells if item["symbol"] == symbol],
            [],
            {"symbol": symbol, "timeframe": "1h"},
        )

    monkeypatch.setattr(
        v4ae.v4ac,
        "_load_diagnostic_datasets",
        lambda *args: (
            {"BTCUSDT": DataFrame(), "ETHUSDT": DataFrame()},
            [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}],
        ),
    )
    monkeypatch.setattr(v4ae, "_run_symbol", fake_runner)
    report_path = tmp_path / "nested" / "report.json"
    trades_path = tmp_path / "trades.csv"
    report = v4ae.run_v4ae(
        tmp_path,
        report_path,
        trades_path,
        days=730,
        commit_sha="b" * 40,
    )
    assert report_path.is_file()
    assert trades_path.is_file()
    assert (
        json.loads(report_path.read_text(encoding="utf-8"))["schema_version"]
        == v4ae.SCHEMA
    )
    assert len(report["cells"]) == 54


def test_summary_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="v4ae_step_summary_report_schema_invalid"):
        v4ae.render_step_summary({"schema_version": "wrong"})
