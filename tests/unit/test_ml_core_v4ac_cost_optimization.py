"""Unit contracts for V4-AC timeframe, costs, exits, and Meta comparison."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4ac_cost_optimization as v4ac
from scripts.run_ml_core_real_data_baseline import SYMBOLS, TIMEFRAME_MINUTES


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


def _trade(
    *,
    gross: float = 100.0,
    net: float = 70.0,
    bars_held: int = 24,
    exit_timestamp: str = "2024-01-02T00:00:00+00:00",
) -> dict[str, Any]:
    return {
        "direction": "LONG",
        "gross_market_pnl_usd": gross,
        "net_pnl_usd": net,
        "fees_usd": 20.0,
        "slippage_cost_usd": 10.0,
        "bars_held": bars_held,
        "exit_reason": "HORIZON_EXPIRY",
        "exit_timestamp": exit_timestamp,
        "net_return_fraction": net / 10_000.0,
    }


def _replay(trades: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"trades": trades or [], "metrics": {"max_drawdown_percent": 4.25}}


def _cell(symbol: str, timeframe: str, policy: str, mode: str) -> dict[str, Any]:
    return {
        "candidate_id": f"{symbol}|{timeframe}|24|{policy}|{mode}",
        "symbol": symbol,
        "pair": f"{symbol.removesuffix('USDT')}/USDT",
        "timeframe": timeframe,
        "horizon_bars": 24,
        "exit_policy": policy,
        "mode": mode,
        "status": "COMPLETED",
        "primary_signal_count": 12,
        "eligible_primary_signal_count": 10,
        "accepted_signal_count": 8,
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
        "round_trip_cost_bps": 30.0,
        "meta_status": "TRAINED" if mode == "META_FILTERED" else None,
        "meta_threshold": 0.6 if mode == "META_FILTERED" else None,
        "trade_count": 8,
        "gross_profit_factor": 1.5,
        "gross_profit_factor_infinite_no_losses": False,
        "net_profit_factor": 1.2,
        "net_profit_factor_infinite_no_losses": False,
        "win_rate": 0.5,
        "median_bars_held": 10.0,
        "max_drawdown_percent": 4.0,
        "gross_pnl_total_usd": 100.0,
        "net_pnl_total_usd": 60.0,
        "fees_total_usd": 20.0,
        "slippage_total_usd": 10.0,
        "meta_diagnostics": {},
    }


def _matrix() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = [
        _cell(symbol, timeframe, policy, mode)
        for symbol, timeframe, policy, mode in itertools.product(
            SYMBOLS, v4ac.TIMEFRAMES, v4ac.EXIT_POLICIES, v4ac.MODES
        )
    ]
    scenarios = [
        {
            "candidate_id": item["candidate_id"],
            "symbol": item["symbol"],
            "timeframe": item["timeframe"],
            "exit_policy": item["exit_policy"],
            "status": "TRAINED",
            "skip_reason": None,
            "diagnostics": {},
        }
        for item in cells
        if item["mode"] == "META_FILTERED"
    ]
    return cells, scenarios


def test_allowed_timeframes_are_exact_and_closed_above_four_hours() -> None:
    for timeframe in ("15m", "30m", "1h", "4h"):
        v4ac._validate_timeframe(timeframe)
    for timeframe in ("5m", "1d", "1w", "12h", "2d"):
        with pytest.raises(ValueError, match="v4ac_timeframe_out_of_policy"):
            v4ac._validate_timeframe(timeframe)
    assert v4ac.TIMEFRAMES == ("15m", "30m", "1h", "4h")
    assert [TIMEFRAME_MINUTES[item] for item in v4ac.TIMEFRAMES] == [
        15,
        30,
        60,
        240,
    ]


def test_request_and_outer_split_contracts_fail_closed() -> None:
    v4ac._validate_run_request(730, "a" * 40)
    with pytest.raises(ValueError, match="v4ac_dataset_window_must_be_730_days"):
        v4ac._validate_run_request(365, "a" * 40)
    with pytest.raises(ValueError, match="v4ac_commit_sha_invalid"):
        v4ac._validate_run_request(730, "bad")
    assert v4ac._outer_boundaries(1_000) == (700, 850)
    with pytest.raises(ValueError, match="v4ac_resampled_dataset_too_short"):
        v4ac._outer_boundaries(999)


def test_fixed_horizon_costs_and_execution_contract() -> None:
    assert v4ac.HORIZON_BARS == 24
    assert v4ac.FEE_BPS_PER_SIDE == 10.0
    assert v4ac.SLIPPAGE_BPS_PER_SIDE == 5.0
    assert v4ac.ROUND_TRIP_COST_BPS == 2 * (10.0 + 5.0)
    assert v4ac._execution_config().fee_bps == 10.0
    assert v4ac._execution_config().slippage_bps == 5.0
    assert v4ac._execution_config().maximum_drawdown == v4ac.MAXIMUM_DRAWDOWN


def test_trade_management_policies_and_unknown_exit_rejection() -> None:
    fixed = v4ac._trade_management_config("TB_1.5_1.0")
    assert (fixed.take_profit_atr_multiplier, fixed.stop_loss_atr_multiplier) == (
        1.5,
        1.0,
    )
    assert fixed.horizon_candles == 24 and not fixed.trailing_stop_enabled
    trailing = v4ac._trade_management_config("TRAIL_2.0")
    assert (trailing.stop_loss_atr_multiplier, trailing.take_profit_atr_multiplier) == (
        2.0,
        4.0,
    )
    assert trailing.trailing_stop_enabled
    assert (trailing.trailing_activation_atr, trailing.trailing_step_atr) == (2.0, 1.0)
    with pytest.raises(ValueError, match="v4ac_exit_policy_out_of_policy"):
        v4ac._trade_management_config("TIME_N")


def test_primary_feature_builder_rejects_missing_profile_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    monkeypatch.setattr(v4ac, "build_feature_profiles", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="v4ac_primary_feature_profile_missing"):
        v4ac._primary_features_and_signals(bars)
    profile = DataFrame({"x": np.ones(len(bars))})
    monkeypatch.setattr(
        v4ac, "build_feature_profiles", lambda *args, **kwargs: {"Combined": profile}
    )
    monkeypatch.setattr(
        v4ac,
        "build_v4y_primary_features",
        lambda data, frame: (
            DataFrame({"x": np.ones(len(data))}),
            Series([False], dtype=bool),
        ),
    )
    with pytest.raises(ValueError, match="v4ac_primary_signal_alignment_invalid"):
        v4ac._primary_features_and_signals(bars)
    monkeypatch.setattr(
        v4ac,
        "build_v4y_primary_features",
        lambda data, frame: (
            DataFrame({"x": np.ones(len(data))}),
            Series(False, index=range(len(data)), dtype=bool),
        ),
    )
    features, signals = v4ac._primary_features_and_signals(bars)
    assert len(features) == len(signals) == len(bars)


def test_signal_records_and_forward_action_guards() -> None:
    bars = _bars()
    action = v4ac._signal_record(bars, 10, "BUY")
    assert action["signal_timestamp"] == bars["date"].iloc[10].isoformat()
    assert action["execution_timestamp"] == bars["date"].iloc[11].isoformat()
    actions = v4ac._build_forward_actions(bars, [10, 20], 100)
    assert [item["signal"] for item in actions] == ["BUY", "BUY", "NO_ENTRY"]
    assert actions[-1]["row_index"] == 98
    with pytest.raises(
        ValueError, match="v4ac_primary_events_must_be_unique_and_ordered"
    ):
        v4ac._build_forward_actions(bars, [20, 10], 100)
    with pytest.raises(ValueError, match="v4ac_primary_event_outside_forward_window"):
        v4ac._build_forward_actions(bars, [99], 100)
    with pytest.raises(ValueError, match="v4ac_forward_sentinel_collides"):
        v4ac._build_forward_actions(bars, [98], 100)
    with pytest.raises(ValueError, match="v4ac_signal_execution_index_invalid"):
        v4ac._signal_record(bars, len(bars) - 1, "BUY")
    naive = bars.copy()
    naive["date"] = pd.date_range("2024-01-01", periods=len(naive), freq="h")
    with pytest.raises(ValueError, match="v4ac_signal_timestamp_timezone_required"):
        v4ac._signal_record(naive, 10, "BUY")
    with pytest.raises(ValueError, match="v4ac_forward_action_window_invalid"):
        v4ac._build_forward_actions(bars, [], len(bars) + 1)


def test_forward_signal_maturity_cutoff_excludes_last_26_bars() -> None:
    signal = Series(False, index=range(1_000), dtype=bool)
    signal.iloc[[700, 824, 825, 849]] = True
    assert v4ac._forward_primary_signal_indices(signal, 700, 850) == [700, 824]
    with pytest.raises(ValueError, match="v4ac_primary_signal_window_invalid"):
        v4ac._forward_primary_signal_indices(signal.iloc[:100], 700, 850)
    with pytest.raises(ValueError, match="v4ac_forward_window_has_no_matured_events"):
        v4ac._forward_primary_signal_indices(signal, 840, 850)


def test_profit_factor_and_replay_summary_calculate_gross_and_net_metrics() -> None:
    assert v4ac._profit_factor([10.0, -5.0]) == (2.0, False)
    assert v4ac._profit_factor([10.0, 2.0]) == (None, True)
    assert v4ac._profit_factor([]) == (None, False)
    replay = _replay(
        [
            _trade(gross=120.0, net=30.0, bars_held=10),
            _trade(gross=-60.0, net=-150.0, bars_held=20),
        ]
    )
    metrics = v4ac._summarize_replay(replay)
    assert metrics["gross_profit_factor"] == 2.0
    assert metrics["net_profit_factor"] == 0.2
    assert metrics["trade_count"] == 2
    assert metrics["median_bars_held"] == 15.0
    assert metrics["gross_pnl_total_usd"] == 60.0
    assert metrics["net_pnl_total_usd"] == -120.0
    assert metrics["win_rate"] == 0.5
    assert metrics["max_drawdown_percent"] == 4.25


def test_replay_summary_rejects_non_spot_nan_and_invalid_drawdown() -> None:
    with pytest.raises(ValueError, match="v4ac_non_spot_trade_detected"):
        v4ac._summarize_replay(_replay([_trade() | {"direction": "SHORT"}]))
    with pytest.raises(ValueError, match="v4ac_trade_pnl_non_finite"):
        v4ac._summarize_replay(
            _replay([_trade() | {"gross_market_pnl_usd": float("nan")}])
        )
    with pytest.raises(ValueError, match="v4ac_candidate_drawdown_invalid"):
        v4ac._summarize_replay(
            {"trades": [], "metrics": {"max_drawdown_percent": -1.0}}
        )


def test_empty_metrics_are_null_not_fake_zeroes() -> None:
    values = v4ac._empty_metric_values()
    assert values["trade_count"] is None
    assert values["gross_profit_factor"] is None
    assert values["net_profit_factor"] is None
    assert values["net_pnl_total_usd"] is None


def test_single_training_event_uses_policy_replay_and_maps_exit_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(150)
    captured: dict[str, Any] = {}

    def fake_simulate(
        data: DataFrame,
        actions: list[dict[str, Any]],
        config: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        captured["rows"] = len(data)
        captured["actions"] = actions
        captured["policy"] = trade_management_config
        exit_time = pd.Timestamp(data["date"].iloc[30]).isoformat()
        return _replay([_trade(exit_timestamp=exit_time)])

    monkeypatch.setattr(v4ac, "simulate_signals", fake_simulate)
    trade, exit_index = v4ac._single_event_trade(bars, 100, "TRAIL_2.0")
    assert exit_index == 100 - v4ac.META_TRAINING_CONTEXT_BARS + 30
    assert captured["rows"] == 26 + v4ac.META_TRAINING_CONTEXT_BARS
    assert [item["signal"] for item in captured["actions"]] == ["BUY", "NO_ENTRY"]
    assert captured["policy"].trailing_stop_enabled
    assert trade["net_pnl_usd"] == 70.0


def test_single_training_event_rejects_unmatured_or_missing_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(100)
    with pytest.raises(ValueError, match="v4ac_training_event_did_not_mature"):
        v4ac._single_event_trade(bars, 80, "TB_1.5_1.0")
    monkeypatch.setattr(v4ac, "simulate_signals", lambda *args, **kwargs: _replay())
    with pytest.raises(
        ValueError, match="v4ac_meta_training_event_trade_count_invalid"
    ):
        v4ac._single_event_trade(bars, 50, "TB_1.5_1.0")


def test_meta_training_rows_build_policy_matched_oof_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    features = DataFrame({"feature": np.arange(len(bars), dtype=float)})
    signals = Series(False, index=range(len(bars)), dtype=bool)
    events = DataFrame(
        {
            "event_index": [250, 275, 674, 675],
            "fold_id": [0, 1, 3, 3],
            "primary_signal": ["BUY"] * 4,
            "primary_oof": [True] * 4,
        }
    )
    monkeypatch.setattr(
        v4ac, "generate_v4y_oof_primary_events", lambda *args, **kwargs: events
    )
    observed: list[int] = []

    def fake_trade(
        data: DataFrame, event_index: int, policy: str
    ) -> tuple[dict[str, Any], int]:
        observed.append(event_index)
        return _trade(net=50.0 if policy == "TB_1.5_1.0" else -10.0), event_index + 5

    monkeypatch.setattr(v4ac, "_single_event_trade", fake_trade)
    training, diagnostics = v4ac._meta_training_rows(
        bars, features, signals, 700, "TRAIL_2.0"
    )
    assert observed == [250, 275, 674]
    assert training["meta_label"].tolist() == [0, 0, 0]
    assert training["primary_oof"].all()
    assert training["label_end_idx"].max() < 700
    assert diagnostics["label_exit_policy"] == "TRAIL_2.0"
    assert diagnostics["labels_matured_before_outer_train_end"] is True


def test_meta_training_rows_and_error_classifier_handle_empty_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    features = DataFrame({"feature": np.ones(len(bars))})
    signals = Series(False, index=range(len(bars)), dtype=bool)
    empty = DataFrame(
        columns=["event_index", "fold_id", "primary_signal", "primary_oof"]
    )
    monkeypatch.setattr(
        v4ac, "generate_v4y_oof_primary_events", lambda *args, **kwargs: empty
    )
    with pytest.raises(ValueError, match="v4ac_meta_training_events_empty"):
        v4ac._meta_training_rows(bars, features, signals, 700, "TB_1.5_1.0")
    assert v4ac._is_expected_meta_support_error(
        ValueError("v4x_insufficient_meta_class_support")
    )
    assert v4ac._is_expected_meta_support_error(
        ValueError("v4ac_meta_training_events_empty")
    )
    assert not v4ac._is_expected_meta_support_error(
        ValueError("v4x_meta_features_non_finite")
    )


def test_meta_model_sample_gate_skips_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    features = DataFrame({"feature": np.ones(len(bars))})
    signals = Series(False, index=range(len(bars)), dtype=bool)
    monkeypatch.setattr(
        v4ac,
        "evaluate_primary_train_sample_gate",
        lambda *args, **kwargs: {"passed": False, "observed_primary_buy_count": 49},
    )

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("meta_training_must_not_run_before_sample_gate")

    monkeypatch.setattr(v4ac, "_meta_training_rows", unexpected)
    model, diagnostics, status = v4ac._train_meta_model(
        bars, features, signals, 700, "TB_1.5_1.0"
    )
    assert model is None
    assert diagnostics["sample_gate"]["observed_primary_buy_count"] == 49
    assert status == "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"


def test_meta_model_fit_records_purge_and_validation_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    features = DataFrame({"feature": np.ones(len(bars))})
    signals = Series(False, index=range(len(bars)), dtype=bool)
    training = DataFrame(
        {
            "event_index": [220, 240, 260],
            "fold_id": [0, 1, 2],
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
        lambda *args, **kwargs: (training, {"label_exit_policy": "TB_1.5_1.0"}),
    )
    monkeypatch.setattr(
        v4ac,
        "split_meta_fit_calibration",
        lambda data, **kwargs: (data.iloc[:2].copy(), data.iloc[2:].copy()),
    )
    monkeypatch.setattr(
        v4ac,
        "fit_lightgbm_meta_filter",
        lambda *args, **kwargs: (
            object(),
            {"selected_threshold": 0.62, "calibration_labels_used": False},
        ),
    )
    model, diagnostics, status = v4ac._train_meta_model(
        bars, features, signals, 700, "TB_1.5_1.0"
    )
    assert model is not None and status == "TRAINED"
    assert diagnostics["fit_rows"] == 2 and diagnostics["calibration_rows"] == 1
    assert diagnostics["fit_labels_purged_before_calibration"] is True
    assert diagnostics["validation_labels_used_for_fit"] is False
    assert diagnostics["meta_model"]["selected_threshold"] == 0.62


def test_meta_model_skips_expected_support_but_propagates_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    features = DataFrame({"feature": np.ones(len(bars))})
    signals = Series(False, index=range(len(bars)), dtype=bool)
    monkeypatch.setattr(
        v4ac,
        "evaluate_primary_train_sample_gate",
        lambda *args, **kwargs: {"passed": True, "observed_primary_buy_count": 60},
    )
    monkeypatch.setattr(
        v4ac,
        "_meta_training_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("v4x_insufficient_meta_class_support")
        ),
    )
    model, diagnostics, status = v4ac._train_meta_model(
        bars, features, signals, 700, "TRAIL_2.0"
    )
    assert model is None
    assert diagnostics["skip_reason"] == "v4x_insufficient_meta_class_support"
    assert status == "SKIPPED_INSUFFICIENT_META_SUPPORT"
    monkeypatch.setattr(
        v4ac,
        "_meta_training_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("v4x_meta_features_non_finite")
        ),
    )
    with pytest.raises(ValueError, match="v4x_meta_features_non_finite"):
        v4ac._train_meta_model(bars, features, signals, 700, "TRAIL_2.0")


def test_forward_probabilities_align_and_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = DataFrame({"x": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(
        v4ac, "predict_meta_probabilities", lambda model, data: np.array([0.25, 0.75])
    )
    assert v4ac._forward_meta_probabilities(features, [0, 2], object()) == {
        0: 0.25,
        2: 0.75,
    }
    assert v4ac._forward_meta_probabilities(features, [], object()) == {}
    monkeypatch.setattr(
        v4ac, "predict_meta_probabilities", lambda model, data: np.array([0.2])
    )
    with pytest.raises(ValueError, match="v4ac_meta_probability_alignment_invalid"):
        v4ac._forward_meta_probabilities(features, [0, 2], object())
    monkeypatch.setattr(
        v4ac, "predict_meta_probabilities", lambda model, data: np.array([float("nan")])
    )
    with pytest.raises(ValueError, match="v4ac_meta_probability_values_invalid"):
        v4ac._forward_meta_probabilities(features, [0], object())


def test_economic_cell_runs_primary_and_preserves_null_meta_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    monkeypatch.setattr(
        v4ac, "simulate_signals", lambda *args, **kwargs: _replay([_trade()])
    )
    primary, rows = v4ac._run_economic_cell(
        symbol="BTCUSDT",
        timeframe="1h",
        policy="TB_1.5_1.0",
        mode="PRIMARY_ONLY",
        bars=bars,
        signal_indices=[10],
        raw_signal_count=10,
        eligible_signal_count=8,
        forward_end=850,
    )
    assert primary["status"] == "COMPLETED" and primary["trade_count"] == 1
    assert primary["fee_bps_per_side"] == 10.0 and len(rows) == 1
    skipped, skipped_rows = v4ac._run_economic_cell(
        symbol="BTCUSDT",
        timeframe="1h",
        policy="TRAIL_2.0",
        mode="META_FILTERED",
        bars=bars,
        signal_indices=[],
        raw_signal_count=10,
        eligible_signal_count=8,
        forward_end=850,
        meta_status="SKIPPED_INSUFFICIENT_META_SUPPORT",
    )
    assert skipped["trade_count"] is None and skipped["net_profit_factor"] is None
    assert skipped_rows == []
    with pytest.raises(ValueError, match="v4ac_mode_out_of_policy"):
        v4ac._run_economic_cell(
            symbol="BTCUSDT",
            timeframe="1h",
            policy="TRAIL_2.0",
            mode="OTHER",
            bars=bars,
            signal_indices=[],
            raw_signal_count=0,
            eligible_signal_count=0,
            forward_end=850,
        )


def test_meta_skip_cell_has_null_unavailable_economics() -> None:
    cell = v4ac._meta_skip_cell(
        symbol="ETHUSDT",
        timeframe="4h",
        policy="TRAIL_2.0",
        raw_signal_count=4,
        eligible_signal_count=3,
        reason="SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS",
        diagnostics={"skip_reason": "minimum_primary_buy_count_not_met"},
    )
    assert cell["status"] == "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"
    assert cell["trade_count"] is None and cell["gross_profit_factor"] is None
    assert cell["net_pnl_total_usd"] is None


def test_symbol_runner_preserves_both_modes_across_all_timeframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars()
    signals = Series(False, index=range(len(bars)), dtype=bool)
    signals.iloc[[700, 720]] = True
    features = DataFrame({"x": np.ones(len(bars))})
    monkeypatch.setattr(v4ac, "resample_ohlcv", lambda source, timeframe: bars)
    monkeypatch.setattr(
        v4ac, "_primary_features_and_signals", lambda data: (features, signals)
    )
    monkeypatch.setattr(v4ac, "_outer_boundaries", lambda count: (700, 850))
    monkeypatch.setattr(
        v4ac, "_forward_primary_signal_indices", lambda *args: [700, 720]
    )
    monkeypatch.setattr(
        v4ac,
        "_run_economic_cell",
        lambda **kwargs: (
            {
                "candidate_id": f"{kwargs['symbol']}|{kwargs['timeframe']}|24|{kwargs['policy']}|{kwargs['mode']}",
                "mode": kwargs["mode"],
                "status": "COMPLETED",
            },
            [],
        ),
    )
    monkeypatch.setattr(
        v4ac,
        "_train_meta_model",
        lambda *args: (
            None,
            {"skip_reason": "insufficient"},
            "SKIPPED_INSUFFICIENT_META_SUPPORT",
        ),
    )
    cells, trades, decisions, windows = v4ac._run_symbol_timeframe("BTCUSDT", bars)
    assert len(cells) == 16
    assert sum(item["mode"] == "PRIMARY_ONLY" for item in cells) == 8
    assert sum(item["mode"] == "META_FILTERED" for item in cells) == 8
    assert len(windows) == 4 and trades == [] and decisions == []


def test_build_report_requires_exact_32_cells_and_sets_non_go_safety() -> None:
    cells, scenarios = _matrix()
    report = v4ac._build_report("a" * 40, 730, [], cells, [], scenarios)
    assert len(report["cells"]) == 32
    assert report["horizon_bars"] == 24
    assert report["horizon_hours_by_timeframe"] == {
        "15m": 6.0,
        "30m": 12.0,
        "1h": 24.0,
        "4h": 96.0,
    }
    assert report["execution_cost_contract"]["round_trip_cost_bps"] == 30.0
    assert report["meta_model"]["enabled"] is True
    assert report["primary_model"]["signal_changed_from_v4ab"] is False
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    with pytest.raises(ValueError, match="v4ac_candidate_matrix_incomplete"):
        v4ac._build_report("a" * 40, 730, [], cells[:-1], [], scenarios)
    bad = [dict(item) for item in cells]
    bad[0]["timeframe"] = "1d"
    with pytest.raises(ValueError, match="v4ac_report_timeframe_out_of_policy"):
        v4ac._build_report("a" * 40, 730, [], bad, [], scenarios)
    with pytest.raises(ValueError, match="v4ac_meta_scenario_matrix_incomplete"):
        v4ac._build_report("a" * 40, 730, [], cells, [], scenarios[:-1])


def test_step_summary_contains_32_rows_modes_costs_and_safety() -> None:
    cells, scenarios = _matrix()
    cells[-1]["status"] = "SKIPPED_INSUFFICIENT_META_SUPPORT"
    cells[-1]["trade_count"] = None
    cells[-1]["gross_profit_factor"] = None
    cells[-1]["net_profit_factor"] = None
    cells[-1]["net_pnl_total_usd"] = None
    report = {
        "schema_version": v4ac.SCHEMA,
        "commit_sha": "a" * 40,
        "cells": cells,
        "meta_scenarios": scenarios,
    }
    rendered = v4ac.render_step_summary(report)
    assert rendered.count("| BTCUSDT |") + rendered.count("| ETHUSDT |") == 32
    assert "10 bps maker-fee assumption" in rendered
    assert "30 bps round trip" in rendered
    assert "PRIMARY_ONLY" in rendered and "META_FILTERED" in rendered
    assert "Gross PF" in rendered and "Net PF" in rendered
    assert "N/A metrics" in rendered
    assert "trading_authorization=NO-GO" in rendered
    with pytest.raises(ValueError, match="v4ac_step_summary_report_schema_invalid"):
        v4ac.render_step_summary({"schema_version": "bad"})


def test_csv_writer_and_pf_formatter_cover_branches(tmp_path: Path) -> None:
    assert v4ac._format_pf(None, True) == "INF"
    assert v4ac._format_pf(None, False) == "N/A"
    assert v4ac._format_pf(1.23456, False) == "1.2346"
    output = tmp_path / "nested" / "trades.csv"
    v4ac._write_csv(output, [{"trade_index": 1, "net_pnl_usd": 2.0}])
    assert (
        output.read_text(encoding="utf-8").splitlines()[0] == "trade_index,net_pnl_usd"
    )
    empty = tmp_path / "empty.csv"
    v4ac._write_csv(empty, [])
    assert empty.read_text(encoding="utf-8") == ""


def test_run_v4ac_writes_full_matrix_and_artifact_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cells, scenarios = _matrix()
    windows = [
        {"symbol": "BTCUSDT", "timeframe": timeframe} for timeframe in v4ac.TIMEFRAMES
    ]

    def fake_run_symbol(
        symbol: str,
        source_bars: DataFrame,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        selected = [dict(item) for item in cells if item["symbol"] == symbol]
        for item in selected:
            item["meta_status"] = "TRAINED" if item["mode"] == "META_FILTERED" else None
            item["meta_diagnostics"] = {}
        return selected, [], [], [dict(item) for item in windows]

    monkeypatch.setattr(
        v4ac,
        "_load_diagnostic_datasets",
        lambda *args: (
            {"BTCUSDT": _bars(), "ETHUSDT": _bars()},
            [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}],
        ),
    )
    monkeypatch.setattr(v4ac, "_run_symbol_timeframe", fake_run_symbol)
    report_path = tmp_path / "report.json"
    trades_path = tmp_path / "trades.csv"
    decisions_path = tmp_path / "decisions.csv"
    report = v4ac.run_v4ac(
        tmp_path,
        report_path,
        trades_path,
        decisions_path,
        days=730,
        commit_sha="a" * 40,
    )
    assert len(report["cells"]) == 32
    assert report_path.is_file()
    assert trades_path.read_text(encoding="utf-8") == ""
    assert decisions_path.read_text(encoding="utf-8") == ""
    assert report["safety"]["Champion"] is None
