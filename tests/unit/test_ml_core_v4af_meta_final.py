"""Contract tests for V4-AF meta-labeling and fixed trailing geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4af_meta_final as v4af


def _bars(rows: int = 140) -> DataFrame:
    close = np.linspace(100.0, 120.0, rows)
    return DataFrame(
        {
            "date": pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="h"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 1_000.0),
        }
    )


def _oof_events(indices: list[int]) -> DataFrame:
    return DataFrame(
        {
            "event_index": indices,
            "fold_id": list(range(len(indices))),
            "primary_signal": ["BUY"] * len(indices),
            "primary_oof": [True] * len(indices),
        }
    )


def _completed_metrics(
    *,
    trade_count: int = 35,
    net_pf: float | None = 1.6,
    drawdown: float = 8.0,
) -> dict[str, Any]:
    return {
        "trade_count": trade_count,
        "gross_profit_factor": 1.9,
        "gross_profit_factor_infinite_no_losses": False,
        "net_profit_factor": net_pf,
        "net_profit_factor_infinite_no_losses": False,
        "max_drawdown_percent": drawdown,
        "net_pnl_total_usd": 250.0,
        "win_rate": 0.55,
    }


def _sample_candidate(
    mode: str,
    fraction: float | None,
) -> dict[str, Any]:
    return {
        "candidate_id": v4af._candidate_id(mode, fraction),
        "symbol": v4af.SYMBOL,
        "timeframe": v4af.TIMEFRAME,
        "mode": mode,
        "status": "COMPLETED",
        "target_accept_fraction": fraction,
        "selected_meta_probability_threshold": 0.5 if fraction is not None else None,
        "threshold_calibration": None,
        "primary_buy_event_count": 20,
        "meta_accepted_buy_count": 10 if fraction is not None else 20,
        "meta_rejected_buy_count": 10 if fraction is not None else 0,
        "achieved_validation_accept_fraction": 0.5,
        "metrics": _completed_metrics(),
        "acceptance_target_met": True,
        "sequence_model_used": False,
    }


def test_run_request_and_fixed_owner_geometry_are_strict() -> None:
    v4af._validate_run_request(730, "a" * 40)
    with pytest.raises(ValueError, match="dataset_window"):
        v4af._validate_run_request(365, "a" * 40)
    with pytest.raises(ValueError, match="commit_sha_invalid"):
        v4af._validate_run_request(730, "not-a-sha")

    management = v4af._trade_management_config(0.6)
    assert v4af.SYMBOL == "ETHUSDT"
    assert v4af.TIMEFRAME == "1h"
    assert v4af.INITIAL_STOP_ATR == 2.0
    assert v4af.TRAILING_ACTIVATION_ATR == 1.5
    assert v4af.TRAILING_STEP_ATR == 1.5
    assert v4af.TAKE_PROFIT_ATR == 4.0
    assert management.trailing_stop_enabled is True
    assert management.stop_loss_atr_multiplier == 2.0
    assert management.trailing_activation_atr == 1.5
    assert management.trailing_step_atr == 1.5
    assert management.take_profit_atr_multiplier == 4.0
    assert management.horizon_candles == 24
    assert management.confidence_threshold == 0.6
    unfiltered_management = v4af._trade_management_config(None)
    assert unfiltered_management.confidence_threshold is None


@pytest.mark.parametrize("threshold", [-0.01, 0.0, 1.01, float("nan"), float("inf")])
def test_invalid_meta_threshold_fails_closed(threshold: float) -> None:
    with pytest.raises(ValueError, match="meta_threshold_invalid"):
        v4af._trade_management_config(threshold)


def test_validation_events_are_unique_matured_and_inside_forward_window() -> None:
    signals = Series(False, index=range(200), dtype=bool)
    signals.iloc[[140, 150, 160, 170, 180]] = True

    selected = v4af._eligible_primary_event_indices(signals, 140, 190)

    assert selected == [140, 150, 160]
    with pytest.raises(ValueError, match="signal_window_invalid"):
        v4af._eligible_primary_event_indices(signals, 190, 140)


def test_oof_labels_use_exact_v4af_geometry_costs_and_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(100)
    events = _oof_events([10, 40, 80])
    observed: list[dict[str, Any]] = []

    monkeypatch.setattr(
        v4af,
        "_build_forward_actions",
        lambda frame, indices, end: [{"event_index": indices[0]}],
    )

    def fake_simulate(
        frame: DataFrame,
        actions: list[dict[str, Any]],
        config: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        event_index = int(actions[0]["event_index"])
        observed.append(
            {
                "event_index": event_index,
                "fee_bps": config.fee_bps,
                "slippage_bps": config.slippage_bps,
                "initial_stop": trade_management_config.stop_loss_atr_multiplier,
                "activation": trade_management_config.trailing_activation_atr,
                "step": trade_management_config.trailing_step_atr,
                "take_profit": trade_management_config.take_profit_atr_multiplier,
                "horizon": trade_management_config.horizon_candles,
                "trailing_enabled": trade_management_config.trailing_stop_enabled,
                "confidence_threshold": trade_management_config.confidence_threshold,
            }
        )
        exit_time = pd.Timestamp(frame["date"].iloc[event_index + 5]).isoformat()
        pnl = 25.0 if event_index == 10 else -10.0
        return {
            "trades": [
                {
                    "exit_timestamp": exit_time,
                    "net_pnl_usd": pnl,
                    "net_return_fraction": pnl / 10_000.0,
                    "exit_reason": "TRAILING_STOP_HIT",
                }
            ]
        }

    monkeypatch.setattr(v4af, "simulate_signals", fake_simulate)

    labelled, diagnostics = v4af._training_oof_labels(bars, events, 100)

    assert labelled["event_index"].tolist() == [10, 40]
    assert labelled["meta_label"].tolist() == [1, 0]
    assert labelled["primary_signal"].eq("BUY").all()
    assert labelled["primary_oof"].all()
    assert int(labelled["label_end_idx"].max()) < 100
    assert diagnostics["immature_oof_event_count"] == 1
    assert diagnostics["training_labels_matured_before_validation"] is True
    assert len(observed) == 2
    for item in observed:
        assert item["confidence_threshold"] is None
        assert item["fee_bps"] == 10.0
        assert item["slippage_bps"] == 5.0
        assert item["initial_stop"] == 2.0
        assert item["activation"] == 1.5
        assert item["step"] == 1.5
        assert item["take_profit"] == 4.0
        assert item["horizon"] == 24
        assert item["trailing_enabled"] is True


def test_oof_label_builder_rejects_non_buy_and_non_oof_events() -> None:
    bars = _bars(100)
    invalid = _oof_events([10])
    invalid.loc[0, "primary_signal"] = "SELL"
    with pytest.raises(ValueError, match="oof_primary_buy_events"):
        v4af._training_oof_labels(bars, invalid, 100)

    invalid = _oof_events([10])
    invalid.loc[0, "primary_oof"] = False
    with pytest.raises(ValueError, match="oof_primary_buy_events"):
        v4af._training_oof_labels(bars, invalid, 100)


def test_oof_label_builder_rejects_multiple_trades_for_one_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(100)
    monkeypatch.setattr(
        v4af,
        "_build_forward_actions",
        lambda frame, indices, end: [{"event_index": indices[0]}],
    )
    monkeypatch.setattr(
        v4af,
        "simulate_signals",
        lambda *args, **kwargs: {"trades": [{}, {}]},
    )
    with pytest.raises(ValueError, match="exactly_one_trade"):
        v4af._training_oof_labels(bars, _oof_events([10]), 100)


def test_oof_label_builder_rejects_naive_exit_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(100)
    monkeypatch.setattr(
        v4af,
        "_build_forward_actions",
        lambda frame, indices, end: [{"event_index": indices[0]}],
    )
    monkeypatch.setattr(
        v4af,
        "simulate_signals",
        lambda *args, **kwargs: {
            "trades": [
                {
                    "exit_timestamp": "2024-01-01T05:00:00",
                    "net_pnl_usd": 1.0,
                    "net_return_fraction": 0.001,
                    "exit_reason": "TP_HIT",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="timestamp_timezone_required"):
        v4af._training_oof_labels(bars, _oof_events([0]), 100)


def test_join_event_features_preserves_exact_event_alignment() -> None:
    features = DataFrame(
        {
            "feature_a": np.arange(10, dtype=float),
            "feature_b": np.linspace(0.0, 1.0, 10),
        }
    )
    labels = DataFrame(
        {
            "event_index": [2, 5],
            "fold_id": [0, 1],
            "label_end_idx": [4, 8],
            "meta_label": [1, 0],
            "primary_signal": ["BUY", "BUY"],
            "primary_oof": [True, True],
        }
    )
    result = v4af._join_event_features(features, labels)
    assert result["event_index"].tolist() == [2, 5]
    assert result["feature_a"].tolist() == [2.0, 5.0]
    assert result["meta_label"].tolist() == [1, 0]


def test_join_event_features_rejects_out_of_range_or_non_finite_rows() -> None:
    features = DataFrame({"feature": [1.0, 2.0]})
    labels = DataFrame({"event_index": [3]})
    with pytest.raises(ValueError, match="event_index_out_of_bounds"):
        v4af._join_event_features(features, labels)

    labels = DataFrame({"event_index": [1]})
    features.loc[1, "feature"] = np.nan
    with pytest.raises(ValueError, match="training_features_invalid"):
        v4af._join_event_features(features, labels)


def test_calibration_thresholds_use_only_order_statistics_and_report_ties() -> None:
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9, 0.9])
    thresholds = v4af._calibrate_thresholds(probabilities)

    assert [item["target_accept_fraction"] for item in thresholds] == [0.3, 0.5, 0.7]
    assert [item["selected_threshold"] for item in thresholds] == [0.8, 0.6, 0.4]
    assert thresholds[0]["calibration_labels_used"] is False
    assert thresholds[0]["tie_expansion_rows"] == 1
    with pytest.raises(ValueError, match="calibration_probabilities_invalid"):
        v4af._calibrate_thresholds(np.array([0.2, np.nan]))


def test_validation_probabilities_are_keyed_to_the_exact_primary_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = DataFrame({"feature": np.arange(8, dtype=float)})
    monkeypatch.setattr(
        v4af,
        "predict_meta_probabilities",
        lambda model, selected: np.array([0.25, 0.75]),
    )
    result = v4af._validation_probability_by_index(features, [2, 6], object())
    assert result == {2: 0.25, 6: 0.75}
    assert v4af._validation_probability_by_index(features, [], object()) == {}


def test_decision_rows_keep_rejections_as_explicit_no_entry() -> None:
    bars = _bars(20)
    rows = v4af._decision_rows(
        bars,
        [1, 2, 3],
        {1: 0.8, 2: 0.4},
        "candidate",
        v4af.MODE,
        0.5,
    )
    assert [item["meta_filter_decision"] for item in rows] == [
        "ACCEPT",
        "REJECT",
        "REJECT",
    ]
    assert [item["executable_signal"] for item in rows] == [
        "BUY",
        "NO_ENTRY",
        "NO_ENTRY",
    ]
    assert rows[0]["split_role"] == "FORWARD_VALIDATION"


@pytest.mark.parametrize(
    ("trades", "profit_factor", "drawdown", "expected"),
    [
        (31, 1.5001, 15.0, True),
        (30, 1.8, 5.0, False),
        (31, 1.5, 5.0, False),
        (31, 1.8, 15.01, False),
        (31, None, 5.0, False),
    ],
)
def test_acceptance_gate_keeps_all_three_thresholds_strict(
    trades: int,
    profit_factor: float | None,
    drawdown: float,
    expected: bool,
) -> None:
    metrics = _completed_metrics(
        trade_count=trades,
        net_pf=profit_factor,
        drawdown=drawdown,
    )
    assert v4af._candidate_acceptance_met(metrics) is expected


def test_economic_candidate_applies_threshold_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(150)
    observed: dict[str, Any] = {}

    def fake_actions(
        frame: DataFrame, indices: list[int], forward_end: int
    ) -> list[dict[str, Any]]:
        observed["accepted_indices"] = indices
        return [{"signal": "BUY", "event_index": index} for index in indices]

    def fake_simulate(
        frame: DataFrame,
        actions: list[dict[str, Any]],
        config: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        observed["management"] = trade_management_config
        observed["config"] = config
        return {"trades": [{"net_pnl_usd": 20.0, "marker": "replayed"}]}

    monkeypatch.setattr(v4af, "_build_forward_actions", fake_actions)
    monkeypatch.setattr(v4af, "simulate_signals", fake_simulate)
    monkeypatch.setattr(v4af, "_summarize_replay", lambda replay: _completed_metrics())

    candidate, trades, decisions = v4af._run_economic_candidate(
        bars=bars,
        event_indices=[10, 11, 12],
        candidate_id="candidate",
        mode=v4af.MODE,
        threshold=0.7,
        target_accept_fraction=0.3,
        probability_by_index={10: 0.4, 11: 0.8, 12: 0.6},
        forward_end=140,
        calibration_diagnostics={"calibration_labels_used": False},
    )

    assert observed["accepted_indices"] == [11]
    assert observed["config"].fee_bps == 10.0
    assert observed["config"].slippage_bps == 5.0
    management = observed["management"]
    assert management.trailing_stop_enabled is True
    assert management.stop_loss_atr_multiplier == 2.0
    assert management.trailing_activation_atr == 1.5
    assert management.trailing_step_atr == 1.5
    assert management.take_profit_atr_multiplier == 4.0
    assert len(trades) == 1
    assert candidate["meta_accepted_buy_count"] == 1
    assert candidate["meta_rejected_buy_count"] == 2
    assert [item["meta_filter_decision"] for item in decisions] == [
        "REJECT",
        "ACCEPT",
        "REJECT",
    ]


def test_unfiltered_control_uses_disabled_confidence_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(150)
    observed: dict[str, Any] = {}

    def fake_actions(
        frame: DataFrame, indices: list[int], forward_end: int
    ) -> list[dict[str, Any]]:
        observed["accepted_indices"] = indices
        return [{"signal": "BUY", "event_index": index} for index in indices]

    def fake_simulate(
        frame: DataFrame,
        actions: list[dict[str, Any]],
        config: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        observed["confidence_threshold"] = trade_management_config.confidence_threshold
        return {"trades": [{"net_pnl_usd": 20.0} for _ in actions]}

    monkeypatch.setattr(v4af, "_build_forward_actions", fake_actions)
    monkeypatch.setattr(v4af, "simulate_signals", fake_simulate)
    monkeypatch.setattr(v4af, "_summarize_replay", lambda replay: _completed_metrics())

    candidate, trades, decisions = v4af._run_unfiltered_control(
        bars,
        [10, 20],
        140,
    )

    assert observed["accepted_indices"] == [10, 20]
    assert observed["confidence_threshold"] is None
    assert candidate["selected_meta_probability_threshold"] is None
    assert candidate["meta_accepted_buy_count"] == 2
    assert candidate["meta_rejected_buy_count"] == 0
    assert len(trades) == 2
    assert [item["meta_filter_decision"] for item in decisions] == [
        "ACCEPT",
        "ACCEPT",
    ]


def test_control_and_meta_candidate_ids_are_unambiguous() -> None:
    control = v4af._candidate_id(v4af.CONTROL_MODE)
    meta = v4af._candidate_id(v4af.MODE, 0.5)
    assert "UNFILTERED_CONTROL" in control
    assert "META_TOP_50" in meta
    with pytest.raises(ValueError, match="fraction_required"):
        v4af._candidate_id(v4af.MODE)


def test_skipped_meta_candidate_is_not_misreported_as_a_completed_fit() -> None:
    skipped = v4af._skipped_meta_candidate(0.7, "insufficient_oof_rows")
    assert skipped["status"] == "SKIPPED_INSUFFICIENT_META_SUPPORT"
    assert skipped["acceptance_target_met"] is False
    assert skipped["selected_meta_probability_threshold"] is None
    assert skipped["sequence_model_used"] is False


def test_report_validates_four_rows_scope_holdout_and_safety() -> None:
    candidates = [
        _sample_candidate(v4af.CONTROL_MODE, None),
        _sample_candidate(v4af.MODE, 0.3),
        _sample_candidate(v4af.MODE, 0.5),
        _sample_candidate(v4af.MODE, 0.7),
    ]
    report = v4af._build_report(
        "a" * 40,
        {"symbol": "ETHUSDT", "row_count": 1_051_200},
        _bars(20),
        700,
        850,
        {"train_primary_buy_count": 60},
        {"passed": True},
        {"matured_costed_oof_label_count": 50},
        {"fit_rows": 40, "calibration_rows": 6},
        [{"target_accept_fraction": value} for value in (0.3, 0.5, 0.7)],
        True,
        candidates,
        [],
        [],
    )
    assert report["meta_model"]["enabled"] is True
    assert report["mode"] == v4af.MODE
    assert report["final_holdout_used"] is False
    assert report["chronological_evaluation"]["final_holdout_used"] is False
    assert report["safety"]["Champion"] is None
    assert report["safety"]["trading_authorization"] == "NO-GO"
    summary = v4af._render_step_summary(report)
    assert "UNFILTERED_CONTROL" in summary
    assert "50.0" in summary
    assert "1.2477" in summary
    assert "final 15% untouched" in summary
    assert "Champion=null" in summary
    assert "trading_authorization=NO-GO" in summary


def test_report_rejects_holdout_or_safety_contamination() -> None:
    candidates = [
        _sample_candidate(v4af.CONTROL_MODE, None),
        _sample_candidate(v4af.MODE, 0.3),
        _sample_candidate(v4af.MODE, 0.5),
        _sample_candidate(v4af.MODE, 0.7),
    ]
    report = v4af._build_report(
        "b" * 40,
        {"symbol": "ETHUSDT"},
        _bars(20),
        700,
        850,
        {},
        {"passed": True},
        {},
        {},
        [],
        False,
        candidates,
        [],
        [],
    )
    report["chronological_evaluation"]["final_holdout_used"] = True
    with pytest.raises(ValueError, match="final_holdout"):
        v4af._validate_report(report, "b" * 40)

    report["chronological_evaluation"]["final_holdout_used"] = False
    report["safety"]["order_permission"] = True
    with pytest.raises(ValueError, match="safety_contract"):
        v4af._validate_report(report, "b" * 40)


def test_csv_writer_persists_decisions_and_trade_provenance(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trades.csv"
    v4af._write_csv(
        path,
        [{"candidate_id": "one", "net_pnl_usd": 12.5}],
        ("candidate_id", "net_pnl_usd", "mode"),
    )
    assert path.is_file()
    frame = pd.read_csv(path)
    assert frame.loc[0, "candidate_id"] == "one"
    assert frame.loc[0, "net_pnl_usd"] == 12.5
    assert pd.isna(frame.loc[0, "mode"])


def test_v4ae_reference_distinguishes_owner_baseline_from_historical_best() -> None:
    selected = cast(dict[str, object], v4af.V4AE_REFERENCE["owner_selected_geometry"])
    historical = cast(dict[str, object], v4af.V4AE_REFERENCE["historical_best_cell"])
    assert selected["trailing_step_atr"] == 1.5
    assert selected["net_profit_factor"] == 1.1538
    assert historical["trailing_step_atr"] == 2.0
    assert historical["net_profit_factor"] == 1.2477
