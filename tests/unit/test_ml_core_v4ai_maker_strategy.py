"""Unit tests for V4-AI maker-fee and ADX-filtered research contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from scripts import run_ml_core_v4ac_cost_optimization as v4ac
from scripts import run_ml_core_v4ai_maker_strategy as v4ai
from services.meta_labeling import v4ah as advanced


def _trade_rows() -> list[dict[str, Any]]:
    return [
        {
            "net_return_fraction": 0.01,
            "gross_market_pnl_usd": 100.0,
            "net_pnl_usd": 50.0,
            "fees_usd": 20.0,
            "slippage_cost_usd": 30.0,
            "direction": "LONG",
            "bars_held": 4,
        },
        {
            "net_return_fraction": -0.005,
            "gross_market_pnl_usd": -50.0,
            "net_pnl_usd": -25.0,
            "fees_usd": 10.0,
            "slippage_cost_usd": 15.0,
            "direction": "LONG",
            "bars_held": 3,
        },
    ]


def _matrix_replay() -> dict[str, Any]:
    return {
        "trades": _trade_rows(),
        "metrics": {"max_drawdown_percent": 1.25},
    }


def _summary_metrics() -> dict[str, Any]:
    return {
        "trade_count": 2,
        "gross_profit_factor": 2.0,
        "gross_profit_factor_infinite_no_losses": False,
        "net_profit_factor": 2.0,
        "net_profit_factor_infinite_no_losses": False,
        "win_rate": 0.5,
        "max_drawdown_percent": 1.25,
        "net_pnl_total_usd": 25.0,
        "gross_pnl_total_usd": 50.0,
        "fees_total_usd": 30.0,
        "slippage_total_usd": 45.0,
    }


def _valid_matrix() -> dict[str, Any]:
    policies = [dict(item) for item in v4ai.EXIT_POLICIES]
    cells = [
        {
            "policy": str(policy["name"]),
            "symbol": "ETHUSDT",
            "timeframe": "1h",
            "fee_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "round_trip_cost_bps": 30.0,
            "diagnostic_only": True,
            "meta_model_used": False,
            "final_holdout_used": False,
        }
        for policy in policies
    ]
    return {
        "evaluation_split": "FORWARD_VALIDATION",
        "final_holdout_used": False,
        "meta_model_used": False,
        "promotion_allowed": False,
        "cost_contract": {
            "fee_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "round_trip_cost_bps": 30.0,
        },
        "policies": policies,
        "cells": cells,
    }


def _valid_report() -> dict[str, Any]:
    return {
        "schema_version": v4ai.SCHEMA,
        "commit_sha": "a" * 40,
        "symbols": ["ETHUSDT"],
        "timeframes": ["1h"],
        "cost_contract": {
            "fee_type_assumption": "MAKER_FEE_PROXY",
            "fee_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "round_trip_cost_bps": 30.0,
            "passive_limit_fill_simulation": False,
        },
        "risk_metric_contract": v4ai._risk_metric_contract(),
        "primary_filter": {
            "feature": v4ai.ADX_FEATURE,
            "period": 14,
            "threshold": 20.0,
            "comparison": "STRICT_GT",
        },
        "final_holdout_used": False,
        "chronological_evaluation": {"final_15_percent_holdout_untouched": True},
        "candidates": [
            {"mode": mode, "sequence_model_used": False}
            for mode in v4ai.CANDIDATE_NAMES
        ],
        "exit_geometry_matrix": _valid_matrix(),
        "meta_model": {
            "enabled": False,
            "separation_diagnostics": {"status": "NOT_AVAILABLE_MODEL_NOT_FIT"},
        },
        "safety": v4ai._safety_report(),
    }


def test_execution_cost_is_locked_to_thirty_bps() -> None:
    execution = v4ai._execution_config()
    assert execution.fee_bps == 10.0
    assert execution.slippage_bps == 5.0
    assert (
        2.0 * (execution.fee_bps + execution.slippage_bps) == v4ai.ROUND_TRIP_COST_BPS
    )
    with pytest.raises(ValueError, match="v4ai_maker_cost_contract_invalid"):
        v4ai._validate_cost_contract(
            type("BadExecution", (), {"fee_bps": 25.0, "slippage_bps": 20.0})()
        )


def test_adx_filter_is_strict_and_rejects_non_finite_rows() -> None:
    features = DataFrame({v4ai.ADX_FEATURE: [19.9, 20.0, 20.1, 31.0, np.nan, np.inf]})
    base = pd.Series([True] * 6, dtype=bool)
    filtered, diagnostics = v4ai._apply_adx_filter(base, features)
    assert filtered.tolist() == [False, False, True, True, False, False]
    assert diagnostics["comparison"] == "STRICT_GT"
    assert diagnostics["base_primary_buy_count"] == 6
    assert diagnostics["valid_adx_rows"] == 4
    assert diagnostics["adx_qualified_rows"] == 2
    assert diagnostics["filtered_primary_buy_count"] == 2
    with pytest.raises(ValueError, match="v4ai_adx_feature_alignment_invalid"):
        v4ai._apply_adx_filter(base, features.drop(columns=[v4ai.ADX_FEATURE]))
    with pytest.raises(ValueError, match="v4ai_primary_signal_invalid"):
        v4ai._apply_adx_filter(
            pd.Series([True, False, None], dtype=object),
            DataFrame({v4ai.ADX_FEATURE: [30.0, 31.0, 32.0]}),
        )


def test_regime_feature_pipeline_uses_fixed_three_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}

    def fake_build_regime_features(
        features: DataFrame,
        *,
        fit_end: int,
        cluster_count: int,
    ) -> tuple[DataFrame, dict[str, Any], object, object]:
        observed["fit_end"] = fit_end
        observed["cluster_count"] = cluster_count
        return (
            features.copy(),
            {
                "fit_rows": fit_end,
                "cluster_count": cluster_count,
                "hard_regime_filter_enabled": False,
            },
            object(),
            object(),
        )

    monkeypatch.setattr(advanced, "build_regime_features", fake_build_regime_features)
    transformed, diagnostics = v4ai._regime_features(DataFrame({"feature": [1.0, 2.0]}))
    assert observed == {"fit_end": 200, "cluster_count": 3}
    assert len(transformed) == 2
    assert diagnostics["cluster_count"] == 3
    assert diagnostics["hard_regime_filter_enabled"] is False


def test_percentile_seventy_threshold_uses_only_probability_values() -> None:
    probabilities = np.linspace(0.1, 0.9, 20)
    threshold, diagnostics = v4ai._select_top_percentile_threshold(probabilities)
    assert threshold == pytest.approx(np.quantile(probabilities, 0.30))
    assert diagnostics["status"] == "SELECTED_TOP_70_PERCENT_ACCEPTANCE"
    assert diagnostics["cutoff_percentile"] == pytest.approx(0.30)
    assert diagnostics["target_accept_fraction"] == pytest.approx(0.70)
    assert diagnostics["labels_used"] is False
    assert diagnostics["selected_rows"] == 14
    assert diagnostics["expected_top_fraction_before_ties"] == pytest.approx(0.70)
    with pytest.raises(ValueError, match="v4ai_threshold_probability_input_invalid"):
        v4ai._select_top_percentile_threshold([0.1, np.nan])


def test_trade_risk_metrics_use_net_closed_trade_returns() -> None:
    values = np.asarray([0.02, -0.01, 0.03, -0.005], dtype=float)
    trades = [{"net_return_fraction": float(value)} for value in values]
    metrics = v4ai._trade_risk_metrics(trades)
    expected_downside = float(
        np.sqrt(np.mean(np.square(np.minimum(values, 0.0))))
    )
    assert metrics["return_observation_count"] == 4
    assert metrics["sharpe_ratio_per_trade"] == pytest.approx(
        np.mean(values) / np.std(values, ddof=1)
    )
    assert metrics["sortino_ratio_per_trade"] == pytest.approx(
        np.mean(values) / expected_downside
    )
    assert np.isfinite(metrics["return_kurtosis_non_excess"])
    assert metrics["sortino_target_return_per_trade"] == 0.0


def test_dsr_is_deflated_within_explicit_run_local_trial_cohort() -> None:
    return_sets = [
        [0.010, -0.010, 0.020, -0.005, 0.015],
        [0.020, -0.015, 0.010, -0.010, 0.005],
        [-0.010, 0.005, 0.012, -0.003, 0.010],
    ]
    metrics_rows = []
    for returns in return_sets:
        metrics = v4ai._trade_risk_metrics(
            [{"net_return_fraction": value} for value in returns]
        )
        metrics["net_profit_factor"] = 1.2
        metrics["net_profit_factor_infinite_no_losses"] = False
        metrics_rows.append(metrics)
    v4ai._attach_dsr_to_cohort(metrics_rows, 'TEST_CURRENT_RUN_COHORT_ONLY')
    assert all(item["dsr_status"] == "COMPUTED" for item in metrics_rows)
    assert all(item["dsr_trial_count"] == 3 for item in metrics_rows)
    assert all(0.0 <= item["dsr_probability"] <= 1.0 for item in metrics_rows)
    assert all(item["zai_net_pf_gt_1_1"] is True for item in metrics_rows)
    assert all(
        item['dsr_trial_scope'] == 'TEST_CURRENT_RUN_COHORT_ONLY'
        for item in metrics_rows
    )


def test_single_and_constant_trade_returns_do_not_invent_risk_ratios() -> None:
    single = v4ai._trade_risk_metrics([{"net_return_fraction": 0.01}])
    assert single["sharpe_ratio_per_trade"] is None
    assert single["sortino_ratio_per_trade"] is None
    constant = v4ai._trade_risk_metrics(
        [{"net_return_fraction": 0.01}] * 4
    )
    assert constant["sharpe_ratio_per_trade"] is None
    assert constant["sharpe_status"] == "ZERO_RETURN_VOLATILITY"
    assert constant["sortino_ratio_per_trade"] is None
    assert constant["sortino_status"] == "ZERO_DOWNSIDE_DEVIATION"


def test_temperature_purge_shortfall_skips_meta_and_preserves_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = DataFrame(
        {
            "event_index": [10, 20, 30],
            "label_end_idx": [11, 21, 31],
            "meta_label": [0, 1, 0],
        }
    )
    training_diagnostics = {"matured_costed_oof_label_count": 3}
    split_diagnostics = {
        "fit_rows": 42,
        "temperature_rows": 18,
        "threshold_rows": 21,
        "split_policy": "CHRONOLOGICAL_70_15_15_WITH_LABEL_END_PURGE",
    }
    monkeypatch.setattr(
        v4ai, "_oof_training_frame",
        lambda *args: (training, training_diagnostics.copy()),
    )

    def fail_temperature_split(*args: Any, **kwargs: Any) -> Any:
        error = ValueError("v4ah_temperature_rows_insufficient_after_purge")
        setattr(error, 'diagnostics', split_diagnostics.copy())
        raise error

    monkeypatch.setattr(
        advanced,
        "split_fit_temperature_threshold",
        fail_temperature_split,
    )
    result = v4ai._fit_meta_model(
        DataFrame({"close": [100.0, 101.0]}),
        DataFrame({"feature": [1.0, 2.0]}),
        pd.Series([True, True], dtype=bool),
        100,
        {"passed": True},
    )
    assert result["skip_reason"] == "v4ah_temperature_rows_insufficient_after_purge"
    assert result["training_diagnostics"]["matured_costed_oof_label_count"] == 3
    assert result["split_diagnostics"]["temperature_rows"] == 18

def test_oof_meta_label_uses_exact_thirty_bps_cost_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    bars = DataFrame({"date": dates})
    monkeypatch.setattr(
        v4ac,
        "_build_forward_actions",
        lambda frame, indices, end: [{"signal": "BUY", "event": indices[0]}],
    )
    seen: list[Any] = []

    def fake_simulate(
        frame: DataFrame,
        actions: list[dict[str, Any]],
        execution: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        seen.append(execution)
        return {
            "trades": [
                {
                    "exit_timestamp": dates[5].isoformat(),
                    "net_pnl_usd": 12.0,
                    "net_return_fraction": 0.012,
                    "exit_reason": "TAKE_PROFIT",
                }
            ]
        }

    monkeypatch.setattr(v4ai, "simulate_signals", fake_simulate)
    result = v4ai._label_one_oof_event(
        {
            "event_index": 1,
            "fold_id": 0,
            "primary_signal": "BUY",
            "primary_oof": True,
        },
        bars,
        {pd.Timestamp(value): index for index, value in enumerate(dates)},
        40,
    )
    assert result is not None
    assert result["meta_label"] == 1
    assert result["net_pnl_usd"] == 12.0
    assert result["label_end_idx"] == 5
    assert seen[0].fee_bps == 10.0
    assert seen[0].slippage_bps == 5.0


def test_oof_training_diagnostics_record_maker_proxy_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    bars = DataFrame({"date": dates})
    events = DataFrame(
        {
            "event_index": [1, 2, 3, 4],
            "fold_id": [0, 0, 1, 1],
            "primary_signal": ["BUY"] * 4,
            "primary_oof": [True] * 4,
        }
    )

    def fake_label(
        event: dict[str, Any],
        train_bars: DataFrame,
        index_by_timestamp: dict[pd.Timestamp, int],
        train_end: int,
    ) -> dict[str, Any]:
        index = int(event["event_index"])
        return {
            "event_index": index,
            "fold_id": int(event["fold_id"]),
            "label_end_idx": index + 1,
            "meta_label": index % 2,
            "net_pnl_usd": 1.0 if index % 2 else -1.0,
            "net_return": 0.001 if index % 2 else -0.001,
            "exit_reason": "TEST",
            "primary_signal": "BUY",
            "primary_oof": True,
        }

    monkeypatch.setattr(v4ai, "_label_one_oof_event", fake_label)
    labelled, diagnostics = v4ai._training_oof_labels(bars, events, 50)
    assert len(labelled) == 4
    assert diagnostics["matured_costed_oof_label_count"] == 4
    assert diagnostics["label_execution_costs"] == {
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
        "round_trip_cost_bps": 30.0,
    }
    assert diagnostics["training_labels_matured_before_validation"] is True


def test_exit_geometry_matrix_has_only_three_thirty_bps_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(
        v4ac,
        "_build_forward_actions",
        lambda bars, indices, end: [{"signal": "BUY", "indices": indices}],
    )

    def fake_simulate(
        bars: DataFrame,
        actions: list[dict[str, Any]],
        execution: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        seen.append((execution, trade_management_config))
        return _matrix_replay()

    monkeypatch.setattr(v4ai, "simulate_signals", fake_simulate)
    monkeypatch.setattr(v4ac, "_summarize_replay", lambda replay: _summary_metrics())
    matrix = v4ai._run_exit_geometry_matrix(
        DataFrame({"close": [100.0, 101.0]}), [10, 20], 100
    )
    assert len(matrix["cells"]) == 3
    assert {cell["round_trip_cost_bps"] for cell in matrix["cells"]} == {30.0}
    assert {call[0].fee_bps for call in seen} == {10.0}
    assert {call[0].slippage_bps for call in seen} == {5.0}
    assert {cell["policy"] for cell in matrix["cells"]} == {
        "V4AI_REFERENCE",
        "V4AE_HISTORICAL_BEST",
        "V4AF_OWNER_GEOMETRY",
    }
    v4ai._validate_exit_matrix(matrix)
    invalid = dict(matrix)
    invalid["cost_contract"] = {
        "fee_bps_per_side": 25.0,
        "slippage_bps_per_side": 20.0,
        "round_trip_cost_bps": 90.0,
    }
    with pytest.raises(ValueError, match="v4ai_exit_matrix_cost_invalid"):
        v4ai._validate_exit_matrix(invalid)


def test_candidate_replay_uses_thirty_bps_and_reports_unscaled_expectancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v4ac,
        "_build_forward_actions",
        lambda bars, indices, end: [{"signal": "BUY", "indices": indices}],
    )
    seen: list[Any] = []

    def fake_simulate(
        bars: DataFrame,
        actions: list[dict[str, Any]],
        execution: Any,
        *,
        trade_management_config: Any,
    ) -> dict[str, Any]:
        seen.append(execution)
        return _matrix_replay()

    monkeypatch.setattr(v4ai, "simulate_signals", fake_simulate)
    monkeypatch.setattr(v4ac, "_summarize_replay", lambda replay: _summary_metrics())
    candidate, trades, decisions = v4ai._run_candidate(
        DataFrame(
            {"date": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")}
        ),
        [10, 20],
        90,
        v4ai.ADX_CONTROL,
        None,
        None,
    )
    assert candidate["metrics"]["trade_count"] == 2
    assert candidate["metrics"][
        "mean_unscaled_net_expectancy_usd_at_full_equity"
    ] == pytest.approx(25.0)
    assert len(trades) == 2
    assert len(decisions) == 2
    assert {item["round_trip_cost_bps"] for item in trades} == {30.0}
    assert seen[0].fee_bps == 10.0 and seen[0].slippage_bps == 5.0


def test_skipped_meta_candidate_and_decisions_are_not_misreported_as_rejects() -> None:
    bars = DataFrame(
        {"date": pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")}
    )
    candidate = v4ai._skipped_meta_candidate(
        "v4ah_binary_class_support_insufficient", 2
    )
    decisions = v4ai._skipped_decision_rows(bars, [5, 8])
    assert candidate["status"] == "SKIPPED_INSUFFICIENT_META_SUPPORT"
    assert all(item["meta_filter_decision"] == "SKIPPED" for item in decisions)
    assert all(item["executable_signal"] == "NOT_EVALUATED" for item in decisions)


def test_report_rejects_any_cost_contract_other_than_thirty_bps() -> None:
    report = _valid_report()
    v4ai._validate_report(report, "a" * 40)
    report["risk_metric_contract"]["annualization"] = "ANNUALIZED"
    with pytest.raises(ValueError, match="v4ai_risk_metric_contract_invalid"):
        v4ai._validate_report(report, "a" * 40)
    report = _valid_report()
    report["cost_contract"]["round_trip_cost_bps"] = 90.0
    with pytest.raises(ValueError, match="v4ai_report_cost_contract_invalid"):
        v4ai._validate_report(report, "a" * 40)


def test_report_rejects_open_safety_state_and_holdout_contamination() -> None:
    report = _valid_report()
    report["safety"]["order_permission"] = True
    with pytest.raises(ValueError, match="v4ai_fail_closed_safety_contract_invalid"):
        v4ai._validate_report(report, "a" * 40)
    report = _valid_report()
    report["final_holdout_used"] = True
    with pytest.raises(ValueError, match="v4ai_final_holdout_contamination"):
        v4ai._validate_report(report, "a" * 40)


def test_report_validates_p70_threshold_scope_when_meta_is_enabled() -> None:
    report = _valid_report()
    report["meta_model"] = {
        "enabled": True,
        "threshold_diagnostics": {
            "status": "SELECTED_TOP_70_PERCENT_ACCEPTANCE",
            "target_accept_fraction": 0.70,
            "cutoff_percentile": 0.30,
            "labels_used": False,
            "selection_scope": "THIRD_PURGED_OOF_TRAINING_SLICE_ONLY",
        },
        "temperature_diagnostics": {"calibration_rows": 38},
        "separation_diagnostics": {
            "metric_scope": "PURGED_OOF_TEMPERATURE_SLICE_OUT_OF_FIT_PREDICTIONS",
            "sample_count": 38,
            "roc_auc": 0.52,
            "pr_auc_average_precision": 0.24,
        },
    }
    v4ai._validate_report(report, "a" * 40)
    report["meta_model"]["threshold_diagnostics"]["labels_used"] = True
    with pytest.raises(
        ValueError, match="v4ai_meta_threshold_or_separation_contract_invalid"
    ):
        v4ai._validate_report(report, "a" * 40)


def test_step_summary_discloses_maker_proxy_and_fill_model_limitation() -> None:
    report = _valid_report()
    report.update(
        {
            "signal_counts": {
                "base_primary_buy_count": 100,
                "adx_filtered_primary_buy_count": 40,
            },
            "meta_model": {
                "enabled": True,
                "training_diagnostics": {"matured_costed_oof_label_count": 20},
                "fit_diagnostics": {"top_k_selected": 30},
                "separation_diagnostics": {
                    "roc_auc": 0.52,
                    "pr_auc_average_precision": 0.24,
                    "positive_count": 4,
                    "sample_count": 38,
                    "positive_support_quality": "LOW_POSITIVE_SUPPORT",
                },
                "threshold_diagnostics": {
                    "threshold": 0.7,
                    "threshold_rows": 20,
                    "selected_fraction": 0.3,
                },
            },
            "exit_geometry_matrix": {
                **_valid_matrix(),
                "cells": [
                    {
                        **cell,
                        "initial_stop_atr": 2.0,
                        "take_profit_atr": 4.0,
                        "trailing_activation_atr": 1.5,
                        "trailing_step_atr": 2.0,
                        "trade_count": 2,
                        "win_rate": 0.5,
                        "net_profit_factor": 2.0,
                        "net_profit_factor_infinite_no_losses": False,
                        "max_drawdown_percent": 1.0,
                        "net_pnl_total_usd": 25.0,
                        "mean_unscaled_net_expectancy_usd_at_full_equity": 25.0,
                    }
                    for cell in _valid_matrix()["cells"]
                ],
            },
            "candidates": [
                {
                    "mode": mode,
                    "status": "COMPLETED",
                    "primary_signal_count": 40,
                    "accepted_signal_count": 12,
                    "acceptance_target_met": False,
                    "metrics": {
                        "trade_count": 12,
                        "win_rate": 0.4,
                        "net_profit_factor": 1.1,
                        "net_profit_factor_infinite_no_losses": False,
                        "max_drawdown_percent": 5.0,
                        "net_pnl_total_usd": 10.0,
                        "mean_unscaled_net_expectancy_usd_at_full_equity": 0.8,
                    },
                }
                for mode in v4ai.CANDIDATE_NAMES
            ],
        }
    )
    summary = v4ai._render_step_summary(report)
    assert "ADX(14) strictly >20" in summary
    assert "30 bps round trip" in summary
    assert "passive Limit queue/partial-fill mechanics are not simulated" in summary
    assert "ROC-AUC=0.52" in summary
    assert "ADX-filtered Primary exit-geometry matrix (30 bps only)" in summary
    assert "DSR Z" in summary
    assert "DSR P" in summary
    assert "Sortino/trade" in summary
    assert "Z.AI diagnostic reference" in summary
    assert "existing Net PF >1.5 acceptance gate remains unchanged" in summary
    assert "historical trials are not included" in summary
