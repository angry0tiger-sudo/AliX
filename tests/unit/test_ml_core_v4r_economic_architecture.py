from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4r_economic_architecture as v4r
from scripts.run_ml_core_real_data_baseline import label_audit_diagnostics
from scripts.run_ml_core_v4q_feature_target_rebuild import (
    _candidate_profit_factor_gates,
)
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    _metrics,
)
from pathlib import Path
from services.ml_core.hyperparameter_tuning import (
    TUNING_OBJECTIVES,
    _select_candidate,
    custom_eval_metric,
    tune_lightgbm_classifier,
)


def test_v4r_factorial_design_uses_two_geometries_and_two_objectives() -> None:
    arms = {item["name"]: item for item in v4r.V4R_ARMS}
    assert len(arms) == 4
    assert arms["V4R_2R_NET_EXPECTANCY"]["take_profit_multiplier"] == 2.0
    assert arms["V4R_2R_NET_EXPECTANCY"]["stop_loss_multiplier"] == 1.0
    assert arms["V4R_2R_NET_EXPECTANCY"]["horizon_candles"] == 24
    assert arms["V4R_2R_NET_EXPECTANCY"]["tuning_objective"] == "NET_EXPECTANCY_PROXY"
    assert arms["CONTROL_SYMMETRIC_BUY_F1"]["tuning_objective"] == "BUY_F1"
    assert "NET_EXPECTANCY_PROXY" in TUNING_OBJECTIVES
    assert v4r.V4R_TIMEFRAMES == ("1h", "4h")


def test_v4r_masks_common_24_candle_tail_for_both_geometry_arms() -> None:
    labels = DataFrame(
        {
            "event_index": list(range(30)),
            "label": ["BUY"] * 30,
            "exit_reason": ["TP_HIT"] * 30,
            "label_end_idx": list(range(1, 31)),
            "unlabeled_reason": [None] * 30,
        }
    )
    masked = v4r._mask_to_common_horizon(labels, 30)
    assert masked["label"].iloc[:6].notna().all()
    assert masked["label"].iloc[6:].isna().all()
    assert masked["unlabeled_reason"].iloc[6:].eq("common_24_candle_tail").all()
    audit = label_audit_diagnostics(masked)
    assert audit["unlabeled_events"] == 24
    assert (
        audit["unlabeled_reason_distribution"]["reasons"]["common_24_candle_tail"][
            "count"
        ]
        == 24
    )


def test_v4r_event_objective_subtracts_90_bps_from_predicted_buy_rows() -> None:
    class FixedPredictions:
        def predict(self, features: DataFrame) -> np.ndarray:
            return np.asarray(["BUY", "SELL", "BUY"])

    features = DataFrame({"x": [0.0, 1.0, 2.0]}, index=[10, 11, 12])
    actual = Series(["BUY", "SELL", "HOLD"], index=features.index)
    gross = Series([0.02, -0.01, 0.04], index=features.index)
    score = custom_eval_metric(
        FixedPredictions(),
        features,
        actual,
        gross_returns_by_index=gross,
        estimated_cost_bps=90.0,
    )
    assert score == pytest.approx(0.042)


def test_v4r_event_economic_outcomes_use_target_exit_references() -> None:
    bars = DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 105.0],
        }
    )
    labels = DataFrame(
        {
            "event_index": [0, 1, 2],
            "label_end_idx": [1, 2, 3],
            "label": ["BUY", "SELL", "HOLD"],
            "take_profit_price": [102.0, 102.0, 102.0],
            "stop_loss_price": [98.0, 99.0, 98.0],
            "exit_reason": ["TP_HIT", "SL_HIT", "HORIZON_EXPIRY"],
        }
    )

    outcomes = v4r._economic_outcomes_by_index(bars, labels)

    assert outcomes.iloc[:3].tolist() == pytest.approx([0.02, -0.01, 0.05])
    assert np.isnan(outcomes.iloc[3])


def test_v4r_arm_aggregate_preserves_trade_and_candidate_summaries() -> None:
    records = [
        {
            "metrics": {
                "economic_data_status": "ELIGIBLE",
                "trade_count": 2,
                "profit_factor_net_of_costs": 1.5,
                "max_drawdown_percent": 4.0,
                "profit_factor_gate_passed": True,
                "economic_candidate_gate_passed": False,
            },
            "trades": [
                {"net_pnl_usd": 10.0, "gross_market_pnl_usd": 12.0},
                {"net_pnl_usd": -4.0, "gross_market_pnl_usd": -3.0},
            ],
            "total_fees_usd": 1.2,
            "total_slippage_cost_usd": 0.8,
        },
        {
            "metrics": {
                "economic_data_status": "INSUFFICIENT_DATA",
                "trade_count": 0,
                "profit_factor_net_of_costs": None,
                "max_drawdown_percent": 0.0,
                "profit_factor_gate_passed": False,
                "economic_candidate_gate_passed": False,
            },
            "trades": [],
            "total_fees_usd": 0.0,
            "total_slippage_cost_usd": 0.0,
        },
    ]

    aggregate = v4r._arm_aggregate(records)

    assert aggregate["candidate_count"] == 2
    assert aggregate["active_candidate_count"] == 1
    assert aggregate["insufficient_data_candidate_count"] == 1
    assert aggregate["trade_count"] == 2
    assert aggregate["winning_trade_count"] == 1
    assert aggregate["losing_trade_count"] == 1
    assert aggregate["combined_net_profit_factor"] == pytest.approx(2.5)
    assert aggregate["combined_gross_profit_factor"] == pytest.approx(4.0)
    assert aggregate["total_gross_market_pnl_usd"] == pytest.approx(9.0)
    assert aggregate["total_fees_usd"] == pytest.approx(1.2)
    assert aggregate["total_slippage_cost_usd"] == pytest.approx(0.8)
    assert aggregate["total_net_pnl_usd"] == pytest.approx(6.0)
    assert aggregate["mean_candidate_profit_factor"] == pytest.approx(1.5)
    assert aggregate["mean_candidate_max_drawdown_percent"] == pytest.approx(2.0)
    assert aggregate["pf_above_1_5_candidate_count"] == 1
    assert aggregate["strict_economic_pass_candidate_count"] == 0


def test_v4r_objective_selects_best_net_expectancy_not_best_buy_f1() -> None:
    results = {
        "mean_test_hold_precision": np.asarray([0.8, 0.5]),
        "mean_test_hold_recall": np.asarray([0.3, 0.3]),
        "mean_test_macro_f1": np.asarray([0.7, 0.6]),
        "mean_test_buy_f1": np.asarray([0.9, 0.4]),
        "mean_test_net_expectancy_proxy": np.asarray([-0.01, 0.02]),
        "params": [
            {"num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.05},
            {"num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.05},
        ],
    }
    index, status, _ = _select_candidate(
        results, minimum_hold_recall=0.2, objective="NET_EXPECTANCY_PROXY"
    )
    assert index == 1
    assert status == "NET_EXPECTANCY_PROXY_MAXIMIZED"


@pytest.mark.parametrize("trade_count", [0, 5, 29])
def test_infinite_pf_below_30_trades_is_insufficient_data(trade_count: int) -> None:
    trades = [
        {"net_pnl_usd": 10.0, "mfe_atr": 0.8, "mae_atr": 0.1}
        for _ in range(trade_count)
    ]
    metrics = _metrics(
        trades, [10_000.0, 10_000.0 + 10.0 * trade_count], ExecutionConfig()
    )
    if trade_count > 0:
        assert metrics["profit_factor_infinite_no_losses"] is True
    assert metrics["economic_data_status"] == "INSUFFICIENT_DATA"
    assert metrics["trade_count_gate_passed"] is False
    assert metrics["profit_factor_gate_passed"] is False
    assert metrics["economic_candidate_gate_passed"] is False
    assert _candidate_profit_factor_gates(metrics)[1:] == (False, False)


def test_infinite_pf_with_30_trades_is_sample_eligible() -> None:
    trades = [{"net_pnl_usd": 10.0, "mfe_atr": 0.8, "mae_atr": 0.1} for _ in range(30)]
    metrics = _metrics(trades, [10_000.0, 10_300.0], ExecutionConfig())
    assert metrics["profit_factor_infinite_no_losses"] is True
    assert metrics["economic_data_status"] == "ELIGIBLE"
    assert metrics["trade_count_gate_passed"] is True
    assert metrics["profit_factor_gate_passed"] is True


def test_v4r_sample_gate_skip_preserves_empty_slot_without_fake_metrics() -> None:
    status = "SKIPPED_INSUFFICIENT_SAMPLES"
    candidate = v4r._sample_gate_skipped_candidate_record(
        arm=v4r.V4R_ARMS[1],
        pair="BTC/USDT",
        timeframe="4h",
        profile="Baseline",
        policy="weight_none",
        experiment={
            "status": status,
            "sample_gate": {
                "status": status,
                "insufficient_validation_classes": ["BUY"],
            },
            "sample_signature_sha256": "shared-sample-signature",
            "train_rows": 120,
            "validation_rows": 30,
        },
    )
    assert candidate["status"] == status
    assert candidate["evaluation_performed"] is False
    assert candidate["metrics"]["trade_count"] == 0
    assert candidate["metrics"]["profit_factor_net_of_costs"] is None
    assert candidate["metrics"]["max_drawdown_percent"] is None
    aggregate = v4r._arm_aggregate([candidate])
    assert aggregate["candidate_count"] == 1
    assert aggregate["trained_candidate_count"] == 0
    assert aggregate["sample_gate_skipped_candidate_count"] == 1
    assert aggregate["trade_count"] == 0
    assert aggregate["mean_candidate_max_drawdown_percent"] is None


def test_v4r_pair_runner_persists_sample_gate_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = DataFrame(
        {
            "open": [100.0] * 120,
            "high": [101.0] * 120,
            "low": [99.0] * 120,
            "close": [100.0] * 120,
            "volume": [1.0] * 120,
        }
    )
    labels = DataFrame(
        {
            "event_index": list(range(120)),
            "label": ["BUY"] * 120,
            "exit_reason": ["TP_HIT"] * 120,
            "label_end_idx": list(range(1, 121)),
            "unlabeled_reason": [None] * 120,
        }
    )
    evaluated: list[str] = []
    monkeypatch.setattr(v4r, "resample_ohlcv", lambda _data, _tf: bars)
    monkeypatch.setattr(
        v4r,
        "build_feature_profiles",
        lambda _bars, profile_names: {
            name: DataFrame({"x": [0.0] * len(bars)}) for name in profile_names
        },
    )
    monkeypatch.setattr(
        v4r, "apply_adaptive_triple_barrier", lambda _bars, **_kwargs: labels.copy()
    )
    monkeypatch.setattr(
        v4r,
        "_economic_outcomes_by_index",
        lambda _bars, _labels: Series(np.zeros(len(bars))),
    )

    def fake_v4h(
        _pair: str,
        _timeframe: str,
        _bars: DataFrame,
        _labels: DataFrame,
        _features: dict[str, DataFrame],
        *,
        scenario: dict[str, Any],
        feature_profile: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        skipped = (
            scenario["name"] == "GEOMETRY_2R_BUY_F1" and feature_profile == "Baseline"
        )
        status = "SKIPPED_INSUFFICIENT_SAMPLES" if skipped else "TRAINED"
        return {
            "status": status,
            "sample_gate": {
                "status": status if skipped else "PASS",
                "insufficient_validation_classes": ["BUY"] if skipped else [],
            },
            "sample_signature_sha256": "same-shared-signature",
            "train_rows": 80,
            "validation_rows": 20,
            "split": {},
            "label_distribution": {},
            "label_audit": {},
            "model_variants": None
            if skipped
            else {
                "weight_none": {"validation_probabilities": []},
                "weight_balanced": {"validation_probabilities": []},
            },
        }

    def fake_evaluate(
        *,
        arm: dict[str, Any],
        pair: str,
        timeframe: str,
        profile: str,
        policy: str,
        experiment: dict[str, Any],
        variant: dict[str, Any],
        bars: DataFrame,
    ) -> dict[str, Any]:
        evaluated.append(policy)
        return {
            "candidate_id": v4r._candidate_identifier(
                arm["name"], pair, timeframe, profile, policy
            ),
            "status": "TRAINED",
            "evaluation_performed": True,
            "arm": arm["name"],
            "geometry": arm["geometry"],
            "objective": arm["objective"],
            "pair": pair,
            "timeframe": timeframe,
            "feature_profile": profile,
            "weighting_policy": policy,
            "sample_signature_sha256": experiment["sample_signature_sha256"],
            "train_rows": experiment["train_rows"],
            "validation_rows": experiment["validation_rows"],
            "tuning_objective": arm["tuning_objective"],
            "selected_parameters": {},
            "tuning_diagnostics": {},
            "feature_importance_gain": [],
            "metrics": {"trade_count": 0, "net_return_percent": 0.0},
            "exit_reason_counts": {},
            "raw_signal_counts": {},
            "confidence_filter": {},
            "total_fees_usd": 0.0,
            "total_slippage_cost_usd": 0.0,
            "trades": [],
        }

    monkeypatch.setattr(v4r, "run_v4h_experiment", fake_v4h)
    monkeypatch.setattr(v4r, "_evaluate_candidate", fake_evaluate)
    experiments, records, _ = v4r._run_pair_timeframe(
        "BTCUSDT",
        "4h",
        DataFrame(),
        calibration_fraction=0.2,
        random_search_iterations=1,
    )
    assert len(experiments) == 8
    assert len(records) == 16
    assert sum(row["status"] == "SKIPPED_INSUFFICIENT_SAMPLES" for row in records) == 2
    assert sum(row["status"] == "TRAINED" for row in records) == 14
    assert len(evaluated) == 14


def test_v4r_paired_comparison_excludes_incomplete_training_groups() -> None:
    records = []
    for pair in ("BTC/USDT", "ETH/USDT"):
        for timeframe in ("1h", "4h"):
            for profile in ("Baseline", "Baseline+Trend"):
                for policy in ("weight_none", "weight_balanced"):
                    skip_group = (
                        pair == "BTC/USDT"
                        and timeframe == "4h"
                        and profile == "Baseline"
                        and policy == "weight_none"
                    )
                    for arm in v4r.V4R_ARMS:
                        skipped = skip_group and arm["name"] == "GEOMETRY_2R_BUY_F1"
                        records.append(
                            {
                                "pair": pair,
                                "timeframe": timeframe,
                                "feature_profile": profile,
                                "weighting_policy": policy,
                                "arm": arm["name"],
                                "status": (
                                    "SKIPPED_INSUFFICIENT_SAMPLES"
                                    if skipped
                                    else "TRAINED"
                                ),
                                "sample_signature_sha256": "shared-signature",
                                "metrics": {
                                    "net_return_percent": None if skipped else 1.0
                                },
                            }
                        )
    result = v4r._paired_comparison(records)
    assert result["candidate_group_count"] == 16
    assert result["paired_group_count"] == 15
    assert result["sample_gate_skipped_group_count"] == 1
    assert result["sample_signature_groups_checked"] == 16


def test_v4r_paired_comparison_rejects_sample_signature_mismatch() -> None:
    records = [
        {
            "pair": "BTC/USDT",
            "timeframe": "1h",
            "feature_profile": "Baseline",
            "weighting_policy": "weight_none",
            "arm": arm["name"],
            "sample_signature_sha256": "different"
            if arm["name"].startswith("V4R_")
            else "same",
            "metrics": {"net_return_percent": 1.0},
        }
        for arm in v4r.V4R_ARMS
    ]
    with pytest.raises(ValueError, match="v4r_paired_sample_signature_mismatch"):
        v4r._paired_comparison(records)


def test_v4r_execution_management_uses_primary_asymmetric_geometry() -> None:
    management = v4r._management_config(v4r.V4R_ARMS[3])
    assert management.take_profit_atr_multiplier == 2.0
    assert management.stop_loss_atr_multiplier == 1.0
    assert management.horizon_candles == 24
    assert management.trailing_stop_enabled is False


def test_v4r_empty_trade_csv_is_still_written_with_header(tmp_path: Path) -> None:
    output = tmp_path / "trades.csv"
    columns = ["candidate_id", "trade_index", "net_pnl_usd"]
    v4r._write_csv(output, [], fieldnames=columns)
    assert output.read_text(encoding="utf-8").strip() == (
        "candidate_id,trade_index,net_pnl_usd"
    )


def test_net_expectancy_objective_fails_closed_without_economic_outcomes() -> None:
    features = DataFrame({"x": [0.0, 1.0, 2.0]})
    labels = Series(["BUY", "SELL", "HOLD"])
    with pytest.raises(ValueError, match="tuning_economic_objective_requires_outcomes"):
        tune_lightgbm_classifier(
            object(),
            features,
            labels,
            [0, 1, 2],
            [0, 1, 2],
            horizon_candles=1,
            class_weight_policy="weight_none",
            minimum_hold_recall=0.2,
            n_iter=1,
            n_splits=1,
            tuning_objective="NET_EXPECTANCY_PROXY",
        )
