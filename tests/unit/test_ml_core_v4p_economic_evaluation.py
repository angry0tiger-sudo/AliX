from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas import DataFrame

import scripts.run_ml_core_v4p_feature_target_rebuild as v4p


def _market_frame(rows: int = 120) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="15min")
    close = pd.Series(range(rows), dtype=float) * 0.01 + 100.0
    return DataFrame(
        {
            "date": dates,
            "open": close - 0.01,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": close + 100.0,
        }
    )


def _variant() -> dict[str, Any]:
    return {
        "hold_metrics": {"precision": 0.45, "recall": 0.30},
        "validation_probabilities": [
            {
                "row_index": 90,
                "timestamp": "2026-01-01T22:30:00+00:00",
                "actual_label": "BUY",
                "probabilities": {"BUY": 0.70, "SELL": 0.10, "HOLD": 0.20},
            }
        ],
        "hyperparameter_tuning": {
            "selected_parameters": {
                "num_leaves": 15,
                "learning_rate": 0.05,
                "min_child_samples": 20,
            },
            "candidate_fit_count": 27,
        },
    }


def _experiment(pair: str, timeframe: str, profile: str) -> dict[str, Any]:
    classes = {
        "BUY": {"count": 15, "percent": 50.0},
        "SELL": {"count": 10, "percent": 33.3333},
        "HOLD": {"count": 5, "percent": 16.6667},
    }
    distribution = {"classes": classes}
    return {
        "status": "TRAINED",
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "feature_count": 13 if profile == "Baseline" else 21,
        "train_rows": 70,
        "validation_rows": 1,
        "label_distribution": {"train": distribution, "validation": distribution},
        "model_variants": {
            "weight_none": _variant(),
            "weight_balanced": _variant(),
        },
    }


def _replay() -> dict[str, Any]:
    return {
        "metrics": {
            "trade_count": 1,
            "profit_factor_net_of_costs": 1.25,
            "profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": 4.0,
            "win_rate": 1.0,
            "net_return_percent": 0.5,
        },
        "exit_reason_counts": {"TP_HIT": 1},
        "raw_signal_counts": {"BUY": 1, "SELL": 0, "HOLD": 0},
        "confidence_filter": {
            "accepted_buy_signal_count": 1,
            "rejected_buy_signal_count": 0,
        },
        "total_fees_usd": 5.0,
        "total_slippage_cost_usd": 4.0,
        "trades": [
            {
                "entry_timestamp": "2026-01-01T22:45:00+00:00",
                "exit_timestamp": "2026-01-01T23:00:00+00:00",
                "entry_fill_price": 100.0,
                "atr_value": 0.5,
                "bars_held": 1,
                "net_pnl_usd": 10.0,
                "mfe_atr": 1.0,
                "mae_atr": 0.1,
                "exit_reason": "TP_HIT",
            }
        ],
        "trade_management": {
            "take_profit_atr_multiplier": 1.0,
            "stop_loss_atr_multiplier": 1.0,
            "horizon_candles": 12,
        },
    }


def _economics_record(
    candidate_id: str,
    pnl: float,
    profit_factor: float,
    drawdown: float,
    exit_reason: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "pair": "BTC/USDT",
        "timeframe": "15m",
        "feature_profile": "Baseline",
        "weighting_policy": "weight_none",
        "feature_count": 13,
        "train_rows": 70,
        "validation_rows": 30,
        "confidence_threshold": 0.60,
        "hold_probability_threshold": 0.60,
        "trade_count": 1,
        "metrics": {
            "trade_count": 1,
            "profit_factor_net_of_costs": profit_factor,
            "profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": drawdown,
            "win_rate": 1.0 if pnl > 0 else 0.0,
            "net_return_percent": pnl / 100.0,
        },
        "exit_reason_counts": {exit_reason: 1},
        "raw_signal_counts": {"BUY": 1, "SELL": 0, "HOLD": 0},
        "confidence_filter": {
            "accepted_buy_signal_count": 1,
            "rejected_buy_signal_count": 0,
        },
        "total_fees_usd": 1.0,
        "total_slippage_cost_usd": 0.5,
        "trades": [
            {
                "entry_timestamp": "2026-01-01T01:00:00+00:00",
                "exit_timestamp": "2026-01-01T01:15:00+00:00",
                "entry_fill_price": 100.0,
                "atr_value": 0.5,
                "bars_held": 1,
                "net_pnl_usd": pnl,
                "mfe_atr": 1.0,
                "mae_atr": 0.1,
                "exit_reason": exit_reason,
            }
        ],
    }


def _report_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for context in range(8):
        for profile in v4p.V4P_FEATURE_PROFILES:
            experiments.append(_experiment("BTC/USDT", "15m", profile))
    return experiments


def _report_records() -> list[dict[str, Any]]:
    return [
        _economics_record(f"candidate-{index}", 10.0, 2.0, 5.0, "TP_HIT")
        for index in range(v4p.V4P_EXPECTED_MODELS)
    ]


def test_v4p_target_and_execution_contract_are_fixed_to_symmetric_one_atr() -> None:
    assert v4p.V4P_SCENARIO == {
        "name": "V4P_1ATR_SYMMETRIC",
        "horizon_candles": 12,
        "take_profit_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
    }
    execution = v4p._execution_config()
    management = v4p._trade_management_config()
    assert execution.initial_equity == 10_000.0
    assert execution.fee_bps == 25.0
    assert execution.slippage_bps == 20.0
    assert execution.position_fraction == 1.0
    assert execution.maximum_drawdown == 0.15
    assert management.atr_period == 14
    assert management.take_profit_atr_multiplier == 1.0
    assert management.stop_loss_atr_multiplier == 1.0
    assert management.horizon_candles == 12
    assert management.confidence_threshold == 0.60
    assert management.trailing_stop_enabled is False


def test_v4p_candidate_replay_uses_fixed_confidence_and_outer_validation_only(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate(
        candidate: dict[str, Any],
        bars: DataFrame,
        config: Any,
        management: Any,
    ) -> dict[str, Any]:
        captured["candidate"] = candidate
        captured["config"] = config
        captured["management"] = management
        return _replay()

    monkeypatch.setattr(v4p, "evaluate_candidate", fake_evaluate)
    experiment = {"feature_count": 13, "train_rows": 70, "validation_rows": 1}
    record = v4p._evaluate_model_candidate(
        "BTC/USDT",
        "15m",
        _market_frame(),
        "Baseline",
        "weight_none",
        experiment,
        _variant(),
    )

    assert record["candidate_id"] == (
        "V4P_1ATR_SYMMETRIC|BTC/USDT|15m|Baseline|weight_none"
    )
    assert captured["candidate"]["selected_hold_threshold"] == 0.60
    assert captured["candidate"]["forward_row_count"] == 1
    assert captured["management"].confidence_threshold == 0.60
    assert captured["management"].take_profit_atr_multiplier == 1.0
    assert captured["management"].stop_loss_atr_multiplier == 1.0
    assert record["trade_count"] == 1
    assert record["exit_reason_counts"] == {"TP_HIT": 1}
    assert len(v4p._trade_csv_rows([record])) == 1
    assert v4p._trade_csv_rows([record])[0]["mfe_atr"] == 1.0


def test_v4p_feature_and_target_orchestration_trains_all_32_model_variants(
    monkeypatch: Any,
) -> None:
    bars = _market_frame()
    datasets = {
        "BTCUSDT": DataFrame({"minute": [1]}),
        "ETHUSDT": DataFrame({"minute": [1]}),
    }
    feature_calls: list[tuple[str, ...]] = []
    target_calls: list[dict[str, Any]] = []
    train_calls: list[dict[str, Any]] = []
    replay_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(v4p, "resample_ohlcv", lambda candles, timeframe: bars)

    def fake_profiles(
        prepared: DataFrame, *, profile_names: tuple[str, ...] | None = None
    ) -> dict[str, DataFrame]:
        assert prepared is bars
        assert profile_names == v4p.V4P_FEATURE_PROFILES
        feature_calls.append(profile_names)
        return {
            "Baseline": DataFrame({"base": range(len(bars))}),
            "Baseline+Trend": DataFrame(
                {"base": range(len(bars)), "trend": range(len(bars))}
            ),
        }

    def fake_targets(prepared: DataFrame, **kwargs: Any) -> DataFrame:
        assert prepared is bars
        target_calls.append(kwargs)
        return DataFrame({"label": ["BUY"] * len(bars)})

    def fake_train(
        pair: str,
        timeframe: str,
        prepared_bars: DataFrame,
        labels: DataFrame,
        matrices: dict[str, DataFrame],
        **kwargs: Any,
    ) -> dict[str, Any]:
        train_calls.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "profile": kwargs["feature_profile"],
                "scenario": kwargs["scenario"],
            }
        )
        return _experiment(pair, timeframe, kwargs["feature_profile"])

    def fake_eval(
        candidate: dict[str, Any],
        prepared_bars: DataFrame,
        config: Any,
        management: Any,
    ) -> dict[str, Any]:
        replay_calls.append(candidate)
        return _replay()

    monkeypatch.setattr(v4p, "build_feature_profiles", fake_profiles)
    monkeypatch.setattr(v4p, "apply_adaptive_triple_barrier", fake_targets)
    monkeypatch.setattr(v4p, "run_v4h_experiment", fake_train)
    monkeypatch.setattr(v4p, "evaluate_candidate", fake_eval)

    experiments, records = v4p.run_v4p_experiments(datasets)

    assert len(experiments) == 16
    assert len(records) == 32
    assert len(feature_calls) == 8
    assert len(target_calls) == 8
    assert len(train_calls) == 16
    assert len(replay_calls) == 32
    assert all(
        call
        == {
            "horizon_candles": 12,
            "take_profit_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
        }
        for call in target_calls
    )
    assert all(call["scenario"] == v4p.V4P_SCENARIO for call in train_calls)
    assert all(
        "validation_probabilities" not in variant
        for experiment in experiments
        for variant in experiment["model_variants"].values()
    )
    assert all(
        record["confidence_threshold"] == 0.60
        and record["hold_probability_threshold"] == 0.60
        for record in records
    )


def test_v4p_aggregate_reports_pooled_pf_candidate_dd_and_exit_distribution() -> None:
    aggregate = v4p._aggregate_economics(
        [
            _economics_record("a", 10.0, 2.0, 5.0, "TP_HIT"),
            _economics_record("b", -5.0, 0.5, 20.0, "SL_HIT"),
        ]
    )
    assert aggregate["candidate_model_count"] == 2
    assert aggregate["total_trade_count"] == 2
    assert aggregate["combined_ledger_profit_factor"] == 2.0
    assert aggregate["take_profit_exit_count"] == 1
    assert aggregate["stop_loss_exit_count"] == 1
    assert aggregate["mean_max_drawdown_percent_active_candidates"] == 12.5
    assert aggregate["preliminary_pf_above_1_and_dd_below_15_candidate_count"] == 1
    assert aggregate["strict_pf_above_1_5_and_dd_below_15_candidate_count"] == 1


def test_v4p_diagnostic_writes_json_comparison_csv_and_trade_csv(
    tmp_path: Path, monkeypatch: Any
) -> None:
    experiments = _report_experiments()
    records = _report_records()
    provenance = [
        {
            "symbol": symbol,
            "requested_days": 730,
            "row_count": 730 * 24 * 60,
            "missing_minute_intervals": 0,
            "public_api_only": True,
            "credentials_used": False,
        }
        for symbol in ("BTCUSDT", "ETHUSDT")
    ]
    monkeypatch.setattr(
        v4p,
        "_load_diagnostic_datasets",
        lambda input_dir, days, commit_sha: (
            {"BTCUSDT": _market_frame(), "ETHUSDT": _market_frame()},
            provenance,
        ),
    )
    monkeypatch.setattr(
        v4p,
        "run_v4p_experiments",
        lambda datasets, random_search_iterations: (experiments, records),
    )
    summary_path = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    report_path = tmp_path / "report.json"
    comparison_path = tmp_path / "comparison.csv"
    trades_path = tmp_path / "trades.csv"

    report = v4p.run_v4p_diagnostic(
        tmp_path,
        report_path,
        comparison_path,
        trades_path,
        days=730,
        commit_sha="a" * 40,
    )

    assert report["schema_version"] == v4p.V4P_REPORT_SCHEMA
    assert report["model_training_performed"] is True
    assert report["economic_evaluation_performed"] is True
    assert report["trained_models"] == 32
    assert report["economic_evaluation"]["candidate_count"] == 32
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    assert json.loads(report_path.read_text())["commit_sha"] == "a" * 40
    assert comparison_path.is_file()
    assert trades_path.is_file()
    assert "AliX V4-P Economic Evaluation" in summary_path.read_text()
    assert len(v4p._comparison_csv_rows(records)) == 32
    assert len(v4p._candidate_summaries(records)) == 32
    assert v4p._pooled_label_distribution(experiments, "validation")["total"] == 240


def test_v4p_rejects_non_730_day_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v4p_requires_exactly_730_days"):
        v4p.run_v4p_diagnostic(
            tmp_path,
            tmp_path / "report.json",
            tmp_path / "comparison.csv",
            tmp_path / "trades.csv",
            days=90,
            commit_sha="b" * 40,
        )


def test_v4p_random_search_iteration_and_dataset_contracts_fail_closed() -> None:
    data = {"BTCUSDT": _market_frame(), "ETHUSDT": _market_frame()}
    for invalid in (0, 28):
        with pytest.raises(ValueError, match="v4p_random_search_iterations_invalid"):
            v4p.run_v4p_experiments(data, random_search_iterations=invalid)
    with pytest.raises(ValueError, match="v4p_dataset_symbol_contract_mismatch"):
        v4p.run_v4p_experiments({"BTCUSDT": _market_frame()})


def test_v4p_main_forwards_explicit_cli_paths_and_policy(
    monkeypatch: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_diagnostic(
        input_dir: Path,
        report: Path,
        comparison: Path,
        trades: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured.update(
            {
                "input": input_dir,
                "report": report,
                "comparison": comparison,
                "trades": trades,
                **kwargs,
            }
        )
        return {}

    monkeypatch.setattr(v4p, "run_v4p_diagnostic", fake_diagnostic)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-v4p",
            "--input-dir",
            str(tmp_path),
            "--output-report",
            str(tmp_path / "r.json"),
            "--output-comparison-csv",
            str(tmp_path / "c.csv"),
            "--output-trades-csv",
            str(tmp_path / "t.csv"),
            "--days",
            "730",
            "--commit-sha",
            "c" * 40,
        ],
    )
    assert v4p.main() == 0
    assert captured["days"] == 730
    assert captured["commit_sha"] == "c" * 40
