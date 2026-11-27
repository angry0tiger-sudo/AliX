from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.run_m5_2_sentiment_model import (
    CONTROL_FEATURES,
    TREATMENT_FEATURES,
    Candle,
    PortfolioOutcome,
    Trade,
    _paired_comparisons,
    align_sentiment,
    build_fold_plans,
    build_model,
    build_price_features,
    check_absolute_gates,
    is_stable_fold,
    is_valid_fold,
    load_preregistration,
    max_drawdown,
    net_profit_factor,
    simulate_portfolio,
)


def _candles(count: int = 32) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: list[Candle] = []
    for index in range(count):
        opened = 100.0 + index
        rows.append(
            Candle(
                start=start + timedelta(hours=index),
                open=opened,
                high=opened + 2.0,
                low=opened - 1.0,
                close=opened + 1.0,
                volume=10.0 + index,
            )
        )
    return rows


def _sentiment_record(
    observation_time: str, available_at: str, value: int
) -> dict[str, object]:
    observation = datetime.fromisoformat(observation_time.replace("Z", "+00:00"))
    return {
        "provider_id": "alternative_me_fear_greed",
        "target_market": "Nobitex Spot/USDT BTCUSDT",
        "market_scope": "global_btc_sentiment_not_nobitex_pair_specific",
        "timestamp": int(observation.timestamp()),
        "observation_time_utc": observation_time,
        "value": value,
        "value_classification": "Fear" if value < 50 else "Greed",
        "fetched_at": "2024-01-05T00:00:00Z",
        "available_at": available_at,
        "available_at_basis": "observation_timestamp_plus_24h_conservative_proxy",
    }


def test_preregistration_is_read_only_and_sha_locked() -> None:
    config, digest = load_preregistration()

    assert digest == "851d6c4b39c41cacec8ccd712cf17bb40dd42dd45f2bb8813920a7f2ebe481ba"
    assert config["preregistration"]["status"] == "PREREGISTERED_LOCKED"
    assert config["scope"]["market"] == "BTCUSDT"
    assert tuple(config["feature_specification"]["feature_names_ohlcv_only_control"]) == CONTROL_FEATURES
    assert tuple(config["feature_specification"]["feature_names_sentiment_augmented"]) == TREATMENT_FEATURES
    assert config["safety"]["holdout_access"] == "PROHIBITED"


def test_sentiment_alignment_is_asof_and_never_backfills() -> None:
    records = [
        _sentiment_record("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", 35),
        _sentiment_record("2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", 80),
    ]
    decisions = [
        "2024-01-01T23:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-02T23:00:00Z",
        "2024-01-03T00:00:00Z",
    ]

    values, available, observations, ages = align_sentiment(records, decisions)

    assert np.isnan(values[0])
    assert available[0] is None
    assert observations[0] is None
    assert values[1] == 35.0
    assert available[1] == "2024-01-02T00:00:00Z"
    assert values[2] == 35.0
    assert values[3] == 80.0
    assert available[3] == "2024-01-03T00:00:00Z"
    assert np.all(ages[1:] >= 0.0)


def test_price_features_match_registered_eight_column_contract() -> None:
    frame = _candles()
    matrix = build_price_features(frame)

    assert matrix.shape == (len(frame), 8)
    assert tuple(CONTROL_FEATURES) == (
        "log_return_1",
        "log_return_4",
        "log_return_24",
        "realized_volatility_6",
        "realized_volatility_24",
        "range_fraction_1",
        "atr_fraction_14",
        "log_volume_change_1",
    )
    assert np.isnan(matrix[0, 0])
    assert matrix[1, 0] == pytest.approx(np.log(frame[1].close / frame[0].close))
    assert matrix[5, 3] == pytest.approx(
        np.std(
            [
                np.log(frame[index].close / frame[index - 1].close)
                for index in range(1, 6)
            ],
            ddof=0,
        )
    )
    assert np.isnan(matrix[22, 4])
    assert np.isfinite(matrix[23, 4])
    assert np.isfinite(matrix[13, 6])
    assert matrix[2, 7] == pytest.approx(
        np.log1p(frame[2].volume) - np.log1p(frame[1].volume)
    )


def test_fold_plan_uses_four_expanding_chronological_blocks_and_purge() -> None:
    eligible = list(range(100))
    plans = build_fold_plans(eligible, [True] * 100, horizon=4)

    assert len(plans) == 4
    assert [plan.fold_id for plan in plans] == ["WF-01", "WF-02", "WF-03", "WF-04"]
    assert plans[0].first_test_index == 20
    assert plans[0].last_test_index == 39
    assert plans[0].train_indices == tuple(range(16))
    assert plans[0].purged_train_rows == 4
    assert plans[0].test_indices == tuple(range(20, 36))
    assert plans[0].mark_indices == tuple(range(20, 40))
    for plan in plans:
        assert all(index + 4 < plan.first_test_index for index in plan.train_indices)
        assert all(index + 4 <= plan.last_test_index for index in plan.test_indices)


def test_fold_plan_rejects_non_monotonic_rows_and_unregistered_horizon() -> None:
    with pytest.raises(ValueError, match="strictly_ordered"):
        build_fold_plans([0, 2, 1, 3, 4], [True] * 5, horizon=1)
    with pytest.raises(ValueError, match="not_registered"):
        build_fold_plans(list(range(10)), [True] * 10, horizon=24)


def test_model_uses_only_frozen_logistic_regression_parameters() -> None:
    config, _ = load_preregistration()
    model = build_model(config)
    classifier = model.named_steps["classifier"]

    assert tuple(name for name, _ in model.steps) == ("scale", "classifier")
    assert classifier.C == 1.0
    assert classifier.penalty == "l2"
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 1000
    assert classifier.tol == 0.0001
    assert classifier.random_state == 42
    assert classifier.class_weight is None


def test_portfolio_simulation_applies_round_trip_cost_once_and_closes_cohorts() -> None:
    candles = _candles(8)
    result = simulate_portfolio(
        candles=candles,
        decisions=[1, 3],
        signals=[True, True],
        horizon=1,
        cost=0.005,
        marks=list(range(1, 6)),
    )

    assert result.metrics["completed_oos_trades"] == 2
    assert result.metrics["maximum_concurrent_cohorts"] == 1
    assert len(result.trades) == 2
    assert result.trades[0].entry_index == 2
    assert result.trades[0].exit_index == 2
    assert result.trades[0].pnl == pytest.approx(
        candles[2].close / candles[2].open - 1.0 - 0.005
    )
    assert result.pnl == pytest.approx(sum(trade.pnl for trade in result.trades))
    assert result.nav[-1] == pytest.approx(1.0 + result.pnl)


def test_portfolio_rejects_open_cohorts_at_fold_end() -> None:
    candles = _candles(8)
    with pytest.raises(ValueError, match="crosses_test_fold"):
        simulate_portfolio(
            candles=candles,
            decisions=[3],
            signals=[True],
            horizon=4,
            cost=0.005,
            marks=[3, 4, 5],
        )


def test_profit_factor_and_drawdown_are_fail_closed_for_undefined_pf() -> None:
    trades = (
        Trade(0, 1, 1, 0.5, 0.01),
        Trade(1, 2, 2, 0.5, -0.005),
    )

    assert net_profit_factor(trades) == pytest.approx(2.0)
    assert net_profit_factor((trades[0],)) is None
    assert max_drawdown((1.0, 1.1, 1.0)) == pytest.approx(1.0 / 11.0)


def test_fold_stability_requires_trade_count_positive_pnl_and_pf() -> None:
    trades = tuple(
        [Trade(index, index + 1, index + 1, 0.05, 0.001) for index in range(19)]
        + [Trade(19, 20, 20, 0.05, -0.0095)]
    )
    stable = PortfolioOutcome(
        metrics={"completed_oos_trades": 20, "net_profit_factor": 2.0},
        nav=(1.0,),
        pnl=0.0095,
        trades=trades,
    )
    insufficient = PortfolioOutcome(
        metrics={"completed_oos_trades": 19, "net_profit_factor": 2.0},
        nav=(1.0,),
        pnl=0.01,
        trades=trades[:-1],
    )

    assert is_valid_fold(stable)
    assert is_stable_fold(stable)
    assert not is_valid_fold(insufficient)
    assert not is_stable_fold(insufficient)


def test_absolute_gates_apply_each_registered_threshold() -> None:
    config, _ = load_preregistration()
    pooled = {
        "completed_oos_trades": 120,
        "net_profit_factor": 1.15,
        "max_drawdown": 0.10,
    }

    result = check_absolute_gates(pooled, valid_folds=4, stable_folds=3, config=config)

    assert all(result.values())
    failed = check_absolute_gates(
        {"completed_oos_trades": 99, "net_profit_factor": 1.09, "max_drawdown": 0.16},
        valid_folds=2,
        stable_folds=2,
        config=config,
    )
    assert not any(failed.values())


def test_paired_comparison_requires_both_netpf_and_net_pnl_improvement() -> None:
    trials: dict[str, Any] = {}
    for horizon in (1, 4):
        for arm, net_pf, pnl in (
            ("OHLCV-LR-BASELINE", 1.00, 0.01),
            ("SENTIMENT-LR-AUGMENTED", 1.20, 0.02),
        ):
            trial_id = f"NOBITEX-M5-2-{horizon}h-{arm}-01"
            trials[trial_id] = {
                "cost_scenarios": {
                    cost_name: {
                        "pooled_oos": {
                            "net_profit_factor": net_pf,
                            "total_net_pnl_fraction_of_initial_equity": pnl,
                        }
                    }
                    for cost_name in ("Cost-50", "Cost-90")
                }
            }

    paired = _paired_comparisons(trials)

    assert paired["1"]["Cost-50"]["incremental_value_passed"]
    assert paired["4"]["Cost-90"]["incremental_value_passed"]
    trials["NOBITEX-M5-2-4h-SENTIMENT-LR-AUGMENTED-01"]["cost_scenarios"]["Cost-90"]["pooled_oos"][
        "total_net_pnl_fraction_of_initial_equity"
    ] = 0.005
    paired = _paired_comparisons(trials)
    assert not paired["4"]["Cost-90"]["incremental_value_passed"]
