from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.oracle_teacher_lstm import ACTION_BUY, ACTION_HOLD, ACTION_SELL

from scripts.oracle_meta_labeling import (
    _deflated_sharpe_ratio,
    _minimum_track_record_length,
    _probability_backtest_overfitting,
    _run_existing_oos_diagnostics,
    _binary_metrics,
    _chronological_boundaries,
    _gap_exit,
    _intrabar_exit,
    _meta_feature_matrix,
    _partition_masks,
    _profitability_targets,
    _simulate_long_flat,
)


def _frame(rows: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.2),
            "low": np.full(rows, 99.8),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1000.0),
        }
    )


def test_chronological_partitions_are_ordered_and_disjoint() -> None:
    boundaries = _chronological_boundaries(2000)
    assert boundaries["base_train_end"] == 1000
    assert boundaries["base_validation_end"] == 1100
    assert boundaries["meta_train_end"] == 1400
    assert boundaries["validation_end"] == 1700
    assert boundaries["holdout_end"] == 2000
    with pytest.raises(ValueError, match="meta_labeling_temporal_partition_too_short"):
        _chronological_boundaries(100)


def test_partition_masks_purge_forward_labels_at_each_boundary() -> None:
    boundaries = {
        "base_train_end": 100,
        "base_validation_end": 160,
        "meta_train_end": 230,
        "validation_end": 300,
        "holdout_end": 400,
    }
    rows = np.arange(400, dtype=np.int64)
    masks = _partition_masks(rows, np.ones(400, dtype=np.bool_), boundaries, horizon=24)
    assert masks["base_train"][75] and not masks["base_train"][76]
    assert masks["base_validation"][135] and not masks["base_validation"][136]
    assert masks["meta_train"][205] and not masks["meta_train"][206]
    assert masks["validation"][275] and not masks["validation"][276]
    assert masks["holdout"][375] and not masks["holdout"][376]


def test_meta_feature_matrix_has_probabilities_atr_and_ma_distances() -> None:
    probabilities = np.asarray([[0.2, 0.6, 0.2]], dtype=np.float32)
    names = [
        "return_1",
        "atr_fraction_14",
        "sma_distance_6",
        "sma_distance_24",
        "sma_distance_72",
        "sma_distance_168",
    ]
    features = np.asarray([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]], dtype=np.float32)
    matrix, meta_names = _meta_feature_matrix(probabilities, features, names)
    assert meta_names[0:3] == [
        "lstm_probability_buy",
        "lstm_probability_sell",
        "lstm_probability_hold",
    ]
    np.testing.assert_allclose(matrix[0], [0.6, 0.2, 0.2, 0.02, 0.03, 0.04, 0.05, 0.06])


def test_binary_metrics_calculate_macro_f1() -> None:
    result = _binary_metrics(
        np.asarray([1, 1, 0, 0], dtype=np.int64),
        np.asarray([1, 0, 0, 1], dtype=np.int64),
    )
    assert result["sample_count"] == 4
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["macro_f1"] == pytest.approx(0.5)


def test_profitability_target_accounts_for_barriers_and_costs() -> None:
    frame = _frame(80)
    frame.loc[20, "close"] = 104.0
    frame.loc[30, "close"] = 90.0
    labels = pd.DataFrame(
        {
            "oracle_action": np.full(80, ACTION_HOLD, dtype=np.int64),
            "entry_price": np.full(80, 100.0),
            "label_valid": np.arange(80, dtype=np.int64) + 24 < 80,
        }
    )
    labels.loc[1, "oracle_action"] = ACTION_BUY
    labels.loc[2, "oracle_action"] = ACTION_SELL
    targets, returns = _profitability_targets(frame, labels, horizon=24)
    assert targets[1] == 1
    assert targets[2] == 0
    assert returns[1] == pytest.approx(0.03 - 0.009)
    assert returns[2] == pytest.approx(-0.015 - 0.009)
    assert np.isnan(returns[-1])


def test_barrier_tie_is_stop_first_and_gaps_are_conservative() -> None:
    position = {
        "entry_index": 10,
        "entry_price": 100.0,
        "entry_equity": 1.0,
        "quantity": 0.009,
    }
    assert _gap_exit(98.0, position) == ("GAP_STOP", 98.0)
    assert _gap_exit(104.0, position) == ("GAP_TAKE_PROFIT", 104.0)
    assert _gap_exit(100.0, position) is None
    assert _intrabar_exit(104.0, 97.0, position) == ("SAME_CANDLE_STOP_FIRST", 98.5)
    assert _intrabar_exit(104.0, 99.0, position) == ("TAKE_PROFIT", 103.0)
    assert _intrabar_exit(102.0, 99.0, position) is None


def test_meta_filter_does_not_enter_short_or_approve_low_probability_buy() -> None:
    frame = _frame(80)
    actions = np.full(80, ACTION_HOLD, dtype=np.int64)
    probabilities = np.zeros(80, dtype=np.float64)
    actions[1] = ACTION_BUY
    probabilities[1] = 0.2
    actions[2] = ACTION_SELL
    result = _simulate_long_flat(
        frame, actions, probabilities, threshold=0.5, start=0, end=40, horizon=24
    )
    assert result["closed_trade_count"] == 0
    assert result["net_return_fraction"] == pytest.approx(0.0)


def test_meta_filter_executes_approved_buy_and_respects_take_profit() -> None:
    frame = _frame(100)
    actions = np.full(100, ACTION_HOLD, dtype=np.int64)
    probabilities = np.zeros(100, dtype=np.float64)
    actions[1] = ACTION_BUY
    probabilities[1] = 0.9
    frame.loc[2, ["high", "low", "close"]] = [104.0, 99.0, 103.0]
    result = _simulate_long_flat(
        frame, actions, probabilities, threshold=0.5, start=0, end=50, horizon=24
    )
    assert result["closed_trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "TAKE_PROFIT"
    assert result["net_return_fraction"] > 0
    assert result["fee_bps_per_side"] == 25.0


def test_meta_economic_simulation_rejects_bad_input() -> None:
    frame = _frame(40)
    with pytest.raises(
        ValueError, match="meta_economic_simulation_configuration_invalid"
    ):
        _simulate_long_flat(
            frame,
            np.zeros(30, dtype=np.int64),
            np.zeros(40),
            threshold=0.5,
            start=0,
            end=20,
        )


def test_dsr_and_minimum_track_record_length_are_explicit_about_track_record() -> None:
    returns = np.asarray([0.01, 0.02, 0.015, 0.03, 0.005, 0.02, 0.01], dtype=np.float64)
    dsr = _deflated_sharpe_ratio(returns, trial_count=11)
    mintrl = _minimum_track_record_length(returns)
    assert dsr["status"] == "estimated"
    assert 0.0 <= dsr["probability"] <= 1.0
    assert dsr["candidate_threshold_trials"] == 11
    assert mintrl["status"] == "estimated"
    assert mintrl["minimum_track_record_length_trades"] >= len(returns)
    assert mintrl["additional_trades_required"] >= 0


def test_dsr_and_mintrl_fail_closed_for_short_or_nonpositive_track_record() -> None:
    short = _deflated_sharpe_ratio(np.asarray([0.01, -0.01]), trial_count=11)
    assert short["probability"] is None
    assert short["status"] == "insufficient_track_record"
    assert (
        _minimum_track_record_length(np.asarray([0.01, -0.01]))[
            "minimum_track_record_length_trades"
        ]
        is None
    )
    negative = _minimum_track_record_length(
        np.asarray([-0.03, -0.02, -0.01, -0.04, -0.02], dtype=np.float64)
    )
    assert negative["minimum_track_record_length_trades"] is None
    assert (
        negative["status"] == "no_finite_track_length_for_nonpositive_observed_sharpe"
    )
    with pytest.raises(ValueError, match="meta_dsr_configuration_invalid"):
        _deflated_sharpe_ratio(np.asarray([0.1, 0.2, 0.3]), trial_count=0)


def test_pbo_uses_purged_cscv_on_threshold_return_streams() -> None:
    x = np.arange(360, dtype=np.float64)
    result = _probability_backtest_overfitting(
        {
            0.30: 0.002 * np.sin(x / 7.0) + 0.0005,
            0.50: 0.0015 * np.cos(x / 9.0) - 0.0001,
            0.70: 0.001 * np.sin(x / 5.0 + 0.5),
        },
        block_count=6,
        purge_candles=12,
    )
    assert result["status"] == "estimated"
    assert result["cscv_split_count"] == 20
    assert result["block_count"] == 6
    assert 0.0 <= result["probability"] <= 1.0
    with pytest.raises(ValueError, match="meta_pbo_track_record_too_short_for_purge"):
        _probability_backtest_overfitting(
            {0.3: np.asarray([0.0] * 40), 0.5: np.asarray([0.0] * 40)},
            block_count=6,
            purge_candles=4,
        )


def test_existing_oos_diagnostics_updates_only_the_matching_artifact(
    tmp_path: Path,
) -> None:
    source_sha = "a" * 40
    code_sha = "b" * 40
    artifact_dir = tmp_path
    x = np.arange(900, dtype=np.float64)
    close = 100.0 + 2.0 * np.sin(x / 9.0) + 0.4 * np.sin(x / 3.0)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=900, freq="h", tz="UTC"),
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.5,
            "low": np.minimum(open_price, close) - 0.5,
            "close": close,
            "volume": np.full(900, 1000.0),
        }
    )
    market_path = artifact_dir / "public-market-data.csv"
    frame.to_csv(market_path, index=False)
    source_hash = hashlib.sha256(market_path.read_bytes()).hexdigest()
    rows = []
    for row_index in range(50, 850):
        step = row_index % 12
        action = "BUY" if step in (0, 1) else "SELL" if step == 5 else "HOLD"
        rows.append(
            {
                "row_index": row_index,
                "timestamp": frame.loc[row_index, "timestamp"],
                "base_lstm_action": action,
                "meta_profit_probability": (row_index % 10) / 10.0,
                "partition": "validation" if 200 <= row_index < 800 else "meta_train",
            }
        )
    pd.DataFrame(rows).to_csv(
        artifact_dir / "meta-out-of-sample-predictions.csv", index=False
    )
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "private_api_calls": 0,
    }
    (artifact_dir / "oracle-meta-labeling-manifest.json").write_text(
        json.dumps(
            {
                "commit_sha": source_sha,
                "row_count": 900,
                "source_csv_sha256": source_hash,
                "safety": safety,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "commit_sha": source_sha,
        "dataset": {"row_count": 900, "csv_sha256": source_hash},
        "threshold_selection": {
            "candidate_thresholds": [
                round(value, 2) for value in np.arange(0.3, 0.801, 0.05)
            ]
        },
        "validation": {
            "meta_filtered_economic_metrics_after_costs": {
                "trades": [
                    {"net_pnl_fraction": value}
                    for value in [-0.02, -0.01, 0.01, -0.015, 0.02, -0.01, 0.005]
                ]
            }
        },
        "economic_simulation": {"maximum_holding_candles": 24},
        "safety": safety,
        "decision": "RESEARCH_ONLY_NOT_PROMOTED",
    }
    report_path = artifact_dir / "oracle-meta-labeling-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    diagnostics = _run_existing_oos_diagnostics(artifact_dir, source_sha, code_sha)
    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert diagnostics["source_artifact_commit_sha"] == source_sha
    assert diagnostics["diagnostics_code_commit_sha"] == code_sha
    assert updated["overfit_diagnostics"]["status"] in (
        "OVERFIT_WARNING",
        "NO_OVERFIT_WARNING",
    )
    assert updated["safety"] == safety
    assert updated["decision"] == "RESEARCH_ONLY_NOT_PROMOTED"
    assert (artifact_dir / "oracle-meta-labeling-diagnostics.json").is_file()
    with pytest.raises(ValueError, match="meta_overfit_source_commit_sha_mismatch"):
        _run_existing_oos_diagnostics(artifact_dir, "c" * 40, code_sha)
