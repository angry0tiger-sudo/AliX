from __future__ import annotations

from typing import Any

import pytest

from scripts.run_ml_core_v4k_trade_management import (
    _validate_trade_ledger,
    trade_management_config,
    validate_v4j_baseline,
)
from services.ml_core.economic_backtest import TradeManagementConfig


def _source_report() -> dict[str, Any]:
    return {
        "input_datasets": [
            {"symbol": "BTCUSDT", "csv_sha256": "btc-sha"},
            {"symbol": "ETHUSDT", "csv_sha256": "eth-sha"},
        ]
    }


def _baseline_report() -> dict[str, Any]:
    return {
        "schema_version": "alix.ml-core.v4j-cost-isolation.v1",
        "commit_sha": "fe5cf50b0084ea412345e82863569d084548faee",
        "source_v4h_commit_sha": "e34861c4873b7cc2170aaaf6c79ed9b464df0a9a",
        "source_v4h_run_id": 41293168191,
        "source_v4h_report_sha256": (
            "3f3c08c583fabb6c91fcdee81c036db6c92a988cdf2537306813af97e89de6a0"
        ),
        "candidate_count": 13,
        "source_v4h_candidate_count": 13,
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "spot_execution_assumptions": {
            "sell_semantics": "CLOSE_LONG_OR_IGNORE_WHEN_FLAT",
            "short_positions_supported": False,
        },
        "cost_profiles": {
            "realistic": {
                "fee_bps_per_side": 25.0,
                "slippage_bps_per_side": 20.0,
            }
        },
        "input_datasets": [
            {
                "symbol": "BTCUSDT",
                "csv_sha256": "btc-sha",
                "byte_identical_to_v4h": True,
            },
            {
                "symbol": "ETHUSDT",
                "csv_sha256": "eth-sha",
                "byte_identical_to_v4h": True,
            },
        ],
        "candidates": [{"candidate_id": f"candidate-{index}"} for index in range(13)],
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }


def test_trade_management_defaults_match_v4h_tighter_sl_geometry() -> None:
    config = trade_management_config()
    assert config.atr_period == 14
    assert config.take_profit_atr_multiplier == 2.0
    assert config.stop_loss_atr_multiplier == 1.5
    assert config.horizon_candles == 12
    assert config.confidence_threshold == 0.55


def test_trade_management_config_exposes_confidence_threshold() -> None:
    assert (
        trade_management_config(confidence_threshold=0.50).confidence_threshold == 0.50
    )


def test_trade_management_config_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        trade_management_config(confidence_threshold=0.0)
    with pytest.raises(ValueError, match="atr_period"):
        trade_management_config(atr_period=0)
    with pytest.raises(ValueError, match="multipliers"):
        trade_management_config(take_profit_atr_multiplier=-1.0)
    with pytest.raises(ValueError, match="horizon_candles"):
        trade_management_config(horizon_candles=0)


def test_trade_ledger_respects_optional_confidence_floor() -> None:
    trade = {
        "direction": "LONG",
        "entry_confidence": 0.40,
        "bars_held": 1,
        "stop_loss_price": 95.0,
        "barrier_reference_price": 100.0,
        "take_profit_price": 105.0,
    }

    assert _validate_trade_ledger(
        [trade],
        TradeManagementConfig(confidence_threshold=None),
    ) == {"LONG": 1, "SHORT": 0}

    with pytest.raises(RuntimeError, match="confidence_invariant"):
        _validate_trade_ledger(
            [trade],
            TradeManagementConfig(confidence_threshold=0.55),
        )


def test_v4j_baseline_requires_exact_spot_only_reference_and_safety() -> None:
    baseline = _baseline_report()
    ids = [f"candidate-{index}" for index in range(13)]
    validate_v4j_baseline(baseline, _source_report(), ids)

    baseline["spot_execution_assumptions"]["short_positions_supported"] = True
    with pytest.raises(ValueError, match="v4j_baseline_report_contract"):
        validate_v4j_baseline(baseline, _source_report(), ids)


def test_v4j_baseline_rejects_candidate_or_market_data_provenance_mismatch() -> None:
    baseline = _baseline_report()
    ids = [f"candidate-{index}" for index in range(13)]
    baseline["candidates"][0]["candidate_id"] = "unexpected-candidate"
    with pytest.raises(ValueError, match="v4j_baseline_report_contract"):
        validate_v4j_baseline(baseline, _source_report(), ids)

    baseline = _baseline_report()
    baseline["input_datasets"][1]["csv_sha256"] = "different-eth-sha"
    with pytest.raises(ValueError, match="v4j_baseline_report_contract"):
        validate_v4j_baseline(baseline, _source_report(), ids)
