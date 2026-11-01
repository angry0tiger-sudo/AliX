from __future__ import annotations

from typing import Any

import pytest

from scripts.run_ml_core_v4j_cost_isolation import (
    cost_profile_configs,
    validate_v4i_baseline,
)


def _source_report() -> dict[str, Any]:
    return {
        "input_datasets": [
            {"symbol": "BTCUSDT", "csv_sha256": "btc-sha"},
            {"symbol": "ETHUSDT", "csv_sha256": "eth-sha"},
        ]
    }


def _baseline_report() -> dict[str, Any]:
    return {
        "schema_version": "alix.ml-core.v4i-economic-evaluation.v1",
        "commit_sha": "6d30de1459ef113e4ef12bc3b07bb44d1001f2b5",
        "source_v4h_commit_sha": "e34861c4873b7cc2170aaaf6c79ed9b464df0a9a",
        "source_v4h_run_id": 41293168191,
        "candidate_count": 13,
        "source_v4h_candidate_count": 13,
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "execution_assumptions": {
            "sell_semantics": "OPEN_SHORT",
            "fee_bps_per_side": 25.0,
            "slippage_bps_per_side": 20.0,
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
        "candidates": [
            {"candidate_id": f"candidate-{index}"}
            for index in range(13)
        ],
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }


def test_cost_profiles_match_the_approved_three_mode_matrix() -> None:
    configs = cost_profile_configs()
    assert set(configs) == {"zero_cost", "fee_only", "realistic"}
    assert (configs["zero_cost"].fee_bps, configs["zero_cost"].slippage_bps) == (
        0.0,
        0.0,
    )
    assert (configs["fee_only"].fee_bps, configs["fee_only"].slippage_bps) == (
        25.0,
        0.0,
    )
    assert (configs["realistic"].fee_bps, configs["realistic"].slippage_bps) == (
        25.0,
        20.0,
    )
    assert all(item.position_fraction == 1.0 for item in configs.values())


def test_v4i_baseline_requires_the_exact_short_enabled_reference() -> None:
    baseline = _baseline_report()
    ids = [f"candidate-{index}" for index in range(13)]
    validate_v4i_baseline(baseline, ids, _source_report())

    baseline["execution_assumptions"]["sell_semantics"] = (
        "CLOSE_LONG_OR_IGNORE_WHEN_FLAT"
    )
    with pytest.raises(ValueError, match="v4i_baseline_report_contract"):
        validate_v4i_baseline(baseline, ids, _source_report())


def test_v4i_baseline_rejects_candidate_order_or_data_provenance_mismatch() -> None:
    baseline = _baseline_report()
    ids = [f"candidate-{index}" for index in range(13)]
    baseline["candidates"][0]["candidate_id"] = "unexpected-candidate"
    with pytest.raises(ValueError, match="v4i_baseline_report_contract"):
        validate_v4i_baseline(baseline, ids, _source_report())

    baseline = _baseline_report()
    baseline["input_datasets"][1]["csv_sha256"] = "different-eth-sha"
    with pytest.raises(ValueError, match="v4i_baseline_report_contract"):
        validate_v4i_baseline(baseline, ids, _source_report())
