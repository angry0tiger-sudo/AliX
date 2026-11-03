from __future__ import annotations

from typing import Any

import pytest

from scripts import audit_v4s_threshold_calibration as audit


def _report() -> dict[str, Any]:
    arms = []
    candidates = []
    experiments = []
    for name in audit.ARM_NAMES:
        arms.append(
            {
                "name": name,
                "aggregate": {
                    "trained_candidate_count": 1,
                    "sample_gate_skipped_candidate_count": 0,
                    "trade_count": 3,
                    "combined_net_profit_factor": 0.8,
                    "total_net_pnl_usd": -10.0,
                    "strict_economic_pass_candidate_count": 0,
                },
            }
        )
        candidates.append(
            {
                "arm": name,
                "confidence_filter": {
                    "threshold": 0.6,
                    "raw_buy_signal_count": 10,
                    "accepted_buy_signal_count": 4,
                    "rejected_buy_signal_count": 6,
                },
            }
        )
        experiments.append(
            {
                "arm": name,
                "pair": "BTC/USDT",
                "timeframe": "1h",
                "feature_profile": "Baseline",
                "label_distribution": {
                    "train": {
                        "total": 10,
                        "classes": {
                            "BUY": {"count": 3},
                            "HOLD": {"count": 2},
                            "SELL": {"count": 5},
                        },
                    },
                    "validation": {
                        "total": 5,
                        "classes": {
                            "BUY": {"count": 1},
                            "HOLD": {"count": 1},
                            "SELL": {"count": 3},
                        },
                    },
                },
                "label_audit": {
                    "exit_reason_distribution": {
                        "total": 10,
                        "reasons": {
                            "TP_HIT": {"count": 3},
                            "SL_HIT": {"count": 5},
                            "HORIZON_EXPIRY": {"count": 1},
                            "COLLISION": {"count": 1},
                        },
                    },
                },
                "model_variants": {
                    "weight_none": {
                        "threshold_optimization": {
                            "status": "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES",
                            "calibration": {
                                "label_distribution": {
                                    "classes": {"HOLD": {"count": 15}}
                                }
                            },
                        },
                        "probability_diagnostics": {
                            "forward_validation": {
                                "hold_auc": 0.7,
                                "hold_pr_auc": 0.1,
                            }
                        },
                    }
                },
            }
        )
    return {
        "schema_version": audit.REPORT_SCHEMA,
        "commit_sha": audit.EXPECTED_COMMIT_SHA,
        "experimental_design": "2x2_PAIRED_FACTORIAL",
        "model_training_performed": True,
        "economic_evaluation_performed": True,
        "arms": arms,
        "candidates": candidates,
        "experiments": experiments,
        "trade_rows": [{"net_pnl_usd": -1.0}],
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "credentials_used": False,
            "public_api_only": True,
        },
    }


def test_run25_audit_fails_closed_without_row_level_probabilities() -> None:
    summary = audit.analyze_report(_report())

    assert summary["training_performed_by_v4s"] is False
    assert summary["model_training_performed_in_source_run"] is True
    assert summary["row_level_probabilities_available"] is False
    assert (
        summary["counterfactual_sweep_status"]
        == "NOT_COMPUTABLE_MISSING_ROW_LEVEL_PROBABILITIES"
    )
    assert summary["activity"][audit.ARM_NAMES[0]]["trade_count"] == 3
    assert summary["activity"][audit.ARM_NAMES[0]]["confidence_threshold_counts"] == {
        "0.60": 1
    }


def test_row_level_probability_vectors_enable_sweep_detection() -> None:
    report = _report()
    report["experiments"][0]["predicted_probabilities"] = [
        {"BUY": 0.7, "HOLD": 0.1, "SELL": 0.2}
    ]

    summary = audit.analyze_report(report)

    assert summary["row_level_probabilities_available"] is True
    assert summary["counterfactual_sweep_status"] == "COMPUTABLE"


def test_label_pool_deduplicates_feature_profiles() -> None:
    experiments = _report()["experiments"]
    duplicate = dict(experiments[0])
    duplicate["feature_profile"] = "Baseline+Trend"
    experiments.append(duplicate)

    summary = audit._pooled_label_summary(experiments, audit.ARM_NAMES[0])

    assert summary["unique_pair_timeframe_contexts"] == 1
    assert summary["splits"]["validation"]["total"] == 5
    assert summary["splits"]["validation"]["percent"]["HOLD"] == 20.0


def test_label_pool_rejects_conflicting_duplicate_contexts() -> None:
    experiments = _report()["experiments"]
    duplicate = dict(experiments[0])
    duplicate["feature_profile"] = "Baseline+Trend"
    duplicate["label_distribution"] = {
        **duplicate["label_distribution"],
        "validation": {
            "total": 5,
            "classes": {
                "BUY": {"count": 2},
                "HOLD": {"count": 1},
                "SELL": {"count": 2},
            },
        },
    }
    experiments.append(duplicate)

    with pytest.raises(ValueError, match="v4s_duplicate_context_label_mismatch"):
        audit._pooled_label_summary(experiments, audit.ARM_NAMES[0])


def test_safety_invariant_mismatch_is_rejected() -> None:
    report = _report()
    report["safety"]["order_permission"] = True

    with pytest.raises(ValueError, match="v4s_safety_invariant_mismatch"):
        audit.analyze_report(report)


def test_row_level_probability_records_are_recognized_in_candidate_artifact() -> None:
    report = _report()
    report["candidates"][0]["probability_records"] = [
        {
            "row_index": 100,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "actual_label": "BUY",
            "probabilities": {"BUY": 0.7, "SELL": 0.2, "HOLD": 0.1},
        }
    ]
    summary = audit.analyze_report(report)
    assert summary["row_level_probabilities_available"] is True
    assert summary["counterfactual_sweep_status"] == "COMPUTABLE"
