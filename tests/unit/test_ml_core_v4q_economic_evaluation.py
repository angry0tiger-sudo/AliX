from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.run_ml_core_v4q_feature_target_rebuild as v4q


def test_v4q_trade_csv_contains_gross_fees_slippage_and_net_pnl() -> None:
    records = [
        {
            "candidate_id": "V4Q|BTC/USDT|15m|Baseline|weight_none",
            "pair": "BTC/USDT",
            "timeframe": "15m",
            "feature_profile": "Baseline",
            "weighting_policy": "weight_none",
            "trades": [
                {
                    "gross_market_pnl_usd": 12.0,
                    "fees_usd": 2.5,
                    "slippage_cost_usd": 2.0,
                    "net_pnl_usd": 7.5,
                    "exit_reason": "TP_HIT",
                }
            ],
        }
    ]

    row = v4q._trade_csv_rows(records)[0]

    assert row["gross_market_pnl_usd"] == 12.0
    assert row["fees_usd"] == 2.5
    assert row["slippage_cost_usd"] == 2.0
    assert row["net_pnl_usd"] == 7.5


def test_v4q_economic_cost_summary_attributes_gross_fees_slippage_and_net() -> None:
    records = [
        {
            "total_fees_usd": 2.5,
            "total_slippage_cost_usd": 2.0,
            "trades": [
                {
                    "gross_market_pnl_usd": 12.0,
                    "net_pnl_usd": 7.5,
                }
            ],
        }
    ]

    assert v4q._economic_cost_summary(records) == {
        "total_gross_market_pnl_usd": 12.0,
        "total_fees_usd": 2.5,
        "total_slippage_cost_usd": 2.0,
        "total_net_pnl_usd": 7.5,
    }


def test_v4q_feature_importance_rows_are_named_and_fail_closed() -> None:
    experiment = {
        "pair": "BTC/USDT",
        "timeframe": "15m",
        "feature_profile": "Baseline",
        "model_variants": {
            "weight_none": {
                "feature_importance_gain": [
                    {"feature": "rsi_14", "gain": 3.0},
                    {"feature": "macd_histogram", "gain": 1.5},
                ]
            }
        },
    }

    rows = v4q._feature_importance_rows([experiment])

    assert rows == [
        {
            "candidate_id": (
                "V4Q_1ATR_SYMMETRIC|BTC/USDT|15m|Baseline|weight_none"
            ),
            "feature": "rsi_14",
            "gain": 3.0,
        },
        {
            "candidate_id": (
                "V4Q_1ATR_SYMMETRIC|BTC/USDT|15m|Baseline|weight_none"
            ),
            "feature": "macd_histogram",
            "gain": 1.5,
        },
    ]

    with pytest.raises(ValueError, match="v4q_feature_importance_missing"):
        v4q._feature_importance_rows(
            [{**experiment, "model_variants": {"weight_none": {}}}]
        )


def test_v4q_comparison_csv_summary_renders_all_32_rows(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "profit_factor"])
        writer.writeheader()
        for index in range(v4q.V4Q_EXPECTED_MODELS):
            writer.writerow(
                {
                    "candidate_id": f"candidate-{index:02d}",
                    "profit_factor": "0.75",
                }
            )

    summary = v4q._render_comparison_csv_markdown(comparison)

    assert "## Comparison CSV (exact persisted rows)" in summary
    assert "| candidate_id | profit_factor |" in summary
    assert "candidate-00" in summary
    assert "candidate-31" in summary
    assert summary.count("| candidate-") == v4q.V4Q_EXPECTED_MODELS


def test_v4q_comparison_csv_summary_rejects_partial_data(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "profit_factor"])
        writer.writeheader()
        writer.writerow({"candidate_id": "candidate-00", "profit_factor": "0.75"})

    with pytest.raises(
        ValueError, match="v4q_comparison_csv_candidate_count_mismatch"
    ):
        v4q._render_comparison_csv_markdown(comparison)
