"""Run a deterministic synthetic Phase 5 labeling / OOF meta-model check."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    add_market_features,
    forward_return_target,
)
from services.meta_labeling.meta_model import OOFFold, generate_oof_primary_predictions
from services.meta_labeling.triple_barrier import (
    TRIPLE_BARRIER_SCHEMA_VERSION,
    TripleBarrierConfig,
    apply_triple_barrier,
)
from services.meta_labeling.walkforward import evaluate_meta_oof

REPORT_SCHEMA_VERSION = "alix.phase5.synthetic-meta-labeling.v1"
MECHANICS_ONLY_CUTOFF = 0.5


def make_synthetic_ohlcv(rows: int = 1400, seed: int = 20261022) -> pd.DataFrame:
    """Create an isolated, deterministic OHLCV fixture; no provider is contacted."""
    if rows < 1250:
        raise ValueError("synthetic_row_count_must_be_at_least_1250")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01T00:00:00+00:00", periods=rows, freq="h")
    opened = np.empty(rows, dtype=float)
    close = np.empty(rows, dtype=float)
    opened[0] = 100.0
    close[0] = 100.0
    for index in range(1, rows):
        if index < rows // 3:
            volatility = 0.0028
        elif index < 2 * rows // 3:
            volatility = 0.0065
        else:
            volatility = 0.0042
        opened[index] = close[index - 1]
        close[index] = opened[index] * math.exp(
            0.00002 + volatility * rng.standard_normal()
        )
    range_scale = rng.uniform(0.0002, 0.8, size=rows)
    bar_volatility = np.where(
        np.arange(rows) < rows // 3,
        0.0028,
        np.where(np.arange(rows) < 2 * rows // 3, 0.0065, 0.0042),
    )
    high = np.maximum(opened, close) * np.exp(range_scale * bar_volatility)
    low = np.minimum(opened, close) * np.exp(-range_scale * bar_volatility)
    volume = rng.lognormal(mean=4.0, sigma=0.35, size=rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_folds() -> list[OOFFold]:
    return [
        OOFFold(0, 400, 400, 600),
        OOFFold(0, 600, 600, 800),
        OOFFold(0, 800, 800, 1000),
        OOFFold(0, 1000, 1000, 1200),
    ]


def build_synthetic_report(commit_sha: str) -> dict[str, Any]:
    """Build the report and assert closed-by-default governance metadata."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("commit_sha_must_be_full_git_sha")
    candles = make_synthetic_ohlcv()
    features_frame = add_market_features(candles)
    feature_frame = features_frame.loc[:, FEATURE_COLUMNS]
    forward = forward_return_target(candles, horizon=1).to_numpy(dtype=float)
    primary_values = np.where(
        np.isfinite(forward),
        (forward > 0.0).astype(float),
        np.nan,
    )
    primary_target = pd.Series(primary_values)
    primary_label_end = np.arange(len(candles), dtype=int) + 1
    folds = _make_folds()
    primary_oof = generate_oof_primary_predictions(
        feature_frame,
        primary_target,
        primary_label_end,
        folds,
        lambda: LogisticRegression(
            C=1.0, max_iter=500, random_state=42, solver="lbfgs"
        ),
    )

    event_indices = primary_oof["event_index"].to_numpy(dtype=int)
    primary_classes = primary_oof["primary_prediction"].to_numpy(dtype=float)
    volatility = (
        feature_frame["%-volatility_12"].iloc[event_indices].to_numpy(dtype=float)
    )
    if not np.isfinite(volatility).all() or np.any(volatility <= 0):
        raise ValueError("synthetic_event_volatility_invalid")
    events = pd.DataFrame(
        {
            "event_index": event_indices,
            "side": np.where(primary_classes >= 0.5, 1, -1),
            "target_volatility": volatility,
        }
    )
    barrier_config = TripleBarrierConfig(
        profit_taking_multiplier=1.0,
        stop_loss_multiplier=1.0,
        max_holding_bars=12,
        fee_bps_per_side=10.0,
        slippage_bps_per_side=5.0,
        timeframe_seconds=3600,
    )
    labels = apply_triple_barrier(candles, events, barrier_config)
    meta_evaluation = evaluate_meta_oof(
        feature_frame,
        primary_oof,
        labels,
        minimum_probability=MECHANICS_ONLY_CUTOFF,
    )
    csv_content = candles.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.12g",
    ).encode("utf-8")
    config = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "triple_barrier": barrier_config.to_dict(),
        "oof_folds": [fold.__dict__ for fold in folds],
        "research_probability_cutoff": MECHANICS_ONLY_CUTOFF,
        "research_cutoff_owner_approved": False,
        "primary_model": "sklearn.LogisticRegression synthetic direction classifier",
        "primary_target": "one-candle future-return direction, label only",
    }
    config_hash = sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    barrier_counts = {
        str(name): int(count)
        for name, count in labels["exit_reason"].value_counts().sort_index().items()
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_status": "SYNTHETIC_MECHANICS_ONLY",
        "commit_sha": commit_sha,
        "dataset": {
            "kind": "synthetic",
            "seed": 20261022,
            "pair": "SYNTHETIC/USDT",
            "timeframe": "1h",
            "row_count": int(len(candles)),
            "valid_feature_rows": int(feature_frame.notna().all(axis=1).sum()),
            "start": candles["date"].iloc[0].isoformat(),
            "end": candles["date"].iloc[-1].isoformat(),
            "sha256": sha256(csv_content).hexdigest(),
        },
        "configuration": config,
        "configuration_sha256": config_hash,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "contracts": {
            "triple_barrier": TRIPLE_BARRIER_SCHEMA_VERSION,
            "feature_schema": "alix.freqai.features.v1",
            "primary_target_schema": "alix.freqai.forward-return.v1",
            "meta_target": "net_return_after_fee_and_slippage_strictly_positive",
            "features_contain_future_label": False,
            "primary_signals_used_for_meta_training": "OUT_OF_FOLD_ONLY",
        },
        "primary_oof": {
            "model_type": "synthetic mechanics LogisticRegression classifier",
            "fold_count": len(folds),
            "oos_sample_count": int(len(primary_oof)),
            "minimum_primary_train_count": int(
                primary_oof["primary_train_count"].min()
            ),
            "purge_rule": "primary label_end_index < test_start",
        },
        "triple_barrier": {
            "event_count": int(len(labels)),
            "barrier_outcomes": barrier_counts,
            "positive_meta_labels": int(labels["meta_label"].sum()),
            "negative_meta_labels": int((labels["meta_label"] == 0).sum()),
            "positive_net_return_threshold_is_only_label_semantics": True,
            "fees_and_slippage_are_assumptions_not_live_cost_estimates": True,
        },
        "meta_evaluation": meta_evaluation,
        "economic_gates": {"status": "NOT_EVALUATED"},
        "champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "trading_authorization": "NO_GO",
        "risk_gate_approval": False,
        "model_output_is_order_permission": False,
        "limitations": [
            "All rows are deterministic synthetic data; no real-market performance is established.",
            "The primary classifier exists only to exercise the OOF pipeline and is not the Phase 4 production baseline.",
            "The 0.5 probability cutoff is a mechanics diagnostic, not owner-approved economic policy.",
            "Historical market data, pre-registered economic criteria, independent risk review and production FreqAI lifecycle validation remain open.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic-only Phase 5 Triple-Barrier and Meta-Labeling mechanics check."
    )
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = build_synthetic_report(args.commit_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "evaluation_status": report["evaluation_status"],
                "commit_sha": report["commit_sha"],
                "primary_oos_samples": report["primary_oof"]["oos_sample_count"],
                "triple_barrier_events": report["triple_barrier"]["event_count"],
                "meta_evaluated_folds": report["meta_evaluation"][
                    "evaluated_fold_count"
                ],
                "champion": report["champion"],
                "promotion_status": report["promotion_status"],
                "report": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
