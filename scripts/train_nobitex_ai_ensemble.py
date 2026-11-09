"""Train and evaluate the Nobitex AI ensemble from archived, aligned data."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

from research.ai_ensemble.ensemble import (
    StackingConfig,
    StackingEnsemble,
    evaluate_walk_forward,
)
from research.ai_ensemble.features import (
    REQUIRED_MICROSTRUCTURE_COLUMNS,
    REQUIRED_OHLCV_COLUMNS,
    build_causal_features,
    merge_market_archives,
)
from research.ai_ensemble.labels import (
    LABEL_HORIZON_CANDLES,
    ROUND_TRIP_COST_BPS,
    build_net_profit_targets,
)
from research.ai_ensemble.risk import KellySizingConfig


def _read_timestamped_csv(path: Path, required: tuple[str, ...]) -> DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"research_input_file_missing:{path}")
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"research_csv_missing_columns:{','.join(missing)}")
    parsed = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"research_csv_invalid_date:{path.name}")
    frame["date"] = parsed
    return frame


def _build_aligned_training_set(
    ohlcv_path: Path,
    microstructure_path: Path,
) -> tuple[DataFrame, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    ohlcv = _read_timestamped_csv(ohlcv_path, REQUIRED_OHLCV_COLUMNS)
    microstructure = _read_timestamped_csv(
        microstructure_path, REQUIRED_MICROSTRUCTURE_COLUMNS
    )
    market = merge_market_archives(ohlcv, microstructure)
    features = build_causal_features(market)
    targets = build_net_profit_targets(market.loc[:, list(REQUIRED_OHLCV_COLUMNS)])
    finite_feature_mask = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    market_positions = np.flatnonzero(finite_feature_mask)
    if len(market_positions) == 0:
        raise ValueError("research_feature_warmup_left_no_eligible_rows")
    feature_position = {
        int(market_index): model_index
        for model_index, market_index in enumerate(market_positions.tolist())
    }
    model_features = features.iloc[market_positions].reset_index(drop=True)
    labels = np.zeros(len(model_features), dtype=int)
    label_ends = np.full(len(model_features), -1, dtype=int)
    net_returns = np.zeros(len(model_features), dtype=float)
    eligible: list[int] = []
    for model_index, market_index in enumerate(market_positions.tolist()):
        record = targets.iloc[int(market_index)]
        target = record["target"]
        label_end = record["label_end_idx"]
        net_return = record["net_return"]
        if pd.isna(target) or pd.isna(label_end) or pd.isna(net_return):
            continue
        original_end = int(label_end)
        if original_end not in feature_position:
            raise ValueError("research_label_end_missing_from_feature_timeline")
        compressed_end = feature_position[original_end]
        if compressed_end < model_index:
            raise ValueError("research_label_end_precedes_signal")
        labels[model_index] = int(target)
        label_ends[model_index] = compressed_end
        net_returns[model_index] = float(net_return)
        eligible.append(model_index)
    if len(eligible) < 500:
        raise ValueError("research_fewer_than_500_fully_labeled_rows")
    if len(np.unique(labels[eligible])) != 2:
        raise ValueError("research_labels_do_not_contain_both_classes")
    return model_features, labels, label_ends, net_returns, eligible


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_training(
    *,
    ohlcv_path: Path,
    microstructure_path: Path,
    model_path: Path,
    report_path: Path,
    tested_strategies: int = 1,
) -> dict[str, Any]:
    features, labels, label_ends, net_returns, eligible = _build_aligned_training_set(
        ohlcv_path, microstructure_path
    )
    config = StackingConfig()
    validation = evaluate_walk_forward(
        features,
        labels,
        label_ends,
        net_returns,
        eligible,
        config=config,
        tested_strategies=tested_strategies,
    )
    model = StackingEnsemble(config).fit(
        features,
        labels,
        label_ends,
        eligible,
    )
    recent_probability = float(model.predict_proba(features.iloc[-1:])[-1])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "alix.nobitex.ai-ensemble.v1",
            "model": model,
            "feature_columns": list(features.columns),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "label_horizon_candles": LABEL_HORIZON_CANDLES,
            "training_rows": len(eligible),
        },
        model_path,
    )
    aggregate = validation["aggregate"]
    manifest: dict[str, Any] = {
        "schema_version": "alix.nobitex.ai-ensemble-training-report.v1",
        "dataset": {
            "ohlcv_path": str(ohlcv_path),
            "ohlcv_sha256": _file_sha256(ohlcv_path),
            "microstructure_path": str(microstructure_path),
            "microstructure_sha256": _file_sha256(microstructure_path),
            "interval": "1h",
            "feature_rows": len(features),
            "eligible_labeled_rows": len(eligible),
        },
        "model": {
            "base_models": ["LightGBM", "XGBoost"],
            "regime_model": "3-state diagonal Gaussian HMM (forward-filtered)",
            "meta_learner": "LogisticRegression",
            "lstm_gru": "NOT_INCLUDED_OWNER_APPROVAL_REQUIRED",
            "feature_columns": list(features.columns),
            "recent_training_probability": recent_probability,
        },
        "labeling": {
            "method": "ATR_TRIPLE_BARRIER",
            "horizon_candles": LABEL_HORIZON_CANDLES,
            "take_profit_atr": 2.0,
            "stop_loss_atr": 1.0,
            "entry_reference": "NEXT_BAR_OPEN",
            "maker_fee_bps_per_side": 25,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "risk": {
            "method": "FRACTIONAL_KELLY",
            "kelly_fraction": KellySizingConfig().kelly_fraction,
            "maximum_risk_fraction": KellySizingConfig().max_risk_fraction,
            "maximum_position_fraction": KellySizingConfig().max_position_fraction,
        },
        "validation": validation,
        "research_result": {
            "trade_count": aggregate["trade_count"],
            "mean_net_return": aggregate["mean_net_return"],
            "profit_factor": aggregate["profit_factor"],
            "deflated_sharpe_ratio": aggregate["deflated_sharpe_ratio"],
        },
        "artifact": {
            "path": str(model_path),
            "sha256": _file_sha256(model_path),
        },
        "safety": {
            "ALIX_DRY_RUN": True,
            "ALIX_ALLOW_LIVE_TRADING": False,
            "Champion": None,
            "order_permission": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "economic_acceptance": "NO-GO",
            "trading_authorization": "NO-GO",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", type=Path, required=True)
    parser.add_argument("--microstructure", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("user_data/models/nobitex-ai-ensemble.joblib"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("user_data/backtest_results/nobitex-ai-ensemble-report.json"),
    )
    parser.add_argument("--tested-strategies", type=int, default=1)
    args = parser.parse_args()
    if args.tested_strategies < 1:
        parser.error("--tested-strategies must be positive")
    report = run_training(
        ohlcv_path=args.ohlcv,
        microstructure_path=args.microstructure,
        model_path=args.model,
        report_path=args.report,
        tested_strategies=args.tested_strategies,
    )
    print(
        json.dumps(
            {
                "status": "RESEARCH_COMPLETE_NOT_PROMOTED",
                "model_path": report["artifact"]["path"],
                "training_rows": report["dataset"]["eligible_labeled_rows"],
                "validation": report["research_result"],
                "safety": report["safety"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
