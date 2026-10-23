"""Train auditable HMM and Meta-Label artifacts from public Nobitex history.

This command never changes repository files, authorizes an order, or promotes a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from services.data_quality.gate import validate_candles
from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    add_market_features,
    forward_return_target,
)
from services.meta_labeling.meta_model import (
    MetaLabelModel,
    OOFFold,
    generate_oof_primary_predictions,
)
from services.meta_labeling.triple_barrier import (
    TripleBarrierConfig,
    apply_triple_barrier,
)
from services.meta_labeling.walkforward import (
    _merge_oof_labels,
    _validate_primary_oof_provenance,
)
from services.strategy_selection.hmm_training import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    calculate_regime_features,
    fit_parameters_for_artifact,
    validate_hmm_holdout,
)
from services.strategy_selection.model_io import MODEL_SCHEMA_VERSION

META_THRESHOLD = 0.70
FEE_BPS_PER_SIDE = 25.0
SLIPPAGE_BPS_PER_SIDE = 5.0
REPORT_SCHEMA_VERSION = "alix.ai-artifact-training-report.v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_dataset(
    dataset_path: Path, manifest_path: Path
) -> tuple[DataFrame, dict[str, Any], str]:
    try:
        raw_bytes = dataset_path.read_bytes()
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ai_training_dataset_or_manifest_unreadable") from error
    if not isinstance(manifest_value, dict):
        raise ValueError("ai_training_manifest_must_be_object")
    digest = _sha256_bytes(raw_bytes)
    if manifest_value.get("csv_sha256") != digest:
        raise ValueError("ai_training_dataset_sha256_mismatch")
    if manifest_value.get("provider") != "Nobitex public UDF history API":
        raise ValueError("ai_training_provider_not_approved_for_research")
    if manifest_value.get("credentials_used") is not False:
        raise ValueError("ai_training_public_dataset_must_use_no_credentials")
    if (
        manifest_value.get("row_count") != 8760
        or manifest_value.get("requested_days") != 365
    ):
        raise ValueError("ai_training_dataset_must_be_8760_hourly_candles")
    candles = pd.read_csv(dataset_path)
    required = ("date", "open", "high", "low", "close", "volume")
    if tuple(candles.columns) != required:
        raise ValueError("ai_training_csv_schema_invalid")
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="raise")
    if len(candles) != 8760:
        raise ValueError("ai_training_row_count_mismatch")
    if (
        candles["date"].duplicated().any()
        or not candles["date"].is_monotonic_increasing
    ):
        raise ValueError("ai_training_timestamps_must_be_unique_and_increasing")
    deltas = candles["date"].diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if not np.all(deltas == 3600.0):
        raise ValueError("ai_training_gap_or_timeframe_mismatch_no_interpolation")
    for name in ("open", "high", "low", "close", "volume"):
        candles[name] = pd.to_numeric(candles[name], errors="coerce")
        values = candles[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"ai_training_non_finite_ohlcv:{name}")
    opened = candles["open"].to_numpy(dtype=float)
    high = candles["high"].to_numpy(dtype=float)
    low = candles["low"].to_numpy(dtype=float)
    close = candles["close"].to_numpy(dtype=float)
    volume = candles["volume"].to_numpy(dtype=float)
    if (
        np.any(np.minimum.reduce([opened, high, low, close]) <= 0)
        or np.any(volume < 0)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
        or np.any(high < low)
    ):
        raise ValueError("ai_training_ohlcv_invariants_failed")
    quality = validate_candles(
        [
            {str(key): value for key, value in row.items()}
            for row in candles.to_dict(orient="records")
        ],
        expected_interval_seconds=3600,
        gap_policy="reject",
    )
    if not quality.passed or quality.fingerprint is None:
        raise ValueError("ai_training_shared_data_quality_gate_rejected_dataset")
    manifest_value["data_quality_fingerprint"] = quality.fingerprint
    return candles, manifest_value, digest


def _folds(row_count: int) -> list[OOFFold]:
    starts = (1800, 3000, 4200, 5400, 6600, 7800)
    folds: list[OOFFold] = []
    for index, start in enumerate(starts):
        stop = min(start + 1200, row_count - 1)
        if stop - start < 100:
            continue
        folds.append(OOFFold(0, start, start, stop))
    if len(folds) < 4:
        raise ValueError("ai_training_walk_forward_folds_insufficient")
    return folds


def _meta_training(
    candles: DataFrame, data_hash: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    enriched = add_market_features(candles)
    feature_frame = enriched.loc[:, list(FEATURE_COLUMNS)]
    forward_one = forward_return_target(candles, horizon=1).to_numpy(dtype=float)
    primary_target = pd.Series(
        np.where(np.isfinite(forward_one), (forward_one > 0.0).astype(float), np.nan)
    )
    label_ends = np.arange(len(candles), dtype=int) + 1
    oof = generate_oof_primary_predictions(
        feature_frame,
        primary_target,
        label_ends,
        _folds(len(candles)),
        lambda: LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, solver="lbfgs"
        ),
    )
    event_indices = oof["event_index"].to_numpy(dtype=int)
    volatility = (
        feature_frame["%-volatility_12"].iloc[event_indices].to_numpy(dtype=float)
    )
    valid_event = (
        np.isfinite(volatility)
        & (volatility > 0)
        & (oof["primary_prediction"].to_numpy(dtype=float) >= 0.5)
    )
    events = pd.DataFrame(
        {
            "event_index": event_indices[valid_event],
            "side": np.ones(int(valid_event.sum()), dtype=int),
            "target_volatility": volatility[valid_event],
        }
    )
    barrier_config = TripleBarrierConfig(
        profit_taking_multiplier=1.5,
        stop_loss_multiplier=1.0,
        max_holding_bars=12,
        fee_bps_per_side=FEE_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
        timeframe_seconds=3600,
    )
    labels = apply_triple_barrier(candles, events, barrier_config)
    merged = _merge_oof_labels(oof, labels)
    _validate_primary_oof_provenance(merged)
    indices = merged["event_index"].to_numpy(dtype=int)
    meta_features = feature_frame.iloc[indices].copy()
    meta_features.index = indices
    finite = np.isfinite(meta_features.to_numpy(dtype=float)).all(axis=1)
    merged = merged.loc[finite].reset_index(drop=True)
    meta_features = meta_features.loc[merged["event_index"].to_numpy(dtype=int)].copy()
    event_ids = merged["event_index"].to_numpy(dtype=int)
    meta_features.index = event_ids
    meta_labels = merged["meta_label"].to_numpy(dtype=int)
    meta_ends = merged["label_end_index"].to_numpy(dtype=int)
    provenance = merged.loc[
        :,
        [
            "event_index",
            "fold_id",
            "primary_train_max_index",
            "primary_train_max_label_end_index",
        ],
    ].reset_index(drop=True)
    if len(merged) < 200 or len(np.unique(meta_labels)) != 2:
        raise ValueError("ai_meta_training_sample_or_class_diversity_insufficient")

    validation_cutoff = int(len(candles) * 0.80)
    train_mask = (event_ids < validation_cutoff) & (meta_ends < validation_cutoff)
    test_mask = (event_ids >= validation_cutoff) & (meta_ends < len(candles))
    train_positions = np.flatnonzero(train_mask)
    test_positions = np.flatnonzero(test_mask)
    validation: dict[str, Any] = {
        "status": "INSUFFICIENT_SAMPLE",
        "passed": False,
        "threshold": META_THRESHOLD,
        "threshold_semantics": "research filter only; not risk authorization",
        "training_count": int(len(train_positions)),
        "test_count": int(len(test_positions)),
        "economic_acceptance": "NO-GO",
    }
    if len(train_positions) >= 100 and len(test_positions) >= 100:
        train_frame = meta_features.iloc[train_positions].copy()
        train_labels = meta_labels[train_positions]
        train_provenance = provenance.iloc[train_positions].reset_index(drop=True)
        train_ends = meta_ends[train_positions]
        if len(np.unique(train_labels)) == 2:
            validation_model = MetaLabelModel().fit(
                train_frame,
                train_labels,
                train_provenance,
                train_ends,
                evaluation_cutoff_index=validation_cutoff,
            )
            test_frame = meta_features.iloc[test_positions].copy()
            probabilities = validation_model.predict_probability(test_frame)
            test_labels = meta_labels[test_positions].astype(float)
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            brier = float(np.mean((probabilities - test_labels) ** 2))
            log_loss = float(
                -np.mean(
                    test_labels * np.log(clipped)
                    + (1.0 - test_labels) * np.log(1.0 - clipped)
                )
            )
            selected = probabilities >= META_THRESHOLD
            net_returns = merged.iloc[test_positions]["net_return"].to_numpy(
                dtype=float
            )
            selected_returns = net_returns[selected]
            validation = {
                "status": "TECHNICAL_VALIDATION_PASSED",
                "passed": bool(
                    math.isfinite(brier)
                    and math.isfinite(log_loss)
                    and len(test_labels) >= 100
                ),
                "threshold": META_THRESHOLD,
                "threshold_semantics": "research filter only; not risk authorization",
                "training_count": int(len(train_positions)),
                "test_count": int(len(test_positions)),
                "brier_score": brier,
                "log_loss": log_loss,
                "baseline_probability": float(np.mean(train_labels)),
                "filter_pass_count": int(np.sum(selected)),
                "filter_pass_fraction": float(np.mean(selected)),
                "all_proposal_mean_net_return": float(np.mean(net_returns)),
                "filtered_mean_net_return": (
                    float(np.mean(selected_returns)) if len(selected_returns) else None
                ),
                "dataset_sha256": data_hash,
                "economic_acceptance": "NO-GO",
            }
    cutoff_all = len(candles)
    final_mask = meta_ends < cutoff_all
    final_positions = np.flatnonzero(final_mask)
    final_frame = meta_features.iloc[final_positions].copy()
    final_labels = meta_labels[final_positions]
    final_provenance = provenance.iloc[final_positions].reset_index(drop=True)
    final_model = MetaLabelModel().fit(
        final_frame,
        final_labels,
        final_provenance,
        meta_ends[final_positions],
        evaluation_cutoff_index=cutoff_all,
    )
    json_model = final_model.to_dict()
    model_hash = _sha256_bytes(final_model.to_json().encode("utf-8"))
    manifest = {
        "schema_version": "alix.meta-label-model-manifest.v1",
        "model_type": "standardized_logistic_regression",
        "model_sha256": model_hash,
        "dataset_sha256": data_hash,
        "dataset_rows": int(len(candles)),
        "feature_names": list(final_model.feature_names_),
        "training_event_count": int(len(final_positions)),
        "training_labels": {
            "positive": int(np.sum(final_labels == 1)),
            "negative": int(np.sum(final_labels == 0)),
        },
        "primary_model": "purged chronological out-of-fold logistic direction classifier",
        "meta_label": "triple-barrier net return strictly positive after modeled costs",
        "triple_barrier": barrier_config.to_dict(),
        "owner_approved": False,
        "economic_acceptance": "NO-GO",
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    return {
        "model": json_model,
        "manifest": manifest,
        "validation": validation,
        "barrier_config": barrier_config.to_dict(),
    }, validation


def train(
    dataset_path: Path, manifest_path: Path, output_dir: Path, commit_sha: str
) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("ai_training_commit_sha_invalid")
    candles, dataset, data_hash = _load_dataset(dataset_path, manifest_path)
    regime_frame = calculate_regime_features(candles).dropna().reset_index(drop=True)
    if len(regime_frame) < 1000:
        raise ValueError("ai_training_regime_sample_count_insufficient")
    cutoff = int(len(regime_frame) * 0.80)
    hmm_validation = validate_hmm_holdout(
        regime_frame.iloc[:cutoff].loc[:, list(FEATURE_NAMES)],
        regime_frame.iloc[cutoff:].loc[:, list(FEATURE_NAMES)],
    )
    if hmm_validation.get("passed") is not True:
        raise ValueError("ai_training_hmm_technical_validation_failed")
    fitted = fit_parameters_for_artifact(regime_frame.loc[:, list(FEATURE_NAMES)])
    trained_at = datetime.now(timezone.utc).isoformat()
    revision = f"nobitex-btcusdt-1h-hmm-{data_hash[:12]}-v1"
    model: dict[str, Any] = {
        "revision": revision,
        "trained_at": trained_at,
        "owner_approved": False,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "regimes": ["trend", "range", "high_volatility"],
        "initial_probabilities": fitted["initial_probabilities"],
        "transition_matrix": fitted["transition_matrix"],
        "emissions": fitted["emissions"],
        "max_model_age_seconds": 30 * 24 * 60 * 60,
        "max_observation_age_seconds": 2 * 3600,
        "minimum_confidence": "0.55",
        "minimum_posterior_margin": "0.10",
    }
    model_hash = _sha256_bytes(_canonical_json(model).encode("utf-8"))
    hmm_artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "TRAINED_UNAPPROVED",
        "model_sha256": model_hash,
        "model": model,
        "dataset": {
            "provider": dataset["provider"],
            "csv_sha256": data_hash,
            "row_count": int(len(candles)),
            "symbol": dataset.get("symbol", "BTCUSDT"),
            "pair": dataset.get("pair", "BTC/USDT"),
            "timeframe": "1h",
            "start": candles["date"].iloc[0].isoformat(),
            "end": candles["date"].iloc[-1].isoformat(),
            "data_quality_fingerprint": dataset["data_quality_fingerprint"],
        },
        "validation": hmm_validation,
        "training": {
            "fit_rows": int(len(regime_frame)),
            "validation_rows": int(hmm_validation["validation_rows"]),
            "fit_diagnostics": fitted["fit_diagnostics"],
            "training_algorithm": "diagonal Gaussian HMM Baum-Welch",
            "random_seed": 42,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "economic_acceptance": "NO-GO",
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    meta_result, meta_validation = _meta_training(candles, data_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hmm-regime-model.json").write_text(
        json.dumps(hmm_artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "meta-label-model.json").write_text(
        json.dumps(meta_result["model"], indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "meta-label-model.manifest.json").write_text(
        json.dumps(meta_result["manifest"], indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "dataset_sha256": data_hash,
        "dataset_rows": int(len(candles)),
        "dataset_start": candles["date"].iloc[0].isoformat(),
        "dataset_end": candles["date"].iloc[-1].isoformat(),
        "data_quality_fingerprint": dataset["data_quality_fingerprint"],
        "hmm": {
            "model_revision": revision,
            "model_sha256": model_hash,
            "validation": hmm_validation,
            "owner_approved": False,
        },
        "meta_model": {
            "model_sha256": meta_result["manifest"]["model_sha256"],
            "training_event_count": meta_result["manifest"]["training_event_count"],
            "validation": meta_validation,
            "owner_approved": False,
        },
        "modeled_costs": {
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "source": "conservative research assumptions; not account-specific verified costs",
        },
        "Champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "economic_acceptance": "NO-GO",
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
        "account_data_used": False,
    }
    (output_dir / "ai-training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    result = train(args.dataset, args.manifest, args.output_dir, args.commit_sha)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
