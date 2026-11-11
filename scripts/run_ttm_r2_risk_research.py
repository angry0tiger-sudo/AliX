"""Run the pre-registered TTM-R2 zero-shot risk-forecast experiment in GitHub Actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from research.ai_ensemble.ttm_risk import (
    CONTEXT_LENGTH,
    DATASET_SHA256,
    EXPECTED_END_UNIX,
    EXPECTED_ROWS,
    EXPECTED_START_UNIX,
    PREDICTION_HORIZON,
    TARGET_COLUMN,
    forecast_dataset_indices,
    make_evaluation_origins,
    prepare_risk_target,
    risk_forecast_gate,
    score_variance_forecasts,
)

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
MODEL_BRANCH = "main (512-96-r2)"
MODEL_REVISION = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
MODEL_WEIGHTS_SHA256 = "a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710"
MODEL_CONTEXT_LENGTH = 512
MODEL_NATIVE_PREDICTION_LENGTH = 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(
    csv_path: Path, manifest_path: Path, commit_sha: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = cast(
        dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    actual_sha = sha256_file(csv_path)
    if manifest.get("source_policy") != "NOBITEX_ONLY_NO_BACKFILL":
        raise ValueError("Dataset source policy is not NOBITEX_ONLY_NO_BACKFILL.")
    if manifest.get("dataset_window") != "PRE_HACK_NOBITEX_ONLY":
        raise ValueError("Dataset is not the approved pre-hack Nobitex window.")
    if manifest.get("data_sources") != ["Nobitex public UDF history API"]:
        raise ValueError(
            "Unexpected data source or external exchange backfill detected."
        )
    if int(manifest.get("row_count", -1)) != EXPECTED_ROWS:
        raise ValueError("Unexpected Nobitex row count.")
    if int(manifest.get("expected_candles", -1)) != EXPECTED_ROWS:
        raise ValueError("Expected candle count mismatch.")
    if int(manifest.get("start_timestamp", -1)) != EXPECTED_START_UNIX:
        raise ValueError("Unexpected dataset start boundary.")
    if int(manifest.get("end_timestamp", -1)) != EXPECTED_END_UNIX:
        raise ValueError("Unexpected dataset end boundary.")
    if int(manifest.get("missing_intervals", -1)) != 0:
        raise ValueError("Missing intervals are not allowed.")
    if manifest.get("csv_sha256") != DATASET_SHA256 or actual_sha != DATASET_SHA256:
        raise ValueError("Immutable Nobitex dataset SHA-256 verification failed.")
    if manifest.get("commit_sha") != commit_sha:
        raise ValueError("Dataset manifest commit SHA does not match this Actions run.")
    if (
        manifest.get("public_api_only") is not True
        or manifest.get("credentials_used") is not False
    ):
        raise ValueError("Public unauthenticated market data is required.")
    if any(
        int(manifest.get(key, -1)) != 0
        for key in ("private_api_calls", "exchange_calls", "real_order_attempts")
    ):
        raise ValueError("Safety invariant violated in data manifest.")
    if (
        manifest.get("Champion") is not None
        or manifest.get("order_permission") is not False
    ):
        raise ValueError("Champion/order permission invariant violated.")
    if manifest.get("trading_authorization") != "NO-GO":
        raise ValueError("Trading authorization must remain NO-GO.")

    ohlcv = pd.read_csv(csv_path)
    if len(ohlcv) != EXPECTED_ROWS:
        raise ValueError("CSV row count does not match the immutable manifest.")
    return ohlcv, manifest


def run_split(
    trainer: Any,
    dataset: Any,
    preprocessor: Any,
    origins: list[int],
    return_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch_data = import_module("torch.utils.data")
    indices = forecast_dataset_indices(origins)
    prediction_output = trainer.predict(torch_data.Subset(dataset, indices))
    raw_predictions = prediction_output.predictions
    predictions = (
        raw_predictions[0] if isinstance(raw_predictions, tuple) else raw_predictions
    )
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if (
        prediction_array.ndim != 3
        or prediction_array.shape[1] != PREDICTION_HORIZON
        or prediction_array.shape[2] != 1
    ):
        raise ValueError(f"Unexpected TTM prediction shape: {prediction_array.shape}")
    if (
        prediction_array.shape[0] != len(origins)
        or not np.isfinite(prediction_array).all()
    ):
        raise ValueError("TTM prediction count or numeric validation failed.")

    inverse_scaled = preprocessor.inverse_scale_targets(
        pd.DataFrame({TARGET_COLUMN: prediction_array.reshape(-1)})
    )
    predicted_log_variance = (
        inverse_scaled[TARGET_COLUMN]
        .to_numpy(dtype=np.float64)
        .reshape(len(origins), PREDICTION_HORIZON)
    )
    predicted_hourly_variance = np.maximum(np.expm1(predicted_log_variance), 1e-12)
    ttm_forecast = np.maximum(predicted_hourly_variance.sum(axis=1), 1e-12)
    actual = np.asarray(
        [
            return_variance[origin + 1 : origin + PREDICTION_HORIZON + 1].sum()
            for origin in origins
        ],
        dtype=np.float64,
    )
    persistence = np.asarray(
        [
            return_variance[origin - PREDICTION_HORIZON + 1 : origin + 1].sum()
            for origin in origins
        ],
        dtype=np.float64,
    )
    return ttm_forecast, actual, persistence, predicted_log_variance[:, -1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TTM-R2 risk-only zero-shot research on pre-hack Nobitex."
    )
    parser.add_argument("--ohlcv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.commit_sha) != 40:
        raise ValueError("--commit-sha must be a full 40-character Git SHA.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ohlcv, data_manifest = verify_dataset(args.ohlcv, args.manifest, args.commit_sha)
    frame = prepare_risk_target(ohlcv)
    return_variance = frame["return_variance"].to_numpy(dtype=np.float64)
    split = make_evaluation_origins(len(frame))
    train_end = int(split["train_end"])

    tsfm_public = import_module("tsfm_public")
    get_model_module = import_module("tsfm_public.toolkit.get_model")
    transformers = import_module("transformers")
    hub = import_module("huggingface_hub")

    preprocessor = tsfm_public.TimeSeriesPreprocessor(
        id_columns=[],
        timestamp_column="timestamp",
        target_columns=[TARGET_COLUMN],
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_HORIZON,
        scaling=True,
        scaler_type="standard",
        encode_categorical=False,
        freq="h",
    )
    # Fit scaling statistics on the training prefix only; the model itself remains frozen.
    preprocessor.train(frame.iloc[:train_end].copy())
    processed_frame = preprocessor.preprocess(frame.copy())

    model_revision = MODEL_REVISION
    weights_path = Path(
        hub.hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors",
            revision="d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
        )
    )
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != MODEL_WEIGHTS_SHA256:
        raise ValueError("TTM-R2 model weight SHA-256 does not match the pinned artifact.")

    model = get_model_module.get_model(
        model_path=MODEL_ID,
        context_length=MODEL_CONTEXT_LENGTH,
        prediction_length=PREDICTION_HORIZON,
        freq_prefix_tuning=False,
        prefer_l1_loss=False,
        prefer_longer_context=True,
        model_revision=model_revision,
    )
    config = model.config
    if int(config.context_length) != CONTEXT_LENGTH:
        raise ValueError(f"Unexpected TTM context length: {config.context_length}")
    if int(config.prediction_length) != MODEL_NATIVE_PREDICTION_LENGTH:
        raise ValueError(
            f"Unexpected native TTM prediction length: {config.prediction_length}"
        )
    if int(config.prediction_filter_length) != PREDICTION_HORIZON:
        raise ValueError(
            "TTM prediction_filter_length does not match the registered 24h horizon."
        )

    dataset = tsfm_public.ForecastDFDataset(
        data=processed_frame,
        id_columns=[],
        timestamp_column="timestamp",
        target_columns=[TARGET_COLUMN],
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_HORIZON,
        stride=1,
        enable_padding=False,
        impute_method=None,
    )
    training_args = transformers.TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="alix-ttm-r2-"),
        per_device_eval_batch_size=16,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(model=model, args=training_args)

    validation_forecast, validation_actual, validation_baseline, _ = run_split(
        trainer, dataset, preprocessor, split["validation_origins"], return_variance
    )
    holdout_forecast, holdout_actual, holdout_baseline, _ = run_split(
        trainer, dataset, preprocessor, split["holdout_origins"], return_variance
    )
    validation_metrics = score_variance_forecasts(
        validation_actual, validation_forecast, validation_baseline
    )
    holdout_metrics = score_variance_forecasts(
        holdout_actual, holdout_forecast, holdout_baseline
    )
    accepted = risk_forecast_gate(holdout_metrics)
    verdict = "PASS_RISK_FORECAST_RESEARCH_ONLY" if accepted else "NO_GO"

    rows: list[dict[str, Any]] = []
    for split_name, origins, forecasts, actuals, baselines in (
        (
            "validation",
            split["validation_origins"],
            validation_forecast,
            validation_actual,
            validation_baseline,
        ),
        (
            "holdout",
            split["holdout_origins"],
            holdout_forecast,
            holdout_actual,
            holdout_baseline,
        ),
    ):
        for origin, forecast, actual, baseline in zip(
            origins, forecasts, actuals, baselines
        ):
            multiplier = min(1.0, float(np.sqrt(baseline / forecast)))
            rows.append(
                {
                    "split": split_name,
                    "decision_time": frame.iloc[origin]["timestamp"].isoformat(),
                    "horizon_end": frame.iloc[origin + PREDICTION_HORIZON][
                        "timestamp"
                    ].isoformat(),
                    "ttm_forecast_24h_variance": float(forecast),
                    "actual_24h_variance": float(actual),
                    "persistence_24h_variance": float(baseline),
                    "risk_only_position_multiplier": multiplier,
                }
            )
    pd.DataFrame(rows).to_csv(
        args.output_dir / "ttm-r2-risk-forecasts.csv", index=False
    )

    report: dict[str, Any] = {
        "experiment": "TTM-R2 zero-shot 24h realized-variance forecast",
        "code_commit_sha": args.commit_sha,
        "dataset": {
            "source": "Nobitex public UDF history API",
            "source_policy": "NOBITEX_ONLY_NO_BACKFILL",
            "row_count": len(ohlcv),
            "model_target_rows": len(frame),
            "csv_sha256": data_manifest["csv_sha256"],
            "start_timestamp": data_manifest["start_timestamp"],
            "end_timestamp": data_manifest["end_timestamp"],
            "missing_intervals": data_manifest["missing_intervals"],
        },
        "model": {
            "model_id": MODEL_ID,
            "model_branch": MODEL_BRANCH,
            "resolved_immutable_revision": model_revision,
            "weights_sha256": weights_sha256,
            "granite_tsfm_version": getattr(tsfm_public, "__version__", "unknown"),
            "context_length": CONTEXT_LENGTH,
            "native_prediction_length": MODEL_NATIVE_PREDICTION_LENGTH,
            "prediction_filter_length": PREDICTION_HORIZON,
            "mode": "ZERO_SHOT_FROZEN_WEIGHTS",
            "fine_tuned_on_nobitex": False,
        },
        "pretraining_disclosure": {
            "ibm_lists_monash_bitcoin_dataset": True,
            "monash_bitcoin_dataset_publication_date": "2021-07-22",
            "monash_bitcoin_dataset_contents": [
                "18 daily Bitcoin-related influence series",
                "hash rate",
                "block size",
                "mining difficulty",
                "tweets and Google search interest",
            ],
            "no_monash_rows_downloaded_or_joined": True,
            "temporal_overlap_claim": "The documented Monash publication predates the Nobitex series start; full pretraining-corpus overlap is not independently provable.",
        },
        "protocol": {
            "target": "hourly log1p(squared close-to-close log return)",
            "risk_horizon_hours": PREDICTION_HORIZON,
            "chronological_split": "70% train-only scaler fit / 15% validation / final 15% holdout",
            "forecast_origins_are_non_overlapping": True,
            "holdout_used_for_selection": False,
            "model_finetuning": False,
            "external_scaler_fit_on_training_prefix_only": True,
            "missing_data_policy": "REJECT; no interpolation, backfill, or zero padding",
            "baseline": "trailing 24h realized variance persistence",
            "trading_cost_bps_round_trip": 50,
            "trading_costs_applied": False,
            "trading_costs_note": "This is a pure forecast evaluation, not a trading/backtest experiment; any later economic test must apply 50 bps round-trip.",
        },
        "validation_metrics": validation_metrics,
        "final_holdout_metrics": holdout_metrics,
        "risk_forecast_acceptance": {
            "status": verdict,
            "minimum_qlike_improvement_fraction": 0.05,
            "log_variance_mae_must_not_worsen": True,
            "position_multiplier_must_never_exceed": 1.0,
            "decision": "Research evidence only; no automatic deployment or promotion.",
        },
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "ALIX_DRY_RUN": True,
            "ALIX_ALLOW_LIVE_TRADING": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "model_generates_entry_signals": False,
            "model_writes_risk_policy": False,
            "position_multiplier_applied_to_runtime": False,
        },
    }
    report_path = args.output_dir / "ttm-r2-risk-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "holdout": holdout_metrics,
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
