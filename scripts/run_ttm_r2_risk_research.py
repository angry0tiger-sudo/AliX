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
    ForecastContextOnlyDataset,
    forecast_dataset_indices,
    make_evaluation_origins,
    prepare_risk_target,
    risk_forecast_gate,
    score_variance_forecasts,
)
from research.ai_ensemble.ttm_portfolio import (
    attach_asof_risk_multipliers,
    build_oos_teacher_trade_events,
    evaluate_sized_portfolio,
)

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
MODEL_BRANCH = "main (512-96-r2)"
MODEL_REVISION = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
MODEL_WEIGHTS_SHA256 = (
    "a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710"
)
MODEL_CONTEXT_LENGTH = 512
MODEL_NATIVE_PREDICTION_LENGTH = 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_source_policy(manifest: dict[str, Any]) -> None:
    if manifest.get("source_policy") != "NOBITEX_ONLY_NO_BACKFILL":
        raise ValueError("Dataset source policy is not NOBITEX_ONLY_NO_BACKFILL.")
    if manifest.get("dataset_window") != "PRE_HACK_NOBITEX_ONLY":
        raise ValueError("Dataset is not the approved pre-hack Nobitex window.")
    if manifest.get("data_sources") != ["Nobitex public UDF history API"]:
        raise ValueError(
            "Unexpected data source or external exchange backfill detected."
        )


def _verify_manifest_boundaries(manifest: dict[str, Any]) -> None:
    expected_values = {
        "row_count": EXPECTED_ROWS,
        "expected_candles": EXPECTED_ROWS,
        "start_timestamp": EXPECTED_START_UNIX,
        "end_timestamp": EXPECTED_END_UNIX,
        "missing_intervals": 0,
    }
    for key, expected in expected_values.items():
        if int(manifest.get(key, -1)) != expected:
            raise ValueError(f"Unexpected immutable dataset boundary: {key}.")


def _verify_manifest_integrity(
    manifest: dict[str, Any], actual_sha: str, commit_sha: str
) -> None:
    if manifest.get("csv_sha256") != DATASET_SHA256 or actual_sha != DATASET_SHA256:
        raise ValueError("Immutable Nobitex dataset SHA-256 verification failed.")
    if manifest.get("commit_sha") != commit_sha:
        raise ValueError("Dataset manifest commit SHA does not match this Actions run.")
    if (
        manifest.get("public_api_only") is not True
        or manifest.get("credentials_used") is not False
    ):
        raise ValueError("Public unauthenticated market data is required.")


def _verify_manifest_safety(manifest: dict[str, Any]) -> None:
    safety_counters = (
        "private_api_calls",
        "exchange_calls",
        "real_order_attempts",
    )
    if any(int(manifest.get(key, -1)) != 0 for key in safety_counters):
        raise ValueError("Safety invariant violated in data manifest.")
    if (
        manifest.get("Champion") is not None
        or manifest.get("order_permission") is not False
    ):
        raise ValueError("Champion/order permission invariant violated.")
    if manifest.get("trading_authorization") != "NO-GO":
        raise ValueError("Trading authorization must remain NO-GO.")


def verify_dataset(
    csv_path: Path, manifest_path: Path, commit_sha: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = cast(
        dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    actual_sha = sha256_file(csv_path)
    _verify_manifest_source_policy(manifest)
    _verify_manifest_boundaries(manifest)
    _verify_manifest_integrity(manifest, actual_sha, commit_sha)
    _verify_manifest_safety(manifest)
    ohlcv = pd.read_csv(csv_path)
    if len(ohlcv) != EXPECTED_ROWS:
        raise ValueError("CSV row count does not match the immutable manifest.")
    return ohlcv, manifest


def _predict_context_only(trainer: Any, dataset: Any, origins: list[int]) -> np.ndarray:
    indices = forecast_dataset_indices(origins)
    prediction_dataset = ForecastContextOnlyDataset(dataset, indices)
    prediction_output = trainer.predict(prediction_dataset)
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
    return prediction_array


def _inverse_forecast_variance(
    prediction_array: np.ndarray,
    preprocessor: Any,
    origin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    inverse_scaled = preprocessor.inverse_scale_targets(
        pd.DataFrame({TARGET_COLUMN: prediction_array.reshape(-1)})
    )
    predicted_log_variance = (
        inverse_scaled[TARGET_COLUMN]
        .to_numpy(dtype=np.float64)
        .reshape(origin_count, PREDICTION_HORIZON)
    )
    predicted_hourly_variance = np.expm1(predicted_log_variance)
    if not np.isfinite(predicted_hourly_variance).all() or np.any(
        predicted_hourly_variance <= 0
    ):
        raise ValueError(
            "TTM-R2 produced a non-positive or non-finite variance forecast; reject."
        )
    forecast_variance = predicted_hourly_variance.sum(axis=1)
    return forecast_variance, predicted_log_variance


def _realized_variance_windows(
    origins: list[int], return_variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    return actual, persistence


def run_split(
    trainer: Any,
    dataset: Any,
    preprocessor: Any,
    origins: list[int],
    return_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction_array = _predict_context_only(trainer, dataset, origins)
    forecast_variance, predicted_log_variance = _inverse_forecast_variance(
        prediction_array, preprocessor, len(origins)
    )
    actual, persistence = _realized_variance_windows(origins, return_variance)
    return forecast_variance, actual, persistence, predicted_log_variance[:, -1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TTM-R2 risk-only zero-shot research on pre-hack Nobitex."
    )
    parser.add_argument("--ohlcv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    return parser


def _prepare_experiment_inputs(args: argparse.Namespace) -> dict[str, Any]:
    ohlcv, data_manifest = verify_dataset(args.ohlcv, args.manifest, args.commit_sha)
    frame = prepare_risk_target(ohlcv)
    split = make_evaluation_origins(len(frame))
    return {
        "ohlcv": ohlcv,
        "data_manifest": data_manifest,
        "frame": frame,
        "return_variance": frame["return_variance"].to_numpy(dtype=np.float64),
        "split": split,
    }


def _load_pinned_model_weights(hub: Any) -> tuple[str, str]:
    weights_path = Path(
        hub.hf_hub_download(
            repo_id=MODEL_ID,
            filename="model.safetensors",
            revision="d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
        )
    )
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != MODEL_WEIGHTS_SHA256:
        raise ValueError(
            "TTM-R2 model weight SHA-256 does not match the pinned artifact."
        )
    return MODEL_REVISION, weights_sha256


def _validate_model_configuration(model: Any) -> None:
    config = model.config
    if int(config.context_length) != CONTEXT_LENGTH:
        raise ValueError(f"Unexpected TTM context length: {config.context_length}")
    if int(config.prediction_length) != MODEL_NATIVE_PREDICTION_LENGTH:
        raise ValueError(
            f"Unexpected native TTM prediction length: {config.prediction_length}"
        )
    if int(model.prediction_filter_length) != PREDICTION_HORIZON:
        raise ValueError(
            "TTM prediction_filter_length does not match the registered 24h horizon."
        )


def _build_inference_runtime(frame: pd.DataFrame, train_end: int) -> dict[str, Any]:
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
    preprocessor.train(frame.iloc[:train_end].copy())
    processed_frame = preprocessor.preprocess(frame.copy())
    model_revision, weights_sha256 = _load_pinned_model_weights(hub)
    model = get_model_module.get_model(
        model_path=MODEL_ID,
        context_length=MODEL_CONTEXT_LENGTH,
        prediction_length=PREDICTION_HORIZON,
        freq_prefix_tuning=False,
        prefer_l1_loss=False,
        prefer_longer_context=True,
        model_revision=model_revision,
    )
    _validate_model_configuration(model)
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
    return {
        "trainer": trainer,
        "dataset": dataset,
        "preprocessor": preprocessor,
        "model_revision": model_revision,
        "weights_sha256": weights_sha256,
        "tsfm_version": getattr(tsfm_public, "__version__", "unknown"),
    }


def _score_forecast_splits(
    runtime: dict[str, Any],
    split: dict[str, Any],
    return_variance: np.ndarray,
) -> dict[str, Any]:
    validation = run_split(
        runtime["trainer"],
        runtime["dataset"],
        runtime["preprocessor"],
        split["validation_origins"],
        return_variance,
    )
    holdout = run_split(
        runtime["trainer"],
        runtime["dataset"],
        runtime["preprocessor"],
        split["holdout_origins"],
        return_variance,
    )
    validation_metrics = score_variance_forecasts(
        validation[1], validation[0], validation[2]
    )
    holdout_metrics = score_variance_forecasts(holdout[1], holdout[0], holdout[2])
    accepted = risk_forecast_gate(holdout_metrics)
    return {
        "validation": validation,
        "holdout": holdout,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "verdict": ("PASS_RISK_FORECAST_RESEARCH_ONLY" if accepted else "NO_GO"),
    }


def _build_position_risk_inputs(
    split: dict[str, Any], forecasts: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    forecast_market_indices = np.asarray(
        [
            origin + 1
            for origin in (split["validation_origins"] + split["holdout_origins"])
        ],
        dtype=np.int64,
    )
    validation_forecast = forecasts["validation"]
    holdout_forecast = forecasts["holdout"]
    forecast_values = np.concatenate([validation_forecast[0], holdout_forecast[0]])
    persistence_values = np.concatenate([validation_forecast[2], holdout_forecast[2]])
    risk_multipliers = np.minimum(
        1.0,
        np.sqrt(np.maximum(persistence_values, 1e-18) / forecast_values),
    )
    return forecast_market_indices, risk_multipliers


def _compare_sized_portfolio(trades: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = evaluate_sized_portfolio(trades, risk_managed=False)
    managed = evaluate_sized_portfolio(trades, risk_managed=True)
    baseline_pf = fixed["net_profit_factor"]
    managed_pf = managed["net_profit_factor"]
    baseline_drawdown = fixed["max_drawdown"]
    managed_drawdown = managed["max_drawdown"]
    comparison = {
        "net_profit_factor_improved": (
            baseline_pf is not None
            and managed_pf is not None
            and float(managed_pf) > float(baseline_pf)
        ),
        "max_drawdown_reduced": (
            baseline_drawdown is not None
            and managed_drawdown is not None
            and float(managed_drawdown) < float(baseline_drawdown)
        ),
        "max_drawdown_below_15_percent": (
            managed_drawdown is not None and float(managed_drawdown) < 0.15
        ),
        "same_trade_count": fixed["trade_count"] == managed["trade_count"],
    }
    comparison["both_objectives_met"] = bool(
        comparison["net_profit_factor_improved"]
        and comparison["max_drawdown_reduced"]
        and comparison["max_drawdown_below_15_percent"]
    )
    return {
        "fixed_size_baseline": fixed,
        "ttm_risk_managed": managed,
        "comparison": comparison,
    }


def _evaluate_portfolio_splits(
    ohlcv: pd.DataFrame,
    split: dict[str, Any],
    forecasts: dict[str, Any],
) -> dict[str, Any]:
    teacher = build_oos_teacher_trade_events(
        ohlcv,
        training_end_market_index=int(split["train_end"]) + 1,
        validation_end_market_index=int(split["validation_end"]) + 1,
    )
    forecast_indices, risk_multipliers = _build_position_risk_inputs(split, forecasts)
    validation_trades = attach_asof_risk_multipliers(
        teacher["validation_trades"], forecast_indices, risk_multipliers
    )
    holdout_trades = attach_asof_risk_multipliers(
        teacher["holdout_trades"], forecast_indices, risk_multipliers
    )
    validation_results = _compare_sized_portfolio(validation_trades)
    holdout_results = _compare_sized_portfolio(holdout_trades)
    evaluation = {
        "entry_teacher": {
            "model": "OHLCV-only LightGBM/XGBoost + causal HMM + logistic stack",
            "training_rows": teacher["training_rows"],
            "validation_rows": teacher["validation_rows"],
            "holdout_rows": teacher["holdout_rows"],
            "probability_threshold": teacher["probability_threshold"],
            "round_trip_cost_bps": teacher["round_trip_cost_bps"],
            "uses_chronos_features": False,
            "ttm_generates_entry_signals": False,
        },
        "position_sizing": {
            "fixed_size_baseline_fraction_of_equity": 0.10,
            "ttm_position_fraction": (
                "0.10 * min(1, sqrt(trailing_24h_realized_variance / "
                "TTM_forecast_24h_variance))"
            ),
            "forecast_refresh_bars": 24,
            "input_interval": "1h",
            "leverage": False,
            "trading_costs_bps_round_trip": 50,
            "applied_to_runtime": False,
        },
        "validation": {
            "fixed_size_baseline": validation_results["fixed_size_baseline"],
            "ttm_risk_managed": validation_results["ttm_risk_managed"],
        },
        "final_holdout": {
            "fixed_size_baseline": holdout_results["fixed_size_baseline"],
            "ttm_risk_managed": holdout_results["ttm_risk_managed"],
        },
        "final_holdout_comparison": holdout_results["comparison"],
        "trade_counts": {
            "validation": len(validation_trades),
            "final_holdout": len(holdout_trades),
        },
        "equity_curve_method": (
            "Sequential compounded equity at realized trade exits; max drawdown "
            "is measured on the trade-exit equity curve. Positions do not overlap."
        ),
    }
    return {
        "evaluation": evaluation,
        "validation_trades": validation_trades,
        "holdout_trades": holdout_trades,
    }


def _build_forecast_rows(
    frame: pd.DataFrame, split: dict[str, Any], forecasts: dict[str, Any]
) -> list[dict[str, Any]]:
    validation = forecasts["validation"]
    holdout = forecasts["holdout"]
    rows: list[dict[str, Any]] = []
    for split_name, origins, result in (
        ("validation", split["validation_origins"], validation),
        ("holdout", split["holdout_origins"], holdout),
    ):
        for origin, forecast, actual, baseline in zip(
            origins, result[0], result[1], result[2]
        ):
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
                    "risk_only_position_multiplier": min(
                        1.0, float(np.sqrt(baseline / forecast))
                    ),
                }
            )
    return rows


def _build_portfolio_rows(
    ohlcv: pd.DataFrame, portfolio: dict[str, Any]
) -> list[dict[str, Any]]:
    trades = portfolio["validation_trades"] + portfolio["holdout_trades"]
    return [
        {
            "split": trade["split"],
            "entry_market_index": trade["entry_market_index"],
            "exit_market_index": trade["exit_market_index"],
            "entry_time": str(ohlcv.iloc[int(trade["entry_market_index"])]["date"]),
            "exit_time": str(ohlcv.iloc[int(trade["exit_market_index"])]["date"]),
            "net_trade_return_after_50bps": trade["net_return"],
            "entry_probability": trade["entry_probability"],
            "ttm_risk_multiplier": trade["risk_multiplier"],
            "fixed_position_fraction": 0.10,
            "ttm_position_fraction": 0.10 * trade["risk_multiplier"],
        }
        for trade in trades
    ]


def _build_report(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    runtime: dict[str, Any],
    forecasts: dict[str, Any],
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    ohlcv = inputs["ohlcv"]
    data_manifest = inputs["data_manifest"]
    frame = inputs["frame"]
    evaluation = portfolio["evaluation"]
    return {
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
            "resolved_immutable_revision": runtime["model_revision"],
            "weights_sha256": runtime["weights_sha256"],
            "granite_tsfm_version": runtime["tsfm_version"],
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
            "trading_costs_note": "Forecast metrics are cost-free; portfolio labels apply 50 bps round-trip. The two evaluations answer separate questions.",
        },
        "validation_metrics": forecasts["validation_metrics"],
        "final_holdout_metrics": forecasts["holdout_metrics"],
        "portfolio_evaluation": evaluation,
        "risk_forecast_acceptance": {
            "status": forecasts["verdict"],
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
            "offline_position_sizing_simulated": True,
        },
    }


def _write_artifacts(
    output_dir: Path,
    frame: pd.DataFrame,
    split: dict[str, Any],
    inputs: dict[str, Any],
    runtime: dict[str, Any],
    forecasts: dict[str, Any],
    portfolio: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    forecast_rows = _build_forecast_rows(frame, split, forecasts)
    portfolio_rows = _build_portfolio_rows(inputs["ohlcv"], portfolio)
    pd.DataFrame(forecast_rows).to_csv(
        output_dir / "ttm-r2-risk-forecasts.csv", index=False
    )
    pd.DataFrame(portfolio_rows).to_csv(
        output_dir / "ttm-r2-risk-portfolio-trades.csv", index=False
    )
    report = _build_report(args, inputs, runtime, forecasts, portfolio)
    report_path = output_dir / "ttm-r2-risk-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": forecasts["verdict"],
                "holdout": forecasts["holdout_metrics"],
                "portfolio_holdout": report["portfolio_evaluation"]["final_holdout"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return report


def main() -> int:
    args = build_parser().parse_args()
    if len(args.commit_sha) != 40:
        raise ValueError("--commit-sha must be a full 40-character Git SHA.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _prepare_experiment_inputs(args)
    split = inputs["split"]
    runtime = _build_inference_runtime(inputs["frame"], int(split["train_end"]))
    forecasts = _score_forecast_splits(runtime, split, inputs["return_variance"])
    portfolio = _evaluate_portfolio_splits(inputs["ohlcv"], split, forecasts)
    _write_artifacts(
        args.output_dir,
        inputs["frame"],
        split,
        inputs,
        runtime,
        forecasts,
        portfolio,
        args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
