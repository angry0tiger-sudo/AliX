"""Re-evaluate an existing RL checkpoint without training or changing its weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


from services.rl_research.rl_trading_environment import (
    DEFAULT_INITIAL_EQUITY,
    MarketObservationScaler,
)
from scripts.run_rl_trading_agent import (
    _economic_assessment,
    _evaluate_baseline,
    _evaluate_policy,
    _exclude_previous_holdout_window,
    _load_market_data,
    _sha256,
    _split_training_validation,
    _utc_from_ms,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/rl-agent-evaluation-only"),
    )
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser.parse_args(argv)


def _validate_source_artifacts(
    args: argparse.Namespace, source_report: dict[str, Any], manifest: dict[str, Any]
) -> str:
    if source_report.get("schema_version") != "alix.rl-trading-agent-report.v3":
        raise ValueError("source_report_schema_invalid")
    expected_model_hash = source_report.get("model", {}).get("model_sha256")
    if (
        not isinstance(expected_model_hash, str)
        or _sha256(args.model) != expected_model_hash
    ):
        raise ValueError("source_model_sha256_mismatch")
    expected_data_hash = source_report.get("dataset", {}).get("csv_sha256")
    if manifest.get("csv_sha256") != expected_data_hash:
        raise ValueError("source_dataset_sha256_mismatch")
    safety = source_report.get("safety_invariants", {})
    if safety.get("Champion", "invalid") is not None:
        raise ValueError("source_report_champion_invariant_invalid")
    if safety.get("order_permission") is not False:
        raise ValueError("source_report_order_permission_invariant_invalid")
    if safety.get("trading_authorization") != "NO-GO":
        raise ValueError("source_report_trading_authorization_invariant_invalid")
    return expected_model_hash


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    data, manifest = _load_market_data(args.data, args.manifest)
    model_sha256 = _validate_source_artifacts(args, source_report, manifest)

    analysis_data, excluded_rows = _exclude_previous_holdout_window(data)
    train_data, validation_data, validation_end = _split_training_validation(
        analysis_data
    )
    holdout_data = analysis_data[validation_end:]
    scaler = MarketObservationScaler.fit(train_data)

    model_metadata = source_report["model"]
    if model_metadata.get("algorithm") != "RecurrentPPO":
        raise ValueError("source_model_algorithm_invalid")
    recurrent_ppo = getattr(
        __import__("sb3_contrib", fromlist=["RecurrentPPO"]), "RecurrentPPO"
    )
    model = recurrent_ppo.load(str(args.model.with_suffix("")), env=None, device="cpu")

    execution = source_report["environment"]["execution"]
    fee_bps = float(execution["fee_bps_per_side"])
    slippage_bps = float(execution["slippage_bps_per_side"])
    validation = _evaluate_policy(
        model, validation_data, scaler, fee_bps, slippage_bps, args.seed
    )
    validation_buy_and_hold = _evaluate_baseline(
        validation_data, scaler, fee_bps, slippage_bps, "buy_and_hold"
    )
    validation_cash = _evaluate_baseline(
        validation_data, scaler, fee_bps, slippage_bps, "cash"
    )
    holdout = _evaluate_policy(
        model, holdout_data, scaler, fee_bps, slippage_bps, args.seed + 1
    )
    holdout_buy_and_hold = _evaluate_baseline(
        holdout_data, scaler, fee_bps, slippage_bps, "buy_and_hold"
    )
    holdout_cash = _evaluate_baseline(
        holdout_data, scaler, fee_bps, slippage_bps, "cash"
    )
    if holdout["steps"] != len(holdout_data) - 1:
        raise RuntimeError("holdout_evaluation_did_not_cover_exactly_one_pass")
    if (
        validation["initial_equity"] != DEFAULT_INITIAL_EQUITY
        or holdout["initial_equity"] != DEFAULT_INITIAL_EQUITY
    ):
        raise RuntimeError("evaluation_initial_equity_mismatch")

    economic = _economic_assessment(validation, holdout)
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "orders_or_execution_api_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "model_output_is_order_permission": False,
        "automatic_promotion_allowed": False,
    }
    report: dict[str, Any] = {
        "schema_version": "alix.rl-trading-agent-evaluation-only.v1",
        "run_mode": "historical_simulation_only_evaluation_only",
        "training_performed": False,
        "model_weights_modified": False,
        "evaluation_commit_sha": args.commit_sha,
        "source_training_commit_sha": source_report["commit_sha"],
        "source_run_report_schema": source_report["schema_version"],
        "model": {
            "algorithm": model_metadata["algorithm"],
            "seed_used_for_training": model_metadata["seed"],
            "training_timesteps": model_metadata["total_timesteps_trained"],
            "ent_coef": model_metadata["ent_coef"],
            "model_artifact": args.model.name,
            "model_sha256": model_sha256,
            "inference_mode": "deterministic",
        },
        "dataset": {
            "symbol": manifest["symbol"],
            "interval": manifest["interval"],
            "row_count": len(data),
            "csv_sha256": manifest["csv_sha256"],
            "start_utc": manifest["start_utc"],
            "end_utc": manifest["end_utc"],
        },
        "split": {
            "training_rows_for_scaler_fit_only": len(train_data),
            "validation_rows": len(validation_data),
            "final_holdout_rows": len(holdout_data),
            "scaler_fit_on_training_only": True,
            "validation_used_for_model_selection": False,
            "final_holdout_used_for_training": False,
            "final_holdout_pass_count": 1,
            "previously_reported_holdout_excluded": True,
            "rows_excluded_after_analysis_window": excluded_rows,
            "validation_start_utc": _utc_from_ms(float(validation_data[0, 0])),
            "validation_end_utc": _utc_from_ms(float(validation_data[-1, 0])),
            "final_holdout_start_utc": _utc_from_ms(float(holdout_data[0, 0])),
            "final_holdout_end_utc": _utc_from_ms(float(holdout_data[-1, 0])),
        },
        "evaluation": {
            "validation": validation,
            "validation_buy_and_hold": validation_buy_and_hold,
            "validation_cash": validation_cash,
            "final_holdout": holdout,
            "final_holdout_buy_and_hold": holdout_buy_and_hold,
            "final_holdout_cash": holdout_cash,
        },
        "economic_assessment": economic,
        "holdout_qualification": (
            "Owner-authorized re-evaluation of the same Run #26 checkpoint and "
            "the same final-holdout rows. This holdout is now considered consumed "
            "for future model selection and must not be described as unseen."
        ),
        "safety_invariants": safety,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "rl-trading-agent-evaluation-only.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
