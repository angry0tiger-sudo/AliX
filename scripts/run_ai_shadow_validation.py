#!/usr/bin/env python3
"""Validate a pinned training bundle, prepare approved artifacts, and probe shadow fail-closed behavior.

This tool never trains a model, changes approval documents, selects a strategy, or grants order permission.
It is intended to run only in GitHub Actions against a previously reviewed training artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas import DataFrame

from risk.gate import RiskGate
from services.data_quality.gate import validate_candles
from services.meta_labeling.meta_model import MetaLabelModel
from services.simulation.shadow_exchange import ShadowExchange
from services.strategy_selection.model_io import load_hmm_model_document
from user_data.strategies.AliXStrategy import AliXStrategy

TRAINING_RUN_COMMIT = "5a74ad4585313a2490416a0c3f83947e603c4738"
DATASET_SHA256 = "177e041034fcd91a57bcee59edf6762fcdec85e7a2f4e8b27951f664d9a88856"
APPROVAL_STATUS = "OWNER_APPROVED_TECHNICAL_ONLY"
APPROVAL_SCHEMA = "alix.model-technical-approval.v1"
MODEL_FIELDS = ("date", "open", "high", "low", "close", "volume")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return _sha256_bytes(encoded.encode("utf-8"))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"shadow_validation_json_unreadable:{path.name}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"shadow_validation_json_root_invalid:{path.name}")
    return decoded


def _require_closed_flags(document: Mapping[str, Any], label: str) -> None:
    if (
        document.get("economic_acceptance") != "NO-GO"
        or document.get("order_permission") is not False
        or document.get("trading_authorization") != "NO-GO"
    ):
        raise ValueError(f"shadow_validation_release_flags_invalid:{label}")


def _validate_approval_identity(
    approval: Mapping[str, Any],
    *,
    model_type: str,
    model_hash: str,
    dataset_hash: str,
) -> None:
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA
        or approval.get("model_type") != model_type
        or approval.get("status") != APPROVAL_STATUS
    ):
        raise ValueError(f"shadow_validation_approval_schema_mismatch:{model_type}")
    if approval.get("owner") != "angry0tiger-sudo":
        raise ValueError(f"shadow_validation_approval_owner_mismatch:{model_type}")
    approval_id = approval.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id.strip():
        raise ValueError(f"shadow_validation_approval_id_missing:{model_type}")
    if (
        approval.get("model_sha256") != model_hash
        or approval.get("dataset_sha256") != dataset_hash
    ):
        raise ValueError(f"shadow_validation_approval_hash_mismatch:{model_type}")


def _validate_approval_timestamp(
    approval: Mapping[str, Any], model_type: str
) -> None:
    approved_at = approval.get("approved_at")
    if not isinstance(approved_at, str):
        raise ValueError(f"shadow_validation_approval_timestamp_missing:{model_type}")
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"shadow_validation_approval_timestamp_invalid:{model_type}"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"shadow_validation_approval_timestamp_naive:{model_type}")


def _validate_approval(
    approval: Mapping[str, Any],
    *,
    model_type: str,
    model_hash: str,
    dataset_hash: str,
) -> None:
    _validate_approval_identity(
        approval,
        model_type=model_type,
        model_hash=model_hash,
        dataset_hash=dataset_hash,
    )
    _validate_approval_timestamp(approval, model_type)
    _require_closed_flags(approval, model_type)


def _validate_candle_file_hash(path: Path, manifest: Mapping[str, Any]) -> None:
    if _sha256_bytes(path.read_bytes()) != manifest.get("csv_sha256"):
        raise ValueError("shadow_validation_csv_hash_mismatch")


def _validate_hourly_timestamps(candles: DataFrame) -> None:
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="raise")
    dates = candles["date"]
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("shadow_validation_timestamps_not_unique_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if not np.all(deltas == 3600.0):
        raise ValueError("shadow_validation_gap_or_timeframe_mismatch")


def _validate_ohlc_invariants(
    opened: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> None:
    if np.any(np.minimum.reduce([opened, high, low, close]) <= 0):
        raise ValueError("shadow_validation_prices_not_positive")
    if np.any(volume < 0):
        raise ValueError("shadow_validation_volume_negative")
    if np.any(high < np.maximum(opened, close)):
        raise ValueError("shadow_validation_high_below_open_or_close")
    if np.any(low > np.minimum(opened, close)):
        raise ValueError("shadow_validation_low_above_open_or_close")
    if np.any(high < low):
        raise ValueError("shadow_validation_high_below_low")


def _validate_numeric_ohlcv(candles: DataFrame) -> None:
    for name in ("open", "high", "low", "close", "volume"):
        candles[name] = pd.to_numeric(candles[name], errors="coerce")
    values = candles.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("shadow_validation_nonfinite_ohlcv")
    opened, high, low, close, volume = (values[:, i] for i in range(5))
    _validate_ohlc_invariants(opened, high, low, close, volume)


def _data_quality_fingerprint(candles: DataFrame) -> str:
    rows = [
        {str(key): value for key, value in row.items()}
        for row in candles.to_dict(orient="records")
    ]
    quality = validate_candles(
        rows, expected_interval_seconds=3600, gap_policy="reject"
    )
    if not quality.passed or quality.fingerprint is None:
        raise ValueError("shadow_validation_data_quality_rejected")
    return quality.fingerprint


def _validate_candles(
    path: Path, manifest: Mapping[str, Any], expected_rows: int
) -> tuple[DataFrame, str]:
    _validate_candle_file_hash(path, manifest)
    candles = pd.read_csv(path)
    if tuple(candles.columns) != MODEL_FIELDS:
        raise ValueError("shadow_validation_csv_schema_invalid")
    if len(candles) != expected_rows or manifest.get("row_count") != expected_rows:
        raise ValueError("shadow_validation_candle_count_mismatch")
    _validate_hourly_timestamps(candles)
    _validate_numeric_ohlcv(candles)
    return candles, _data_quality_fingerprint(candles)


def _validate_training_provenance(
    manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    if (
        report.get("commit_sha") != TRAINING_RUN_COMMIT
        or manifest.get("commit_sha") != TRAINING_RUN_COMMIT
    ):
        raise ValueError("shadow_validation_training_commit_mismatch")
    if (
        report.get("dataset_rows") != 8760
        or report.get("dataset_sha256") != DATASET_SHA256
        or manifest.get("csv_sha256") != DATASET_SHA256
    ):
        raise ValueError("shadow_validation_training_dataset_identity_mismatch")
    if (
        manifest.get("provider") != "Nobitex public UDF history API"
        or manifest.get("requested_days") != 365
    ):
        raise ValueError("shadow_validation_training_provider_mismatch")
    if manifest.get("public_api_only") is not True:
        raise ValueError("shadow_validation_training_not_public_only")
    if manifest.get("credentials_used") is not False:
        raise ValueError("shadow_validation_training_credentials_detected")
    if (
        manifest.get("exchange_calls") != 0
        or manifest.get("real_order_attempts") != 0
        or manifest.get("Champion") is not None
    ):
        raise ValueError("shadow_validation_training_execution_invariant_failed")
    _require_closed_flags(manifest, "dataset_manifest")
    _require_closed_flags(report, "training_report")


def _validate_training_holdout(report: Mapping[str, Any]) -> None:
    hmm_report = report.get("hmm", {})
    hmm_validation = hmm_report.get("validation", {})
    if hmm_validation.get("passed") is not True:
        raise ValueError("shadow_validation_hmm_holdout_failed")
    if hmm_validation.get("status") != "TECHNICAL_VALIDATION_PASSED":
        raise ValueError("shadow_validation_hmm_holdout_status_invalid")
    if hmm_report.get("owner_approved") is not False:
        raise ValueError("shadow_validation_hmm_source_must_remain_unapproved")
    meta_report = report.get("meta_model", {})
    meta_validation = meta_report.get("validation", {})
    if meta_validation.get("passed") is not True:
        raise ValueError("shadow_validation_meta_holdout_failed")
    if meta_validation.get("status") != "TECHNICAL_VALIDATION_PASSED":
        raise ValueError("shadow_validation_meta_holdout_status_invalid")
    if meta_report.get("owner_approved") is not False:
        raise ValueError("shadow_validation_meta_source_must_remain_unapproved")


def _validate_hmm_training_artifact(
    hmm: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    dataset_hash: str,
    quality_fingerprint: str,
    artifact_path: Path,
) -> None:
    dataset = hmm.get("dataset", {})
    if (
        report.get("data_quality_fingerprint") != quality_fingerprint
        or dataset.get("data_quality_fingerprint") != quality_fingerprint
    ):
        raise ValueError("shadow_validation_hmm_quality_fingerprint_mismatch")
    if hmm.get("status") != "TRAINED_UNAPPROVED":
        raise ValueError("shadow_validation_hmm_source_status_invalid")
    if hmm.get("owner_approved") is not False:
        raise ValueError("shadow_validation_hmm_source_must_remain_unapproved")
    if hmm.get("model_sha256") != report.get("hmm", {}).get("model_sha256"):
        raise ValueError("shadow_validation_hmm_model_hash_mismatch")
    if dataset.get("csv_sha256") != dataset_hash:
        raise ValueError("shadow_validation_hmm_dataset_hash_mismatch")
    _require_closed_flags(hmm, "hmm_training_artifact")
    load_hmm_model_document(artifact_path.read_text(encoding="utf-8"))


def _validate_meta_training_artifact(
    model_root: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    dataset_hash: str,
) -> None:
    model = MetaLabelModel.from_dict(_read_object(model_root / "meta-label-model.json"))
    model_hash = _sha256_bytes(model.to_json().encode("utf-8"))
    if model_hash != report.get("meta_model", {}).get("model_sha256"):
        raise ValueError("shadow_validation_meta_report_hash_mismatch")
    if model_hash != manifest.get("model_sha256"):
        raise ValueError("shadow_validation_meta_manifest_hash_mismatch")
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("shadow_validation_meta_dataset_hash_mismatch")
    if manifest.get("dataset_rows") != 8760:
        raise ValueError("shadow_validation_meta_dataset_rows_mismatch")
    if manifest.get("owner_approved") is not False:
        raise ValueError("shadow_validation_meta_source_must_remain_unapproved")
    _require_closed_flags(manifest, "meta_training_manifest")


def _load_training_bundle(
    root: Path,
) -> tuple[DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_root = root / "nobitex-market-data"
    model_root = root / "alix-ai-artifacts"
    manifest = _read_object(data_root / "nobitex-btcusdt-1h.manifest.json")
    report = _read_object(model_root / "ai-training-report.json")
    hmm = _read_object(model_root / "hmm-regime-model.json")
    meta_manifest = _read_object(model_root / "meta-label-model.manifest.json")
    _validate_training_provenance(manifest, report)
    candles, quality_fingerprint = _validate_candles(
        data_root / "nobitex-btcusdt-1h.csv", manifest, expected_rows=8760
    )
    _validate_training_holdout(report)
    _validate_hmm_training_artifact(
        hmm,
        report,
        dataset_hash=DATASET_SHA256,
        quality_fingerprint=quality_fingerprint,
        artifact_path=model_root / "hmm-regime-model.json",
    )
    _validate_meta_training_artifact(
        model_root,
        report,
        meta_manifest,
        dataset_hash=DATASET_SHA256,
    )
    return candles, manifest, report, hmm, meta_manifest

def _prepare_technically_approved_models(
    root: Path,
    output_dir: Path,
    hmm: dict[str, Any],
    meta_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    approval_root = Path("config/strategy_selection")
    hmm_approval = _read_object(approval_root / "hmm-model-approval.json")
    meta_approval = _read_object(approval_root / "meta-model-approval.json")
    promoted_hmm = json.loads(json.dumps(hmm))
    promoted_hmm["status"] = APPROVAL_STATUS
    promoted_hmm["model"]["owner_approved"] = True
    promoted_hash = _canonical_sha256(promoted_hmm["model"])
    promoted_hmm["model_sha256"] = promoted_hash
    if promoted_hash != hmm_approval.get("model_sha256"):
        raise ValueError("shadow_validation_promoted_hmm_hash_mismatch")
    dataset_hash = str(promoted_hmm["dataset"]["csv_sha256"])
    _validate_approval(
        hmm_approval,
        model_type="gaussian_hmm",
        model_hash=promoted_hash,
        dataset_hash=dataset_hash,
    )
    hmm_text = (
        json.dumps(promoted_hmm, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    loaded_hmm = load_hmm_model_document(
        hmm_text, approval_document=json.dumps(hmm_approval)
    )
    meta_path = root / "alix-ai-artifacts" / "meta-label-model.json"
    meta = _read_object(meta_path)
    meta_model = MetaLabelModel.from_dict(meta)
    meta_hash = _sha256_bytes(meta_model.to_json().encode("utf-8"))
    _validate_approval(
        meta_approval,
        model_type="meta_label_logistic",
        model_hash=meta_hash,
        dataset_hash=dataset_hash,
    )
    if meta_hash != meta_manifest.get("model_sha256"):
        raise ValueError("shadow_validation_promoted_meta_hash_mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hmm-regime-model.json").write_text(hmm_text, encoding="utf-8")
    shutil.copyfile(meta_path, output_dir / "meta-label-model.json")
    shutil.copyfile(
        root / "alix-ai-artifacts" / "meta-label-model.manifest.json",
        output_dir / "meta-label-model.manifest.json",
    )
    promotion = {
        "source_training_commit_sha": TRAINING_RUN_COMMIT,
        "source_dataset_sha256": dataset_hash,
        "source_hmm_sha256": hmm["model_sha256"],
        "promoted_hmm_sha256": promoted_hash,
        "meta_model_sha256": meta_hash,
        "approval_status": APPROVAL_STATUS,
        "hmm_revision": loaded_hmm.revision,
        "technical_validation_only": True,
        "economic_acceptance": "NO-GO",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    (output_dir / "technical-promotion-manifest.json").write_text(
        json.dumps(promotion, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return promotion, meta_approval


def _validate_inference_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("provider") != "Nobitex public UDF history API":
        raise ValueError("shadow_validation_inference_provider_invalid")
    if manifest.get("timeframe") != "1h":
        raise ValueError("shadow_validation_inference_timeframe_invalid")
    if manifest.get("public_api_only") is not True:
        raise ValueError("shadow_validation_inference_not_public_only")
    if manifest.get("credentials_used") is not False:
        raise ValueError("shadow_validation_inference_credentials_detected")
    if manifest.get("exchange_calls") != 0:
        raise ValueError("shadow_validation_inference_exchange_calls_detected")
    if manifest.get("real_order_attempts") != 0:
        raise ValueError("shadow_validation_inference_real_orders_detected")
    if manifest.get("Champion") is not None:
        raise ValueError("shadow_validation_inference_champion_must_be_null")
    _require_closed_flags(manifest, "inference_manifest")


def _load_inference_bundle(
    inference_csv: Path, inference_manifest_path: Path
) -> tuple[DataFrame, dict[str, Any], str]:
    manifest = _read_object(inference_manifest_path)
    _validate_inference_manifest(manifest)
    row_count = manifest.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise ValueError("shadow_validation_inference_row_count_invalid")
    if row_count <= 0:
        raise ValueError("shadow_validation_inference_row_count_invalid")
    candles, fingerprint = _validate_candles(
        inference_csv, manifest, expected_rows=row_count
    )
    return candles, manifest, fingerprint


def _validate_model_timing(
    inference_candles: DataFrame, hmm: Mapping[str, Any]
) -> tuple[datetime, datetime, float]:
    now = datetime.now(timezone.utc)
    latest_candle = inference_candles["date"].iloc[-1].to_pydatetime()
    age_seconds = (now - latest_candle).total_seconds()
    if age_seconds < 0 or age_seconds > 7200:
        raise ValueError("shadow_validation_latest_candle_stale_or_future")
    trained_at = datetime.fromisoformat(
        str(hmm["model"]["trained_at"]).replace("Z", "+00:00")
    )
    if trained_at.tzinfo is None or trained_at.utcoffset() is None:
        raise ValueError("shadow_validation_model_training_timestamp_naive")
    if latest_candle <= trained_at:
        raise ValueError("shadow_validation_latest_candle_not_after_model_training")
    return now, latest_candle, age_seconds


def _install_promoted_runtime_models(
    training_artifact_root: Path,
    output_path: Path,
    hmm: dict[str, Any],
) -> dict[str, Any]:
    promoted_dir = output_path.parent / "promoted-models"
    meta_manifest = _read_object(
        training_artifact_root
        / "alix-ai-artifacts"
        / "meta-label-model.manifest.json"
    )
    promotion, _ = _prepare_technically_approved_models(
        training_artifact_root, promoted_dir, hmm, meta_manifest
    )
    runtime_model_dir = Path("user_data/models")
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "hmm-regime-model.json",
        "meta-label-model.json",
        "meta-label-model.manifest.json",
    ):
        shutil.copyfile(promoted_dir / name, runtime_model_dir / name)
    return promotion


def _infer_latest_candidate_state(
    inference_candles: DataFrame,
) -> tuple[Any, float, int, dict[str, Any]]:
    strategy = AliXStrategy()
    indicators = strategy.populate_indicators(
        inference_candles.copy(), {"pair": "BTC/USDT"}
    )
    latest = indicators.iloc[-1]
    if not bool(latest.get("alix_hmm_valid", False)):
        raise ValueError("shadow_validation_latest_hmm_inference_rejected")
    if not bool(latest.get("alix_meta_model_approved", False)):
        raise ValueError("shadow_validation_latest_meta_model_not_approved")
    meta_probability = float(latest["alix_meta_probability"])
    if not math.isfinite(meta_probability) or not 0 <= meta_probability <= 1:
        raise ValueError("shadow_validation_latest_meta_probability_invalid")
    entries = strategy.populate_entry_trend(indicators, {"pair": "BTC/USDT"})
    entry_count = int(entries["enter_long"].astype(int).sum())
    registry = _read_object(Path("config/strategy_selection/strategy-registry.json"))
    return latest, meta_probability, entry_count, registry


def _validate_no_economic_candidate(
    registry: Mapping[str, Any], entry_count: int
) -> None:
    if registry.get("strategies") != []:
        raise ValueError("shadow_validation_expected_empty_economic_registry")
    if entry_count != 0:
        raise ValueError("shadow_validation_entry_emitted_without_selector_evidence")


def _run_denied_shadow_probe(
    *,
    now: datetime,
    meta_probability: float,
    current_price: float,
) -> dict[str, Any]:
    denied = RiskGate().evaluate_entry(
        allocation=None,
        account=None,
        market=None,
        rules=None,
        policy=None,
        kill_switch=None,
        evaluated_at=now,
    )
    with tempfile.TemporaryDirectory(prefix="alix-shadow-validation-") as tmp:
        exchange = ShadowExchange(Path(tmp) / "shadow.sqlite")
        probe = exchange.simulate_round_trip(
            decision=denied,
            meta_confidence=Decimal(str(meta_probability)),
            quantity=Decimal("0.001"),
            entry_reference_price=Decimal(str(current_price)),
            exit_reference_price=Decimal(str(current_price)),
            fee_rate=Decimal("0.0025"),
            slippage_rate=Decimal("0.0005"),
        )
        if probe.status != "REJECTED" or probe.order_permission is not False:
            raise ValueError("shadow_validation_denied_probe_was_not_rejected")
        if (
            probe.quantity != 0
            or probe.net_pnl != 0
            or probe.source_status != "SYNTHETIC_ONLY"
        ):
            raise ValueError("shadow_validation_denied_probe_claimed_fill")
        return {
            "status": probe.status,
            "reason_codes": list(probe.rejection_reasons),
            "rows": exchange.row_count(),
        }


def run_validation(
    *,
    training_artifact_root: Path,
    inference_csv: Path,
    inference_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _, _, _, hmm, _ = _load_training_bundle(training_artifact_root)
    inference_candles, inference_manifest, inference_fingerprint = (
        _load_inference_bundle(inference_csv, inference_manifest_path)
    )
    now, latest_candle, age_seconds = _validate_model_timing(inference_candles, hmm)
    promotion = _install_promoted_runtime_models(
        training_artifact_root, output_path, hmm
    )
    latest, meta_probability, entry_count, registry = _infer_latest_candidate_state(
        inference_candles
    )
    _validate_no_economic_candidate(registry, entry_count)
    shadow_probe = _run_denied_shadow_probe(
        now=now,
        meta_probability=meta_probability,
        current_price=float(latest["close"]),
    )
    report = {
        "schema_version": "alix.ai-shadow-validation.v1",
        "training_run_commit_sha": TRAINING_RUN_COMMIT,
        "training_dataset_sha256": DATASET_SHA256,
        "hmm_model_sha256": promotion["promoted_hmm_sha256"],
        "meta_model_sha256": promotion["meta_model_sha256"],
        "approval_status": APPROVAL_STATUS,
        "inference_data_sha256": inference_manifest["csv_sha256"],
        "inference_rows": len(inference_candles),
        "inference_data_quality_fingerprint": inference_fingerprint,
        "latest_candle": latest_candle.isoformat(),
        "latest_candle_age_seconds": age_seconds,
        "latest_hmm_revision": latest.get("alix_hmm_revision"),
        "latest_hmm_regime": latest.get("alix_regime"),
        "latest_hmm_valid": bool(latest.get("alix_hmm_valid", False)),
        "latest_meta_probability": meta_probability,
        "latest_meta_model_approved": bool(
            latest.get("alix_meta_model_approved", False)
        ),
        "strategy_registry_count": len(registry.get("strategies", [])),
        "candidate_entry_count": entry_count,
        "shadow_denial_probe_status": shadow_probe["status"],
        "shadow_denial_probe_reason_codes": shadow_probe["reason_codes"],
        "shadow_denial_probe_rows": shadow_probe["rows"],
        "shadow_fill_count": 0,
        "shadow_scope": "actual-model-inference; no strategy candidate due to empty economic registry; separate synthetic risk-denial probe",
        "technical_validation": "PASS",
        "economic_acceptance": "NO-GO",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact-root", type=Path, required=True)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_validation(
        training_artifact_root=args.training_artifact_root,
        inference_csv=args.inference_csv,
        inference_manifest_path=args.inference_manifest,
        output_path=args.output,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
