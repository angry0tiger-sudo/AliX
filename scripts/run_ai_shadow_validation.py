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


def _validate_approval(
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
        or approval.get("owner") != "angry0tiger-sudo"
        or not isinstance(approval.get("approval_id"), str)
        or not approval["approval_id"].strip()
        or approval.get("model_sha256") != model_hash
        or approval.get("dataset_sha256") != dataset_hash
    ):
        raise ValueError(f"shadow_validation_approval_mismatch:{model_type}")
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
    _require_closed_flags(approval, model_type)


def _validate_candles(
    path: Path, manifest: Mapping[str, Any], expected_rows: int
) -> tuple[DataFrame, str]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != manifest.get("csv_sha256"):
        raise ValueError("shadow_validation_csv_hash_mismatch")
    candles = pd.read_csv(path)
    if tuple(candles.columns) != MODEL_FIELDS:
        raise ValueError("shadow_validation_csv_schema_invalid")
    if len(candles) != expected_rows or manifest.get("row_count") != expected_rows:
        raise ValueError("shadow_validation_candle_count_mismatch")
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="raise")
    if candles["date"].duplicated().any() or not candles["date"].is_monotonic_increasing:
        raise ValueError("shadow_validation_timestamps_not_unique_increasing")
    deltas = candles["date"].diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if not np.all(deltas == 3600.0):
        raise ValueError("shadow_validation_gap_or_timeframe_mismatch")
    for name in ("open", "high", "low", "close", "volume"):
        candles[name] = pd.to_numeric(candles[name], errors="coerce")
    values = candles.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("shadow_validation_nonfinite_ohlcv")
    opened, high, low, close, volume = (values[:, i] for i in range(5))
    if (
        np.any(np.minimum.reduce([opened, high, low, close]) <= 0)
        or np.any(volume < 0)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
        or np.any(high < low)
    ):
        raise ValueError("shadow_validation_ohlcv_invariants_failed")
    rows = [
        {str(key): value for key, value in row.items()}
        for row in candles.to_dict(orient="records")
    ]
    quality = validate_candles(rows, expected_interval_seconds=3600, gap_policy="reject")
    if not quality.passed or quality.fingerprint is None:
        raise ValueError("shadow_validation_data_quality_rejected")
    return candles, quality.fingerprint


def _load_training_bundle(root: Path) -> tuple[DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_root = root / "nobitex-market-data"
    model_root = root / "alix-ai-artifacts"
    manifest = _read_object(data_root / "nobitex-btcusdt-1h.manifest.json")
    report = _read_object(model_root / "ai-training-report.json")
    hmm = _read_object(model_root / "hmm-regime-model.json")
    meta_manifest = _read_object(model_root / "meta-label-model.manifest.json")
    if (
        report.get("commit_sha") != TRAINING_RUN_COMMIT
        or manifest.get("commit_sha") != TRAINING_RUN_COMMIT
        or report.get("dataset_rows") != 8760
        or report.get("dataset_sha256") != DATASET_SHA256
        or manifest.get("csv_sha256") != DATASET_SHA256
        or manifest.get("provider") != "Nobitex public UDF history API"
        or manifest.get("requested_days") != 365
        or manifest.get("public_api_only") is not True
        or manifest.get("credentials_used") is not False
        or manifest.get("exchange_calls") != 0
        or manifest.get("real_order_attempts") != 0
        or manifest.get("Champion") is not None
    ):
        raise ValueError("shadow_validation_training_provenance_mismatch")
    _require_closed_flags(manifest, "dataset_manifest")
    _require_closed_flags(report, "training_report")
    candles, quality_fingerprint = _validate_candles(
        data_root / "nobitex-btcusdt-1h.csv", manifest, expected_rows=8760
    )
    if (
        report.get("hmm", {}).get("validation", {}).get("passed") is not True
        or report.get("hmm", {}).get("validation", {}).get("status")
        != "TECHNICAL_VALIDATION_PASSED"
        or report.get("meta_model", {}).get("validation", {}).get("passed") is not True
        or report.get("meta_model", {}).get("validation", {}).get("status")
        != "TECHNICAL_VALIDATION_PASSED"
        or report.get("hmm", {}).get("owner_approved") is not False
        or report.get("meta_model", {}).get("owner_approved") is not False
    ):
        raise ValueError("shadow_validation_training_technical_gate_failed")
    if (
        report.get("data_quality_fingerprint") != quality_fingerprint
        or hmm.get("dataset", {}).get("data_quality_fingerprint") != quality_fingerprint
        or hmm.get("status") != "TRAINED_UNAPPROVED"
        or hmm.get("owner_approved") is not False
        or hmm.get("model_sha256") != report.get("hmm", {}).get("model_sha256")
        or hmm.get("dataset", {}).get("csv_sha256") != DATASET_SHA256
    ):
        raise ValueError("shadow_validation_hmm_source_identity_mismatch")
    _require_closed_flags(hmm, "hmm_training_artifact")
    # The source artifact must load while unapproved; approval is applied only from the
    # separate, reviewed config below.
    load_hmm_model_document((root / "alix-ai-artifacts" / "hmm-regime-model.json").read_text(encoding="utf-8"))
    meta = _read_object(model_root / "meta-label-model.json")
    meta_model = MetaLabelModel.from_dict(meta)
    meta_hash = _sha256_bytes(meta_model.to_json().encode("utf-8"))
    if (
        meta_hash != report.get("meta_model", {}).get("model_sha256")
        or meta_hash != meta_manifest.get("model_sha256")
        or meta_manifest.get("dataset_sha256") != DATASET_SHA256
        or meta_manifest.get("dataset_rows") != 8760
        or meta_manifest.get("owner_approved") is not False
    ):
        raise ValueError("shadow_validation_meta_identity_mismatch")
    _require_closed_flags(meta_manifest, "meta_training_manifest")
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
    hmm_text = json.dumps(promoted_hmm, sort_keys=True, indent=2, allow_nan=False) + "\n"
    loaded_hmm = load_hmm_model_document(hmm_text, approval_document=json.dumps(hmm_approval))
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


def run_validation(
    *,
    training_artifact_root: Path,
    inference_csv: Path,
    inference_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _, _, _, hmm, _ = _load_training_bundle(training_artifact_root)
    inference_manifest = _read_object(inference_manifest_path)
    if (
        inference_manifest.get("provider") != "Nobitex public UDF history API"
        or inference_manifest.get("timeframe") != "1h"
        or inference_manifest.get("public_api_only") is not True
        or inference_manifest.get("credentials_used") is not False
        or inference_manifest.get("exchange_calls") != 0
        or inference_manifest.get("real_order_attempts") != 0
        or inference_manifest.get("Champion") is not None
    ):
        raise ValueError("shadow_validation_inference_provenance_invalid")
    _require_closed_flags(inference_manifest, "inference_manifest")
    inference_rows = inference_manifest.get("row_count")
    if isinstance(inference_rows, bool) or not isinstance(inference_rows, int) or inference_rows <= 0:
        raise ValueError("shadow_validation_inference_row_count_invalid")
    inference_candles, inference_fingerprint = _validate_candles(
        inference_csv,
        inference_manifest,
        expected_rows=inference_rows,
    )
    now = datetime.now(timezone.utc)
    latest_candle = inference_candles["date"].iloc[-1].to_pydatetime()
    age_seconds = (now - latest_candle).total_seconds()
    if age_seconds < 0 or age_seconds > 7200:
        raise ValueError("shadow_validation_latest_candle_stale_or_future")
    trained_at = datetime.fromisoformat(
        str(hmm["model"]["trained_at"]).replace("Z", "+00:00")
    )
    if latest_candle <= trained_at:
        raise ValueError("shadow_validation_latest_candle_not_after_model_training")
    promoted_dir = output_path.parent / "promoted-models"
    promotion, _ = _prepare_technically_approved_models(
        training_artifact_root, promoted_dir, hmm, _read_object(
            training_artifact_root / "alix-ai-artifacts" / "meta-label-model.manifest.json"
        )
    )
    runtime_model_dir = Path("user_data/models")
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "hmm-regime-model.json",
        "meta-label-model.json",
        "meta-label-model.manifest.json",
    ):
        shutil.copyfile(promoted_dir / name, runtime_model_dir / name)
    frame = inference_candles.copy()
    strategy = AliXStrategy()
    indicators = strategy.populate_indicators(frame, {"pair": "BTC/USDT"})
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
    if registry.get("strategies") != []:
        raise ValueError("shadow_validation_expected_empty_economic_registry")
    if entry_count != 0:
        raise ValueError("shadow_validation_entry_emitted_without_selector_evidence")
    # Probe the simulator's rejection path using an explicitly synthetic, denied Risk Gate
    # decision. This is not a strategy proposal and must never produce a fill.
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
            entry_reference_price=Decimal(str(float(latest["close"]))),
            exit_reference_price=Decimal(str(float(latest["close"]))),
            fee_rate=Decimal("0.0025"),
            slippage_rate=Decimal("0.0005"),
        )
        if probe.status != "REJECTED" or probe.order_permission is not False:
            raise ValueError("shadow_validation_denied_probe_was_not_rejected")
        if probe.quantity != 0 or probe.net_pnl != 0 or probe.source_status != "SYNTHETIC_ONLY":
            raise ValueError("shadow_validation_denied_probe_claimed_fill")
        probe_row_count = exchange.row_count()
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
        "latest_meta_model_approved": bool(latest.get("alix_meta_model_approved", False)),
        "strategy_registry_count": len(registry.get("strategies", [])),
        "candidate_entry_count": entry_count,
        "shadow_denial_probe_status": probe.status,
        "shadow_denial_probe_reason_codes": list(probe.rejection_reasons),
        "shadow_denial_probe_rows": probe_row_count,
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
