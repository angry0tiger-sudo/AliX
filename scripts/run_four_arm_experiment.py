"""Fail-closed four-arm real pre-hack experiment runner plus synthetic smoke mode."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.ai_ensemble.four_arm_execution_lock import (
    ExecutionAuthorizationError,
    load_owner_authorized_execution_lock,
)

PREHACK_DATASET_SHA256 = (
    "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
)
PREHACK_ROW_COUNT = 31_801
PREHACK_FILENAME = "nobitex-btcusdt-1h.csv"
MODEL_PIN_PATH = Path("config/four_arm/model_pins.json")
CONTRACT_PATHS = {
    "kronos_adapter_sha256": Path(
        "docs/research/preregistration/contracts/kronos_adapter.md"
    ),
    "chronos_adapter_sha256": Path(
        "docs/research/preregistration/contracts/chronos_adapter.md"
    ),
    "combined_rule_sha256": Path(
        "docs/research/preregistration/contracts/combined_rule.md"
    ),
    "fold_plan_sha256": Path("docs/research/preregistration/contracts/fold_plan.md"),
    "triple_barrier_sha256": Path(
        "docs/research/preregistration/contracts/triple_barrier.md"
    ),
    "cost_fill_simulator_sha256": Path(
        "docs/research/preregistration/contracts/cost_fill_simulator.md"
    ),
    "dsr_method_sha256": Path("docs/research/preregistration/contracts/dsr_method.md"),
    "license_and_corpus_disclosure_sha256": Path(
        "docs/research/preregistration/contracts/model_provenance.md"
    ),
    "runtime_lock_sha256": Path(
        "docs/research/preregistration/contracts/runtime_lock.md"
    ),
}
TRIALS = {
    "A": "FOUR-ARM-MOCK-A",
    "B": "FOUR-ARM-MOCK-B",
    "C": "FOUR-ARM-MOCK-C",
    "D": "FOUR-ARM-MOCK-D",
}
THRESHOLDS_BPS = (120, 180, 240)
TRIAL_ID_MATRIX = {
    threshold: {
        arm_id: f"NOBITEX-KC26-FD-{threshold}-{arm_id}"
        for arm_id in ("A", "B", "C", "D")
    }
    for threshold in THRESHOLDS_BPS
}
CONFIRMATORY_ROW_COUNT = 27_825
EXCLUDED_TAIL_ROWS = PREHACK_ROW_COUNT - CONFIRMATORY_ROW_COUNT
HISTORICAL_TRIAL_LOWER_BOUND = 7_912


DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
MAX_HOLDING_BARS = 24
PURGE_EMBARGO_BARS = 24
MODEL_CONTEXT_BARS = 512
FORECAST_REFRESH_BARS = 24
COST_SCENARIOS_BPS = (50, 90)
SEED = 42


class FourArmExecutionError(RuntimeError):
    """Raised for invalid four-arm data, model or contract evidence."""


@dataclass(frozen=True, slots=True)
class Fold:
    fold_id: int
    prefix_end: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class Trade:
    entry_index: int
    exit_index: int
    entry_timestamp: str
    exit_timestamp: str
    gross_return: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class Forecast:
    kronos_return: float
    chronos_return: float


@dataclass(frozen=True, slots=True)
class ForecastCache:
    """Immutable full-prefix forecast values shared by all folds and trials."""

    anchors: tuple[int, ...]
    forecasts: tuple[Forecast, ...]
    identity_sha256: str
    cache_sha256: str
    entries_json: str



def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_walk_forward_folds(row_count: int) -> tuple[Fold, ...]:
    """Build three expanding-prefix 70/15/15 chronological folds."""
    if isinstance(row_count, bool) or row_count < 1_000:
        raise ValueError("walk_forward_row_count_invalid")
    prefix_ends = (
        math.floor(0.70 * row_count),
        math.floor(0.85 * row_count),
        row_count,
    )
    folds: list[Fold] = []
    previous_test_end = -1
    for fold_number, prefix_end in enumerate(prefix_ends, start=1):
        train_boundary = math.floor(0.70 * prefix_end)
        validation_boundary = math.floor(0.85 * prefix_end)
        fold = Fold(
            fold_id=fold_number,
            prefix_end=prefix_end,
            train_start=0,
            train_end=train_boundary - PURGE_EMBARGO_BARS,
            validation_start=train_boundary + PURGE_EMBARGO_BARS,
            validation_end=validation_boundary - PURGE_EMBARGO_BARS,
            test_start=validation_boundary + PURGE_EMBARGO_BARS,
            test_end=prefix_end - MAX_HOLDING_BARS,
        )
        if not (
            0
            <= fold.train_start
            < fold.train_end
            < fold.validation_start
            < fold.validation_end
            < fold.test_start
            < fold.test_end
            <= row_count
        ):
            raise FourArmExecutionError("walk_forward_fold_boundaries_invalid")
        if fold.test_start <= previous_test_end:
            raise FourArmExecutionError("walk_forward_test_windows_not_chronological")
        previous_test_end = fold.test_end
        folds.append(fold)
    return tuple(folds)


def _validate_lock_contract_fingerprints(
    lock: Mapping[str, Any],
    repository_root: Path,
) -> None:
    contracts = lock.get("contracts")
    if not isinstance(contracts, Mapping):
        raise FourArmExecutionError("execution_lock_contracts_missing")
    for key, relative_path in CONTRACT_PATHS.items():
        expected = contracts.get(key)
        path = repository_root / relative_path
        if not isinstance(expected, str) or len(expected) != 64:
            raise FourArmExecutionError(f"execution_lock_contract_hash_missing:{key}")
        if not path.is_file():
            raise FourArmExecutionError(f"execution_contract_file_missing:{key}")
        if _sha256_file(path) != expected:
            raise FourArmExecutionError(f"execution_contract_hash_mismatch:{key}")


def _read_json(path: Path, error_code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise FourArmExecutionError(error_code) from None


def _require_fields(
    value: Any,
    fields: Sequence[str],
    error_code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FourArmExecutionError(error_code)
    if any(field not in value for field in fields):
        raise FourArmExecutionError(error_code)
    return value


def _assert_pin_lock_match(pin_key: str, pin: Any, locked: Any) -> None:
    if not isinstance(pin, Mapping):
        raise FourArmExecutionError(f"model_pin_missing:{pin_key}")
    if not isinstance(locked, Mapping):
        raise FourArmExecutionError(f"model_lock_missing:{pin_key}")
    for field in ("repository", "revision", "weights_sha256"):
        if pin.get(field) != locked.get(field):
            raise FourArmExecutionError(f"model_pin_lock_mismatch:{pin_key}:{field}")


def _coerce_pins(raw_pins: Any, lock: Mapping[str, Any]) -> dict[str, Any]:
    pins = _require_fields(
        raw_pins,
        ("schema_version", "models"),
        "model_pin_manifest_schema_invalid",
    )
    if pins.get("schema_version") != "alix.four-arm.model-pins.v1":
        raise FourArmExecutionError("model_pin_manifest_schema_invalid")
    models = pins.get("models")
    locked_models = lock.get("models")
    if not isinstance(models, Mapping) or not isinstance(locked_models, Mapping):
        raise FourArmExecutionError("model_pin_manifest_models_invalid")
    kronos_pin = models.get("kronos_small")
    if not isinstance(kronos_pin, Mapping):
        raise FourArmExecutionError("model_pin_missing:kronos_small")
    tokenizer_pin = kronos_pin.get("tokenizer")
    kronos_lock = locked_models.get("kronos_small")
    locked_tokenizer = (
        kronos_lock.get("tokenizer") if isinstance(kronos_lock, Mapping) else None
    )
    _assert_pin_lock_match("kronos_small", kronos_pin, kronos_lock)
    _assert_pin_lock_match("kronos_tokenizer", tokenizer_pin, locked_tokenizer)
    _assert_pin_lock_match(
        "chronos_2", models.get("chronos_2"), locked_models.get("chronos_2")
    )
    return dict(pins)


def _load_model_pin_manifest(
    path: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    return _coerce_pins(
        _read_json(path, "model_pin_manifest_unreadable"),
        lock,
    )


def _read_csv_bytes(csv_path: Path) -> bytes:
    if csv_path.name != PREHACK_FILENAME:
        raise FourArmExecutionError("only_registered_prehack_csv_is_allowed")
    if not csv_path.is_file():
        raise FourArmExecutionError("prehack_artifact_file_missing")
    try:
        return csv_path.read_bytes()
    except OSError:
        raise FourArmExecutionError("prehack_artifact_file_missing") from None


def _assert_sha256(
    csv_bytes: bytes,
    expected_sha256: str = PREHACK_DATASET_SHA256,
) -> str:
    csv_sha = sha256(csv_bytes).hexdigest()
    if csv_sha != expected_sha256:
        raise FourArmExecutionError("prehack_csv_sha256_mismatch")
    return csv_sha


def _validate_data_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FourArmExecutionError("prehack_artifact_file_missing")
    manifest = _read_json(manifest_path, "prehack_manifest_unreadable")
    _require_fields(
        manifest,
        (
            "csv_sha256",
            "dataset_window",
            "row_count",
            "private_api_calls",
            "real_order_attempts",
        ),
        "prehack_manifest_schema_invalid",
    )
    if manifest.get("csv_sha256") != PREHACK_DATASET_SHA256:
        raise FourArmExecutionError("prehack_manifest_sha256_mismatch")
    if manifest.get("dataset_window") != "PRE_HACK_NOBITEX_ONLY":
        raise FourArmExecutionError("prehack_manifest_window_mismatch")
    if manifest.get("row_count") != PREHACK_ROW_COUNT:
        raise FourArmExecutionError("prehack_manifest_row_count_mismatch")
    if (
        manifest.get("private_api_calls") != 0
        or manifest.get("real_order_attempts") != 0
    ):
        raise FourArmExecutionError("prehack_manifest_safety_invariant_violation")


def _parse_rows(csv_bytes: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(csv_bytes))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise FourArmExecutionError(
            f"prehack_csv_parse_failed:{type(exc).__name__}"
        ) from exc
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise FourArmExecutionError("prehack_csv_schema_invalid")
    if len(frame) != PREHACK_ROW_COUNT:
        raise FourArmExecutionError("prehack_csv_row_count_mismatch")
    frame = frame.rename(columns={"date": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _validate_hourly_grid(frame: pd.DataFrame) -> None:
    if frame["timestamp"].is_monotonic_increasing is not True:
        raise FourArmExecutionError("prehack_timestamps_not_strictly_increasing")
    seconds = frame["timestamp"].astype("int64") // 1_000_000_000
    differences = seconds.diff().dropna()
    if not bool((differences == 3600).all()):
        raise FourArmExecutionError("prehack_hourly_grid_invalid")
    if not bool((seconds % 3600 == 1800).all()):
        raise FourArmExecutionError("prehack_utc_offset_grid_invalid")


def _validate_ohlcv(frame: pd.DataFrame) -> None:
    numeric_values = frame[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric_values).all():
        raise FourArmExecutionError("prehack_non_finite_ohlcv")
    if (numeric_values[:, :4] <= 0).any() or (numeric_values[:, 4] < 0).any():
        raise FourArmExecutionError("prehack_ohlcv_values_invalid")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise FourArmExecutionError("prehack_high_geometry_invalid")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise FourArmExecutionError("prehack_low_geometry_invalid")


def _load_verified_prehack_data(
    csv_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, str]:
    csv_bytes = _read_csv_bytes(csv_path)
    csv_sha = _assert_sha256(csv_bytes)
    _validate_data_manifest(manifest_path)
    frame = _parse_rows(csv_bytes)
    _validate_hourly_grid(frame)
    _validate_ohlcv(frame)
    return frame.reset_index(drop=True), csv_sha


def _calculate_atr(frame: pd.DataFrame) -> np.ndarray:
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    previous_close = np.concatenate(([np.nan], close[:-1]))
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)),
    )
    return (
        pd.Series(true_range)
        .ewm(alpha=1.0 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _baseline_signals(frame: pd.DataFrame, atr: np.ndarray) -> np.ndarray:
    prior_high = (
        frame["high"]
        .rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD)
        .max()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )
    close = frame["close"].to_numpy(dtype=np.float64)
    return (
        np.isfinite(prior_high) & np.isfinite(atr) & (close > prior_high) & (atr > 0.0)
    )


def _model_pin_groups(
    pins: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    models = pins.get("models")
    if not isinstance(models, Mapping):
        raise FourArmExecutionError("model_pin_manifest_models_invalid")
    kronos = models.get("kronos_small")
    chronos = models.get("chronos_2")
    if not isinstance(kronos, Mapping) or not isinstance(chronos, Mapping):
        raise FourArmExecutionError("model_pin_manifest_models_invalid")
    tokenizer = kronos.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise FourArmExecutionError("model_pin_missing:kronos_tokenizer")
    return {
        "kronos_small": kronos,
        "kronos_tokenizer": tokenizer,
        "chronos_2": chronos,
    }


def _artifact_record_values(
    record: Any,
) -> tuple[str, str, int]:
    if not isinstance(record, Mapping):
        raise FourArmExecutionError("model_artifact_record_invalid")
    relative = record.get("path")
    expected_sha = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(relative, str) or not isinstance(expected_sha, str):
        raise FourArmExecutionError("model_artifact_record_invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise FourArmExecutionError("model_artifact_size_invalid")
    return relative, expected_sha, expected_size


def _resolve_artifact_path(
    artifact_dir: Path,
    root: Path,
    relative: str,
) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise FourArmExecutionError("model_artifact_path_invalid")
    path = artifact_dir / relative_path
    resolved = path.resolve()
    if root not in resolved.parents or path.is_symlink() or not path.is_file():
        raise FourArmExecutionError("model_artifact_file_missing")
    return path


def _verify_artifact_record(
    artifact_dir: Path,
    root: Path,
    record: Any,
) -> tuple[str, Mapping[str, Any]]:
    relative, expected_sha, expected_size = _artifact_record_values(record)
    path = _resolve_artifact_path(artifact_dir, root, relative)
    if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
        raise FourArmExecutionError("model_artifact_sha256_mismatch")
    if not isinstance(record, Mapping):
        raise FourArmExecutionError("model_artifact_record_invalid")
    return relative, record


def _verify_artifact_files(
    artifact_dir: Path,
    records: Any,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list):
        raise FourArmExecutionError("model_artifact_files_invalid")
    verified: dict[str, Mapping[str, Any]] = {}
    root = artifact_dir.resolve()
    for record in records:
        relative, validated = _verify_artifact_record(artifact_dir, root, record)
        if relative in verified:
            raise FourArmExecutionError("model_artifact_duplicate_path")
        verified[relative] = validated
    return verified


def _load_artifact_manifest(
    pins: Mapping[str, Any],
    pin_manifest_path: Path,
    artifact_dir: Path,
) -> Mapping[str, Any]:
    manifest = _read_json(
        artifact_dir / "manifest.json",
        "model_artifact_manifest_unreadable",
    )
    if not isinstance(manifest, Mapping):
        raise FourArmExecutionError("model_artifact_manifest_invalid")
    if manifest.get("schema_version") != "alix.four-arm.model-artifacts.v1":
        raise FourArmExecutionError("model_artifact_manifest_schema_invalid")
    if manifest.get("pins_manifest_sha256") != _sha256_file(pin_manifest_path):
        raise FourArmExecutionError("model_artifact_pin_fingerprint_mismatch")
    return manifest


def _require_verified_weight_record(
    group: str,
    pin: Mapping[str, Any],
    verified: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, Any, Mapping[str, Any]]:
    weight_file = pin.get("weight_file")
    expected_sha = pin.get("weights_sha256")
    expected_size = pin.get("weight_size_bytes")
    if not isinstance(weight_file, str) or not isinstance(expected_sha, str):
        raise FourArmExecutionError(f"model_pin_fields_invalid:{group}")
    relative = f"{group}/{weight_file}"
    record = verified.get(relative)
    if not isinstance(record, Mapping):
        raise FourArmExecutionError(f"model_artifact_weight_missing:{group}")
    return relative, expected_sha, expected_size, record


def _assert_model_artifact_identity(
    group: str,
    pin: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_sha: str,
    expected_size: Any,
) -> None:
    if record.get("sha256") != expected_sha:
        raise FourArmExecutionError(f"model_artifact_weight_sha256_mismatch:{group}")
    if record.get("repository") != pin.get("repository"):
        raise FourArmExecutionError(f"model_artifact_repository_mismatch:{group}")
    if record.get("revision") != pin.get("revision"):
        raise FourArmExecutionError(f"model_artifact_revision_mismatch:{group}")
    if expected_size is not None and record.get("size_bytes") != expected_size:
        raise FourArmExecutionError(f"model_artifact_weight_size_mismatch:{group}")


def _verify_model_weight(
    group: str,
    pin: Mapping[str, Any],
    verified: Mapping[str, Mapping[str, Any]],
    artifact_dir: Path,
) -> Path:
    _, expected_sha, expected_size, record = _require_verified_weight_record(
        group,
        pin,
        verified,
    )
    _assert_model_artifact_identity(
        group,
        pin,
        record,
        expected_sha,
        expected_size,
    )
    return artifact_dir / group


def _verify_model_artifacts(
    pins: Mapping[str, Any],
    pin_manifest_path: Path,
    artifact_dir: Path,
) -> dict[str, Path]:
    manifest = _load_artifact_manifest(pins, pin_manifest_path, artifact_dir)
    verified = _verify_artifact_files(artifact_dir, manifest.get("files"))
    return {
        group: _verify_model_weight(group, pin, verified, artifact_dir)
        for group, pin in _model_pin_groups(pins).items()
    }


def _load_cpu_models(
    pins: Mapping[str, Any],
    pin_manifest_path: Path,
    kronos_code_dir: Path,
    artifact_dir: Path,
) -> tuple[Any, Any]:
    model_dirs = _verify_model_artifacts(pins, pin_manifest_path, artifact_dir)
    torch = importlib.import_module("torch")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    kronos_code = str(kronos_code_dir.resolve())
    if not (kronos_code_dir / "model" / "__init__.py").is_file():
        raise FourArmExecutionError("pinned_kronos_source_tree_missing")
    if kronos_code not in sys.path:
        sys.path.insert(0, kronos_code)
    kronos_module = importlib.import_module("model")
    chronos_module = importlib.import_module("chronos")
    kronos_class = getattr(kronos_module, "Kronos")
    predictor_class = getattr(kronos_module, "KronosPredictor")
    tokenizer_class = getattr(kronos_module, "KronosTokenizer")
    chronos_class = getattr(chronos_module, "Chronos2Pipeline")
    tokenizer = tokenizer_class.from_pretrained(str(model_dirs["kronos_tokenizer"]))
    kronos_model = kronos_class.from_pretrained(str(model_dirs["kronos_small"]))
    kronos_model.eval()
    tokenizer.eval()
    kronos_predictor = predictor_class(
        kronos_model,
        tokenizer,
        device="cpu",
        max_context=MODEL_CONTEXT_BARS,
    )
    chronos_pipeline = chronos_class.from_pretrained(
        str(model_dirs["chronos_2"]),
        device_map="cpu",
    )
    chronos_pipeline.model.eval()
    return kronos_predictor, chronos_pipeline


def _forecast_at(
    frame: pd.DataFrame,
    decision_index: int,
    kronos_predictor: Any,
    chronos_pipeline: Any,
) -> Forecast:
    start = decision_index - MODEL_CONTEXT_BARS + 1
    if start < 0 or decision_index + MAX_HOLDING_BARS >= len(frame):
        raise FourArmExecutionError("model_forecast_context_or_horizon_unavailable")
    context = frame.iloc[start : decision_index + 1]
    context_df = (
        context[["open", "high", "low", "close", "volume"]]
        .copy()
        .reset_index(drop=True)
    )
    x_timestamps = context["timestamp"].reset_index(drop=True)
    y_timestamps = (
        frame["timestamp"]
        .iloc[decision_index + 1 : decision_index + 1 + MAX_HOLDING_BARS]
        .reset_index(drop=True)
    )
    if len(context_df) != MODEL_CONTEXT_BARS or len(y_timestamps) != MAX_HOLDING_BARS:
        raise FourArmExecutionError("model_forecast_input_length_invalid")
    current_close = float(frame["close"].iloc[decision_index])

    kronos_frame = kronos_predictor.predict(
        df=context_df,
        x_timestamp=x_timestamps,
        y_timestamp=y_timestamps,
        pred_len=MAX_HOLDING_BARS,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    kronos_close = float(kronos_frame["close"].iloc[-1])
    kronos_return = kronos_close / current_close - 1.0

    torch = importlib.import_module("torch")

    close_context = (
        frame["close"].iloc[start : decision_index + 1].to_numpy(dtype=np.float32)
    )
    chronos_forecasts = chronos_pipeline.predict(
        [torch.tensor(close_context, dtype=torch.float32)],
        prediction_length=MAX_HOLDING_BARS,
        context_length=MODEL_CONTEXT_BARS,
        batch_size=1,
        cross_learning=False,
    )
    quantiles = list(chronos_pipeline.quantiles)
    median_index = quantiles.index(0.5)
    chronos_close = float(
        chronos_forecasts[0][0, median_index, -1].detach().cpu().item()
    )
    chronos_return = chronos_close / current_close - 1.0
    if not math.isfinite(kronos_return) or not math.isfinite(chronos_return):
        raise FourArmExecutionError("non_finite_model_forecast")
    return Forecast(kronos_return=kronos_return, chronos_return=chronos_return)


def _build_forecast_cache(
    frame: pd.DataFrame,
    models: tuple[Any, Any],
    pins: Mapping[str, Any],
    lock: Mapping[str, Any],
    dataset_sha256: str,
) -> ForecastCache:
    """Infer each pinned model once per anchor over the eligible prefix."""
    model_groups = _model_pin_groups(pins)
    model_identity = {
        group: {
            "repository": record["repository"],
            "revision": record["revision"],
            "weights_sha256": record["weights_sha256"],
        }
        for group, record in model_groups.items()
    }
    identity = {
        "schema_version": "alix.four-arm.forecast-cache-identity.v1",
        "dataset_sha256": dataset_sha256,
        "confirmatory_rows": len(frame),
        "models": model_identity,
        "forecast_horizon_bars": MAX_HOLDING_BARS,
        "context_bars": MODEL_CONTEXT_BARS,
        "refresh_bars": FORECAST_REFRESH_BARS,
        "adapter_contract_fingerprints": lock["contracts"],
        "deterministic_seed": SEED,
        "runtime": {
            "device": "cpu",
            "torch_threads": 2,
            "deterministic_algorithms": True,
            "fine_tuning": False,
        },
    }
    identity_bytes = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    identity_sha256 = sha256(identity_bytes).hexdigest()

    anchors: list[int] = []
    forecasts: list[Forecast] = []
    for anchor in range(
        MODEL_CONTEXT_BARS - 1,
        len(frame) - MAX_HOLDING_BARS,
        FORECAST_REFRESH_BARS,
    ):
        forecast = _forecast_at(frame, anchor, models[0], models[1])
        if not (
            math.isfinite(forecast.kronos_return)
            and math.isfinite(forecast.chronos_return)
        ):
            raise FourArmExecutionError("forecast_cache_non_finite_value")
        anchors.append(anchor)
        forecasts.append(forecast)

    if not anchors or len(anchors) != len(set(anchors)):
        raise FourArmExecutionError("forecast_cache_anchor_inventory_invalid")

    entries = [
        {
            "anchor_index": anchor,
            "kronos_return": forecast.kronos_return,
            "chronos_return": forecast.chronos_return,
        }
        for anchor, forecast in zip(anchors, forecasts, strict=True)
    ]
    entries_json = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    cache_sha256 = sha256(entries_json.encode("utf-8")).hexdigest()
    return ForecastCache(
        anchors=tuple(anchors),
        forecasts=tuple(forecasts),
        identity_sha256=identity_sha256,
        cache_sha256=cache_sha256,
        entries_json=entries_json,
    )


def _signals_from_forecast_cache(
    cache: ForecastCache,
    row_count: int,
    arm_id: str,
    threshold_bps: int,
) -> np.ndarray:
    """Build an independent forecast-entry stream; never intersect Donchian."""
    if threshold_bps not in THRESHOLDS_BPS or arm_id not in {"B", "C", "D"}:
        raise FourArmExecutionError("forecast_signal_cell_invalid")
    threshold = threshold_bps / 10_000.0
    signals = np.zeros(row_count, dtype=bool)
    for anchor, forecast in zip(cache.anchors, cache.forecasts, strict=True):
        if arm_id == "B":
            signal = forecast.kronos_return > threshold
        elif arm_id == "C":
            signal = forecast.chronos_return > threshold
        else:
            combined_return = (
                forecast.kronos_return + forecast.chronos_return
            ) / 2.0
            signal = combined_return > threshold
        stop = min(anchor + FORECAST_REFRESH_BARS, row_count)
        signals[anchor:stop] = signal
    return signals


def _entry_signals(
    signals: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    return np.asarray(signals, dtype=bool) & np.isfinite(atr)


def _apply_barriers(
    frame: pd.DataFrame,
    fold: Fold,
    atr: np.ndarray,
    signal_index: int,
) -> Trade:
    open_values = frame["open"].to_numpy(dtype=np.float64)
    high_values = frame["high"].to_numpy(dtype=np.float64)
    low_values = frame["low"].to_numpy(dtype=np.float64)
    close_values = frame["close"].to_numpy(dtype=np.float64)
    timestamps = frame["timestamp"]
    entry_index = signal_index + 1
    entry_price = float(open_values[entry_index])
    initial_risk = ATR_MULTIPLIER * float(atr[signal_index])
    stop_price = entry_price - initial_risk
    target_price = entry_price + initial_risk
    high_water = entry_price
    last_bar = min(entry_index + MAX_HOLDING_BARS - 1, fold.test_end - 1)
    exit_index = last_bar
    exit_price = float(close_values[last_bar])
    exit_reason = "maximum_holding_time"
    for bar_index in range(entry_index, last_bar + 1):
        bar_open = float(open_values[bar_index])
        bar_high = float(high_values[bar_index])
        bar_low = float(low_values[bar_index])
        bar_close = float(close_values[bar_index])
        if bar_open <= stop_price:
            exit_index, exit_price, exit_reason = bar_index, bar_open, "gap_stop"
            break
        if bar_open >= target_price:
            exit_index, exit_price, exit_reason = bar_index, bar_open, "gap_target"
            break
        stop_touched = bar_low <= stop_price
        target_touched = bar_high >= target_price
        if stop_touched:
            exit_index = bar_index
            exit_price = stop_price
            exit_reason = "dual_touch_stop_first" if target_touched else "stop_loss"
            break
        if target_touched:
            exit_index, exit_price, exit_reason = bar_index, target_price, "take_profit"
            break
        if bar_index == last_bar:
            exit_index, exit_price, exit_reason = (
                bar_index,
                bar_close,
                "maximum_holding_time",
            )
            break
        high_water = max(high_water, bar_high)
        current_atr = float(atr[bar_index])
        if math.isfinite(current_atr) and current_atr > 0.0:
            stop_price = max(stop_price, high_water - ATR_MULTIPLIER * current_atr)
    return Trade(
        entry_index=entry_index,
        exit_index=exit_index,
        entry_timestamp=timestamps.iloc[entry_index].isoformat(),
        exit_timestamp=timestamps.iloc[exit_index].isoformat(),
        gross_return=exit_price / entry_price - 1.0,
        exit_reason=exit_reason,
    )


def _aggregate_trades(trades: Sequence[Trade]) -> list[Trade]:
    return sorted(trades, key=lambda trade: (trade.exit_index, trade.entry_index))


def _simulate_fold_trades(
    frame: pd.DataFrame,
    fold: Fold,
    signals: np.ndarray,
    atr: np.ndarray,
) -> list[Trade]:
    eligible = _entry_signals(signals, atr)
    trades: list[Trade] = []
    cursor = fold.test_start
    while cursor < fold.test_end - 1:
        if not bool(eligible[cursor]):
            cursor += 1
            continue
        trade = _apply_barriers(frame, fold, atr, cursor)
        trades.append(trade)
        cursor = trade.exit_index + 1
    return _aggregate_trades(trades)


def _apply_costs(gross_return: float, cost_bps: int) -> float:
    return gross_return - cost_bps / 10_000.0


def _calculate_metrics(trades: Sequence[Trade], cost_bps: int) -> dict[str, Any]:
    net_returns = [_apply_costs(trade.gross_return, cost_bps) for trade in trades]
    profits = sum(value for value in net_returns if value > 0.0)
    losses = abs(sum(value for value in net_returns if value < 0.0))
    profit_factor = profits / losses if losses > 0.0 else None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in net_returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    total_return = equity - 1.0
    return {
        "net_profit_factor": profit_factor,
        "trades": len(trades),
        "max_drawdown": max_drawdown,
        "net_total_return": total_return,
    }


def _fold_stability_pass(fold_metrics: Mapping[str, Any]) -> bool:
    for cost in COST_SCENARIOS_BPS:
        metric = fold_metrics.get(str(cost))
        if not isinstance(metric, Mapping):
            return False
        net_pf = metric.get("net_profit_factor")
        if not isinstance(net_pf, (int, float)) or net_pf < 1.10:
            return False
    drawdown = fold_metrics.get("max_drawdown")
    return isinstance(drawdown, (int, float)) and drawdown <= 0.15


def _fold_summary(
    fold_trades: Mapping[int, Sequence[Trade]],
    folds: Sequence[Fold],
) -> tuple[list[dict[str, Any]], list[bool]]:
    fold_results: list[dict[str, Any]] = []
    fold_passes: list[bool] = []
    for fold in folds:
        trades = fold_trades.get(fold.fold_id, [])
        metrics = {
            str(cost): _calculate_metrics(trades, cost) for cost in COST_SCENARIOS_BPS
        }
        fold_drawdown = max(
            float(metrics[str(cost)]["max_drawdown"]) for cost in COST_SCENARIOS_BPS
        )
        combined = {"max_drawdown": fold_drawdown, **metrics}
        passed = _fold_stability_pass(combined)
        fold_passes.append(passed)
        fold_results.append(
            {
                "fold_id": fold.fold_id,
                "ranges": asdict(fold),
                "metrics": combined,
                "fold_stability_pass": passed,
            }
        )
    return fold_results, fold_passes


def _gate_summary(
    aggregate: Mapping[str, Any],
    aggregate_trades: int,
    fold_passes: Sequence[bool],
) -> tuple[float, bool]:
    aggregate_drawdown = max(
        float(aggregate[str(cost)]["max_drawdown"]) for cost in COST_SCENARIOS_BPS
    )
    economics_pass = (
        aggregate_trades >= 100
        and all(
            isinstance(aggregate[str(cost)]["net_profit_factor"], (int, float))
            and aggregate[str(cost)]["net_profit_factor"] >= 1.10
            for cost in COST_SCENARIOS_BPS
        )
        and aggregate_drawdown <= 0.15
        and sum(fold_passes) >= 3
    )
    return aggregate_drawdown, economics_pass


def _arm_manifest(
    arm_id: str,
    trial_id: str,
    threshold_bps: int,
    fold_trades: Mapping[int, Sequence[Trade]],
    dataset_sha256: str,
    folds: Sequence[Fold],
    forecast_cache: ForecastCache,
) -> dict[str, Any]:
    all_trades = _aggregate_trades(
        [trade for trades in fold_trades.values() for trade in trades]
    )
    aggregate = {
        str(cost): _calculate_metrics(all_trades, cost) for cost in COST_SCENARIOS_BPS
    }
    fold_results, fold_passes = _fold_summary(fold_trades, folds)
    aggregate_trades = len(all_trades)
    aggregate_drawdown, economics_pass = _gate_summary(
        aggregate,
        aggregate_trades,
        fold_passes,
    )
    sample_status = (
        "INSUFFICIENT_SAMPLE" if aggregate_trades < 100 else "SAMPLE_GATE_MET"
    )
    decision = (
        "INSUFFICIENT_SAMPLE"
        if sample_status == "INSUFFICIENT_SAMPLE"
        else (
            "NO_GO_INSUFFICIENT_DSR_EVIDENCE"
            if economics_pass
            else "NO_GO_ECONOMIC_GATES"
        )
    )
    return {
        "schema_version": "alix.four-arm-arm-manifest.v2",
        "arm_id": arm_id,
        "trial_id": trial_id,
        "threshold_bps": threshold_bps,
        "threshold_applies_to_arm": arm_id != "A",
        "arm_a_control_is_threshold_invariant": arm_id == "A",
        "dataset_sha256": dataset_sha256,
        "dataset_rows": CONFIRMATORY_ROW_COUNT,
        "source_dataset_rows": PREHACK_ROW_COUNT,
        "excluded_tail_rows": EXCLUDED_TAIL_ROWS,
        "cost_scenarios_bps": list(COST_SCENARIOS_BPS),
        "metrics": {
            "50": aggregate["50"],
            "90": aggregate["90"],
            "trades": aggregate_trades,
            "max_drawdown": aggregate_drawdown,
        },
        "folds": fold_results,
        "folds_passed": sum(fold_passes),
        "sample_status": sample_status,
        "dsr": None,
        "dsr_status": "INSUFFICIENT_EVIDENCE_GLOBAL_N_UNKNOWN",
        "dsr_sensitivity": {
            "global_n": "UNKNOWN",
            "audited_historical_lower_bound": HISTORICAL_TRIAL_LOWER_BOUND,
            "multipliers": [1, 2, 5, 10],
            "numeric_dsr_computed": False,
            "reason": "GLOBAL_N_UNKNOWN; no defensible candidate-level reconciliation",
        },
        "dsr_registered_cell_count": 12,
        "dsr_repeated_control_disclosure": (
            "Arm A is the identical Donchian control in all three threshold cells; "
            "all three preregistered cell IDs remain in the registered candidate count."
        ),
        "economic_gates_except_dsr_pass": economics_pass,
        "pass": False,
        "winner": False,
        "decision": decision,
        "forecast_cache_identity_sha256": forecast_cache.identity_sha256,
        "forecast_cache_sha256": forecast_cache.cache_sha256,
        "forecast_anchors": len(forecast_cache.anchors),
        "model_inference_calls_per_model": len(forecast_cache.anchors),
        "model_inference_calls_total": 2 * len(forecast_cache.anchors),
        "forecast_cache_computed_for_this_trial": False,
        "trade_exit_reasons": {
            reason: sum(1 for trade in all_trades if trade.exit_reason == reason)
            for reason in sorted({trade.exit_reason for trade in all_trades})
        },
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "posthack_holdout_access": False,
        },
    }




def run_synthetic_mode(output_dir: Path) -> dict[str, Any]:
    """Write structural four-arm manifests from a deterministic synthetic fixture."""
    generator = np.random.default_rng(SEED)
    steps = 1_200
    close = 100.0 * np.exp(np.cumsum(generator.normal(0.0, 0.001, steps)))
    high = close * (1.0 + generator.uniform(0.0, 0.001, steps))
    low = close * (1.0 - generator.uniform(0.0, 0.001, steps))
    open_values = close.copy()
    volume = generator.uniform(1.0, 100.0, steps)
    fixture = np.column_stack((open_values, high, low, close, volume)).astype(
        np.float64
    )
    fixture_sha = sha256(fixture.tobytes()).hexdigest()
    folds = build_walk_forward_folds(steps)
    manifests = []
    for arm_id, trial_id in TRIALS.items():
        manifest = {
            "schema_version": "alix.four-arm-arm-manifest.v1",
            "arm_id": arm_id,
            "trial_id": trial_id,
            "mode": "MOCK_ONLY",
            "synthetic_fixture_rows": steps,
            "synthetic_fixture_sha256": fixture_sha,
            "folds": [asdict(fold) for fold in folds],
            "economic_evaluation_performed": False,
            "metrics": None,
            "dsr": None,
            "winner": False,
            "pass": False,
            "safety": {
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "private_api_calls": 0,
                "real_order_attempts": 0,
            },
        }
        _write_json(output_dir / f"arm-{arm_id}.json", manifest)
        manifests.append(manifest)
    report = {
        "schema_version": "alix.four-arm-run-report.v1",
        "mode": "MOCK_ONLY",
        "synthetic_fixture_rows": steps,
        "synthetic_fixture_sha256": fixture_sha,
        "arm_count": len(manifests),
        "economic_evaluation_performed": False,
        "result": "MOCK_VALIDATION_ONLY_NOT_AN_ECONOMIC_RESULT",
        "arms": [item["arm_id"] for item in manifests],
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }
    _write_json(output_dir / "four-arm-run-report.json", report)
    return report


def run_real_experiment(
    *,
    csv_path: Path,
    data_manifest_path: Path,
    lock_path: Path,
    model_pin_path: Path,
    kronos_code_dir: Path,
    output_dir: Path,
    repository_root: Path,
    model_artifacts_dir: Path = Path("artifacts/four-arm-models"),
    validate_lock_only: bool = False,
) -> dict[str, Any] | None:
    """Run the locked ADR-038 12-cell research design on the pre-hack prefix."""
    lock = load_owner_authorized_execution_lock(lock_path)
    _validate_lock_contract_fingerprints(lock, repository_root)
    pins = _load_model_pin_manifest(model_pin_path, lock)
    if validate_lock_only:
        return None

    full_frame, dataset_sha = _load_verified_prehack_data(
        csv_path, data_manifest_path
    )
    if len(full_frame) != PREHACK_ROW_COUNT:
        raise FourArmExecutionError("prehack_source_row_count_invalid")
    frame = full_frame.iloc[:CONFIRMATORY_ROW_COUNT].copy().reset_index(drop=True)
    if len(frame) != CONFIRMATORY_ROW_COUNT:
        raise FourArmExecutionError("confirmatory_prefix_row_count_invalid")

    atr = _calculate_atr(frame)
    # The repeated A control is computed once and referenced by all three cells.
    baseline_control = _baseline_signals(frame, atr)
    folds = build_walk_forward_folds(len(frame))
    models = _load_cpu_models(
        pins,
        model_pin_path,
        kronos_code_dir,
        model_artifacts_dir,
    )

    # One global immutable cache. No model inference occurs inside fold or trial loops.
    forecast_cache = _build_forecast_cache(
        frame, models, pins, lock, dataset_sha
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "forecast-cache.json",
        {
            "schema_version": "alix.four-arm.forecast-cache.v1",
            "identity_sha256": forecast_cache.identity_sha256,
            "cache_sha256": forecast_cache.cache_sha256,
            "confirmatory_rows": CONFIRMATORY_ROW_COUNT,
            "anchor_count": len(forecast_cache.anchors),
            "model_inference_calls_per_model": len(forecast_cache.anchors),
            "model_inference_calls_total": 2 * len(forecast_cache.anchors),
            "entries": json.loads(forecast_cache.entries_json),
        },
    )

    trial_manifests: list[dict[str, Any]] = []
    for threshold_bps in THRESHOLDS_BPS:
        for arm_id in ("A", "B", "C", "D"):
            trial_id = TRIAL_ID_MATRIX[threshold_bps][arm_id]
            signals = (
                baseline_control
                if arm_id == "A"
                else _signals_from_forecast_cache(
                    forecast_cache,
                    len(frame),
                    arm_id,
                    threshold_bps,
                )
            )
            fold_trades: dict[int, Sequence[Trade]] = {}
            for fold in folds:
                fold_trades[fold.fold_id] = _simulate_fold_trades(
                    frame,
                    fold,
                    signals,
                    atr,
                )

            manifest = _arm_manifest(
                arm_id,
                trial_id,
                threshold_bps,
                fold_trades,
                dataset_sha,
                folds,
                forecast_cache,
            )
            manifest["mode"] = "REAL_PREHACK_ADR038_FORECAST_DRIVEN"
            manifest["model_pins_captured_at_utc"] = pins["captured_at_utc"]
            manifest["model_pins_path"] = str(model_pin_path)
            manifest["forecast_refresh_bars"] = FORECAST_REFRESH_BARS
            manifest["context_bars"] = MODEL_CONTEXT_BARS
            manifest["common_risk_contract"] = {
                "atr_period": ATR_PERIOD,
                "trailing_stop_atr_multiple": ATR_MULTIPLIER,
                "maximum_holding_bars": MAX_HOLDING_BARS,
                "maximum_drawdown": 0.15,
            }
            manifest["signal_definition"] = (
                "Donchian-20 close breakout control"
                if arm_id == "A"
                else (
                    "Kronos-Small forecast return > fixed threshold"
                    if arm_id == "B"
                    else (
                        "Chronos-2 forecast return > fixed threshold"
                        if arm_id == "C"
                        else "arithmetic mean of Kronos and Chronos returns > fixed threshold"
                    )
                )
            )
            filename = f"trial-{threshold_bps}-{arm_id}.json"
            _write_json(output_dir / filename, manifest)
            trial_manifests.append(manifest)

    if len(trial_manifests) != 12:
        raise FourArmExecutionError("registered_trial_inventory_mismatch")
    observed_ids = {item["trial_id"] for item in trial_manifests}
    expected_ids = {
        trial_id
        for threshold_map in TRIAL_ID_MATRIX.values()
        for trial_id in threshold_map.values()
    }
    if observed_ids != expected_ids:
        raise FourArmExecutionError("registered_trial_ids_mismatch")

    report = {
        "schema_version": "alix.four-arm-run-report.v2",
        "mode": "REAL_PREHACK_ADR038_FORECAST_DRIVEN",
        "dataset_sha256": dataset_sha,
        "source_dataset_rows": PREHACK_ROW_COUNT,
        "confirmatory_dataset_rows": len(frame),
        "excluded_tail_rows": EXCLUDED_TAIL_ROWS,
        "confirmatory_boundary": "first_27825_rows_only",
        "folds": [asdict(fold) for fold in folds],
        "thresholds_bps": list(THRESHOLDS_BPS),
        "trial_count": len(trial_manifests),
        "registered_trial_ids": sorted(observed_ids),
        "dsr_cohort": {
            "registered_candidate_cells": 12,
            "unique_signal_rules": 10,
            "threshold_invariant_arm_a_control_cells": 3,
            "arm_a_control_computed_once_and_referenced_three_times": True,
            "global_n": "UNKNOWN",
            "audited_historical_lower_bound": HISTORICAL_TRIAL_LOWER_BOUND,
            "sensitivity_multipliers": [1, 2, 5, 10],
            "numeric_dsr_computed": False,
        },
        "forecast_cache": {
            "identity_sha256": forecast_cache.identity_sha256,
            "cache_sha256": forecast_cache.cache_sha256,
            "anchors": len(forecast_cache.anchors),
            "inference_calls_per_model": len(forecast_cache.anchors),
            "inference_calls_total": 2 * len(forecast_cache.anchors),
            "immutable_in_memory": True,
        },
        "arms": [
            {
                "arm_id": item["arm_id"],
                "trial_id": item["trial_id"],
                "threshold_bps": item["threshold_bps"],
                "netPF_50bps": item["metrics"]["50"]["net_profit_factor"],
                "netPF_90bps": item["metrics"]["90"]["net_profit_factor"],
                "trades": item["metrics"]["trades"],
                "sample_status": item["sample_status"],
                "maxDD": item["metrics"]["max_drawdown"],
                "DSR": None,
                "DSR_status": item["dsr_status"],
                "DSR_sensitivity": item["dsr_sensitivity"],
                "folds_passed": item["folds_passed"],
                "per_fold": item["folds"],
                "winner": False,
                "pass": False,
                "decision": item["decision"],
            }
            for item in trial_manifests
        ],
        "economic_verdict": "NO_GO_INSUFFICIENT_DSR_EVIDENCE",
        "directional_family_verdict": "NO_GO_UNDER_GLOBAL_N_UNKNOWN",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "posthack_holdout_access": False,
    }
    _write_json(output_dir / "four-arm-run-report.json", report)
    return report




def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "synthetic"), default="real")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("config/experiment/four_arm_execution_lock.json"),
    )
    parser.add_argument("--model-pins", type=Path, default=MODEL_PIN_PATH)
    parser.add_argument(
        "--model-artifacts-dir", type=Path, default=Path("artifacts/four-arm-models")
    )
    parser.add_argument("--kronos-code-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--validate-lock-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.mode == "synthetic":
        synthetic_report = run_synthetic_mode(arguments.output_dir)
        print(f"FOUR_ARM_SYNTHETIC_VALIDATION=PASS:{synthetic_report['arm_count']}")
        return 0
    if not arguments.validate_lock_only and (
        arguments.csv is None
        or arguments.data_manifest is None
        or arguments.kronos_code_dir is None
    ):
        raise SystemExit("real_mode_requires_csv_manifest_and_pinned_kronos_source")
    try:
        real_report = run_real_experiment(
            csv_path=arguments.csv,
            data_manifest_path=arguments.data_manifest,
            lock_path=arguments.lock,
            model_pin_path=arguments.model_pins,
            kronos_code_dir=arguments.kronos_code_dir,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
            model_artifacts_dir=arguments.model_artifacts_dir,
            validate_lock_only=arguments.validate_lock_only,
        )
    except ExecutionAuthorizationError as exc:
        print(f"Execution not authorized: {exc}")
        return 2
    except FourArmExecutionError as exc:
        print(f"FOUR_ARM_EXECUTION_REJECTED={exc}")
        return 3
    if arguments.validate_lock_only:
        print("FOUR_ARM_EXECUTION_LOCK=PASS")
        return 0
    if real_report is None:
        print("FOUR_ARM_REPORT_MISSING")
        return 4
    print(
        "FOUR_ARM_REAL_EXECUTION_COMPLETE:"
        f"dataset_sha256={real_report['dataset_sha256']}:"
        f"rows={real_report['dataset_rows']}:"
        "economic_verdict=NO_GO_INSUFFICIENT_DSR_EVIDENCE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
