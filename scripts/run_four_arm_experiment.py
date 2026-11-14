"""Fail-closed four-arm real pre-hack experiment runner plus synthetic smoke mode."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
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
MODEL_PIN_PATH = Path("config/experiment/four_arm_model_pins.json")
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
    "A": "NOBITEX-KC26-A",
    "B": "NOBITEX-KC26-B",
    "C": "NOBITEX-KC26-C",
    "D": "NOBITEX-KC26-D",
}
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


def _load_model_pin_manifest(path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    try:
        pins = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise FourArmExecutionError("model_pin_manifest_unreadable") from None
    if (
        not isinstance(pins, dict)
        or pins.get("schema_version") != "alix.four-arm.model-pins.v1"
    ):
        raise FourArmExecutionError("model_pin_manifest_schema_invalid")
    models = pins.get("models")
    locked_models = lock.get("models")
    if not isinstance(models, dict) or not isinstance(locked_models, Mapping):
        raise FourArmExecutionError("model_pin_manifest_models_invalid")
    checks = (
        ("kronos_small", "kronos_small"),
        ("kronos_tokenizer", "kronos_small.tokenizer"),
        ("chronos_2", "chronos_2"),
    )
    for pin_key, lock_key in checks:
        pin = models.get(pin_key)
        if not isinstance(pin, dict):
            raise FourArmExecutionError(f"model_pin_missing:{pin_key}")
        if lock_key == "kronos_small.tokenizer":
            lock_model = locked_models.get("kronos_small")
            locked = (
                lock_model.get("tokenizer") if isinstance(lock_model, Mapping) else None
            )
        else:
            locked = locked_models.get(lock_key)
        if not isinstance(locked, Mapping):
            raise FourArmExecutionError(f"model_lock_missing:{pin_key}")
        comparisons = (
            ("repository", "repository"),
            ("revision", "revision"),
            ("weights_sha256", "weights_sha256"),
        )
        for pin_field, lock_field in comparisons:
            if pin.get(pin_field) != locked.get(lock_field):
                raise FourArmExecutionError(
                    f"model_pin_lock_mismatch:{pin_key}:{pin_field}"
                )
    return pins


def _load_verified_prehack_data(
    csv_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, str]:
    if csv_path.name != PREHACK_FILENAME:
        raise FourArmExecutionError("only_registered_prehack_csv_is_allowed")
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FourArmExecutionError("prehack_artifact_file_missing")
    csv_sha = _sha256_file(csv_path)
    if csv_sha != PREHACK_DATASET_SHA256:
        raise FourArmExecutionError("prehack_csv_sha256_mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise FourArmExecutionError("prehack_manifest_unreadable") from None
    if not isinstance(manifest, dict):
        raise FourArmExecutionError("prehack_manifest_schema_invalid")
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

    try:
        frame = pd.read_csv(csv_path)
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
    if frame["timestamp"].is_monotonic_increasing is not True:
        raise FourArmExecutionError("prehack_timestamps_not_strictly_increasing")
    seconds = frame["timestamp"].astype("int64") // 1_000_000_000
    differences = seconds.diff().dropna()
    if not bool((differences == 3600).all()):
        raise FourArmExecutionError("prehack_hourly_grid_invalid")
    if not bool((seconds % 3600 == 1800).all()):
        raise FourArmExecutionError("prehack_utc_offset_grid_invalid")
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


def _snapshot_model(
    repository: str,
    revision: str,
    expected_sha: str,
    cache_dir: Path,
) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            cache_dir=str(cache_dir),
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "*.json",
                "*.txt",
                "*.model",
            ],
        )
    )
    weight_path = snapshot_path / "model.safetensors"
    if not weight_path.is_file() or _sha256_file(weight_path) != expected_sha:
        raise FourArmExecutionError(
            f"downloaded_model_weight_hash_mismatch:{repository}"
        )
    return snapshot_path


def _load_cpu_models(
    pins: Mapping[str, Any],
    kronos_code_dir: Path,
    cache_dir: Path,
) -> tuple[Any, Any]:
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    kronos_code = str(kronos_code_dir.resolve())
    if not (kronos_code_dir / "model" / "__init__.py").is_file():
        raise FourArmExecutionError("pinned_kronos_source_tree_missing")
    if kronos_code not in sys.path:
        sys.path.insert(0, kronos_code)
    from model import Kronos, KronosPredictor, KronosTokenizer
    from chronos import Chronos2Pipeline

    models = pins["models"]
    kronos_pin = models["kronos_small"]
    tokenizer_pin = models["kronos_tokenizer"]
    chronos_pin = models["chronos_2"]

    kronos_path = _snapshot_model(
        kronos_pin["repository"],
        kronos_pin["revision"],
        kronos_pin["weights_sha256"],
        cache_dir,
    )
    tokenizer_path = _snapshot_model(
        tokenizer_pin["repository"],
        tokenizer_pin["revision"],
        tokenizer_pin["weights_sha256"],
        cache_dir,
    )
    chronos_path = _snapshot_model(
        chronos_pin["repository"],
        chronos_pin["revision"],
        chronos_pin["weights_sha256"],
        cache_dir,
    )

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    kronos_model = Kronos.from_pretrained(str(kronos_path))
    kronos_model.eval()
    tokenizer.eval()
    kronos_predictor = KronosPredictor(
        kronos_model,
        tokenizer,
        device="cpu",
        max_context=MODEL_CONTEXT_BARS,
    )
    chronos_pipeline = Chronos2Pipeline.from_pretrained(
        str(chronos_path),
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

    import torch

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


def _build_model_filter_signals(
    frame: pd.DataFrame,
    test_start: int,
    test_end: int,
    kronos_predictor: Any,
    chronos_pipeline: Any,
) -> dict[str, np.ndarray]:
    result = {
        "B": np.zeros(len(frame), dtype=bool),
        "C": np.zeros(len(frame), dtype=bool),
        "D": np.zeros(len(frame), dtype=bool),
    }
    for anchor in range(test_start, test_end, FORECAST_REFRESH_BARS):
        forecast = _forecast_at(frame, anchor, kronos_predictor, chronos_pipeline)
        mean_return = (forecast.kronos_return + forecast.chronos_return) / 2.0
        stop = min(anchor + FORECAST_REFRESH_BARS, test_end)
        result["B"][anchor:stop] = forecast.kronos_return > 0.0
        result["C"][anchor:stop] = forecast.chronos_return > 0.0
        result["D"][anchor:stop] = mean_return > 0.0
    return result


def _simulate_fold_trades(
    frame: pd.DataFrame,
    fold: Fold,
    signals: np.ndarray,
    atr: np.ndarray,
) -> list[Trade]:
    open_values = frame["open"].to_numpy(dtype=np.float64)
    high_values = frame["high"].to_numpy(dtype=np.float64)
    low_values = frame["low"].to_numpy(dtype=np.float64)
    close_values = frame["close"].to_numpy(dtype=np.float64)
    timestamps = frame["timestamp"]
    trades: list[Trade] = []
    cursor = fold.test_start
    while cursor < fold.test_end - 1:
        if not bool(signals[cursor]) or not math.isfinite(float(atr[cursor])):
            cursor += 1
            continue
        entry_index = cursor + 1
        entry_price = float(open_values[entry_index])
        initial_risk = ATR_MULTIPLIER * float(atr[cursor])
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
                exit_index = bar_index
                exit_price = bar_open
                exit_reason = "gap_stop"
                break
            if bar_open >= target_price:
                exit_index = bar_index
                exit_price = bar_open
                exit_reason = "gap_target"
                break
            stop_touched = bar_low <= stop_price
            target_touched = bar_high >= target_price
            if stop_touched:
                exit_index = bar_index
                exit_price = stop_price
                exit_reason = "dual_touch_stop_first" if target_touched else "stop_loss"
                break
            if target_touched:
                exit_index = bar_index
                exit_price = target_price
                exit_reason = "take_profit"
                break
            if bar_index == last_bar:
                exit_index = bar_index
                exit_price = bar_close
                exit_reason = "maximum_holding_time"
                break
            high_water = max(high_water, bar_high)
            current_atr = float(atr[bar_index])
            if math.isfinite(current_atr) and current_atr > 0.0:
                stop_price = max(stop_price, high_water - ATR_MULTIPLIER * current_atr)

        gross_return = exit_price / entry_price - 1.0
        trades.append(
            Trade(
                entry_index=entry_index,
                exit_index=exit_index,
                entry_timestamp=timestamps.iloc[entry_index].isoformat(),
                exit_timestamp=timestamps.iloc[exit_index].isoformat(),
                gross_return=gross_return,
                exit_reason=exit_reason,
            )
        )
        cursor = exit_index + 1
    return trades


def _calculate_metrics(trades: Sequence[Trade], cost_bps: int) -> dict[str, Any]:
    net_returns = [trade.gross_return - cost_bps / 10_000.0 for trade in trades]
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


def _arm_manifest(
    arm_id: str,
    trial_id: str,
    fold_trades: Mapping[int, Sequence[Trade]],
    dataset_sha256: str,
    folds: Sequence[Fold],
) -> dict[str, Any]:
    all_trades = sorted(
        (trade for trades in fold_trades.values() for trade in trades),
        key=lambda trade: (trade.exit_index, trade.entry_index),
    )
    aggregate = {
        str(cost): _calculate_metrics(all_trades, cost) for cost in COST_SCENARIOS_BPS
    }
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
    aggregate_trades = len(all_trades)
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
    return {
        "schema_version": "alix.four-arm-arm-manifest.v1",
        "arm_id": arm_id,
        "trial_id": trial_id,
        "dataset_sha256": dataset_sha256,
        "dataset_rows": PREHACK_ROW_COUNT,
        "cost_scenarios_bps": list(COST_SCENARIOS_BPS),
        "metrics": {
            "50": aggregate["50"],
            "90": aggregate["90"],
            "trades": aggregate_trades,
            "max_drawdown": aggregate_drawdown,
        },
        "folds": fold_results,
        "folds_passed": sum(fold_passes),
        "dsr": None,
        "dsr_status": "INSUFFICIENT_EVIDENCE_GLOBAL_N_UNKNOWN",
        "economic_gates_except_dsr_pass": economics_pass,
        "pass": False,
        "winner": False,
        "decision": "NO_GO_INSUFFICIENT_DSR_EVIDENCE",
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
    validate_lock_only: bool = False,
) -> dict[str, Any] | None:
    """Validate authorization and all immutable contracts before any research I/O."""
    lock = load_owner_authorized_execution_lock(lock_path)
    _validate_lock_contract_fingerprints(lock, repository_root)
    pins = _load_model_pin_manifest(model_pin_path, lock)
    if validate_lock_only:
        return None

    frame, dataset_sha = _load_verified_prehack_data(csv_path, data_manifest_path)
    atr = _calculate_atr(frame)
    baseline = _baseline_signals(frame, atr)
    folds = build_walk_forward_folds(len(frame))
    models = _load_cpu_models(
        pins,
        kronos_code_dir,
        output_dir / "huggingface-cache",
    )

    fold_trades: dict[str, dict[int, Sequence[Trade]]] = {
        arm_id: {} for arm_id in TRIALS
    }
    for fold in folds:
        filters = _build_model_filter_signals(
            frame,
            fold.test_start,
            fold.test_end,
            models[0],
            models[1],
        )
        arm_signals = {
            "A": baseline.copy(),
            "B": baseline & filters["B"],
            "C": baseline & filters["C"],
            "D": baseline & filters["D"],
        }
        for arm_id, signals in arm_signals.items():
            fold_trades[arm_id][fold.fold_id] = _simulate_fold_trades(
                frame,
                fold,
                signals,
                atr,
            )

    arm_manifests = []
    for arm_id, trial_id in TRIALS.items():
        manifest = _arm_manifest(
            arm_id,
            trial_id,
            fold_trades[arm_id],
            dataset_sha,
            folds,
        )
        manifest["mode"] = "REAL_PREHACK_ZERO_SHOT"
        manifest["model_pins_captured_at_utc"] = pins["captured_at_utc"]
        manifest["forecast_refresh_bars"] = FORECAST_REFRESH_BARS
        manifest["context_bars"] = MODEL_CONTEXT_BARS
        _write_json(output_dir / f"arm-{arm_id}.json", manifest)
        arm_manifests.append(manifest)

    report = {
        "schema_version": "alix.four-arm-run-report.v1",
        "mode": "REAL_PREHACK_ZERO_SHOT",
        "dataset_sha256": dataset_sha,
        "dataset_rows": len(frame),
        "folds": [asdict(fold) for fold in folds],
        "arms": [
            {
                "arm_id": item["arm_id"],
                "trial_id": item["trial_id"],
                "netPF_50bps": item["metrics"]["50"]["net_profit_factor"],
                "netPF_90bps": item["metrics"]["90"]["net_profit_factor"],
                "trades": item["metrics"]["trades"],
                "maxDD": item["metrics"]["max_drawdown"],
                "DSR": None,
                "DSR_status": item["dsr_status"],
                "winner": False,
                "pass": False,
            }
            for item in arm_manifests
        ],
        "economic_verdict": "NO_GO_INSUFFICIENT_DSR_EVIDENCE",
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
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
    parser.add_argument("--kronos-code-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--validate-lock-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.mode == "synthetic":
        report = run_synthetic_mode(arguments.output_dir)
        print(f"FOUR_ARM_SYNTHETIC_VALIDATION=PASS:{report['arm_count']}")
        return 0
    if not arguments.validate_lock_only and (
        arguments.csv is None
        or arguments.data_manifest is None
        or arguments.kronos_code_dir is None
    ):
        raise SystemExit("real_mode_requires_csv_manifest_and_pinned_kronos_source")
    try:
        report = run_real_experiment(
            csv_path=arguments.csv,
            data_manifest_path=arguments.data_manifest,
            lock_path=arguments.lock,
            model_pin_path=arguments.model_pins,
            kronos_code_dir=arguments.kronos_code_dir,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
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
    if report is None:
        print("FOUR_ARM_REPORT_MISSING")
        return 4
    print(
        "FOUR_ARM_REAL_EXECUTION_COMPLETE:"
        f"dataset_sha256={report['dataset_sha256']}:"
        f"rows={report['dataset_rows']}:"
        "economic_verdict=NO_GO_INSUFFICIENT_DSR_EVIDENCE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
