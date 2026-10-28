"""Train and evaluate an offline LSTM against a forward-looking Triple-Barrier oracle.

Future candles are used only to create target labels. Input features are point-in-time,
and this research script never creates orders or accesses private exchange APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scripts.run_imitation_learning_agent import (
    _compute_class_weights,
    _weighted_cross_entropy_loss,
    action_distribution,
    classification_metrics,
    load_public_market_data,
    make_sequences,
    mask_spot_actions,
    simulate_economics,
)

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
_ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}
_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_DEFAULT_HORIZON = 24
_DEFAULT_TAKE_PROFIT = 0.03
_DEFAULT_STOP_LOSS = 0.015
_DEFAULT_WINDOW = 48
_DEFAULT_EPOCHS = 20
_DEFAULT_SEED = 20261028
_FEE_BPS_PER_SIDE = 25.0
_SLIPPAGE_BPS_PER_SIDE = 20.0
_MINIMUM_PROFIT_FACTOR = 1.5
_MAXIMUM_DRAWDOWN = 0.15
_MINIMUM_OOS_SIGNAL_COUNT = 800


def generate_oracle_labels(
    frame: pd.DataFrame,
    *,
    horizon: int = _DEFAULT_HORIZON,
    take_profit_fraction: float = _DEFAULT_TAKE_PROFIT,
    stop_loss_fraction: float = _DEFAULT_STOP_LOSS,
) -> pd.DataFrame:
    """Label candle t using barriers touched from the next candle through t+horizon.

    The entry reference is the next candle's open, so a decision at candle t can be
    executed no earlier than candle t+1. When both barriers occur inside one OHLC
    candle, stop-loss is treated as first because intrabar ordering is unknowable.
    """
    _validate_oracle_configuration(horizon, take_profit_fraction, stop_loss_fraction)
    _validate_ohlcv_frame(frame)
    count = len(frame)
    opens = frame["open"].to_numpy(dtype=np.float64)
    highs = frame["high"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64)

    actions = np.full(count, ACTION_HOLD, dtype=np.int64)
    outcomes = np.full(count, "timeout_hold", dtype=object)
    entry_prices = np.full(count, np.nan, dtype=np.float64)
    valid_targets = np.zeros(count, dtype=np.bool_)

    for index in range(count - horizon):
        entry_price = float(opens[index + 1])
        upper_barrier = entry_price * (1.0 + take_profit_fraction)
        lower_barrier = entry_price * (1.0 - stop_loss_fraction)
        entry_prices[index] = entry_price
        valid_targets[index] = True

        for future_index in range(index + 1, index + horizon + 1):
            upper_hit = highs[future_index] >= upper_barrier
            lower_hit = lows[future_index] <= lower_barrier
            if lower_hit:
                actions[index] = ACTION_SELL
                outcomes[index] = (
                    "same_candle_both_barriers_stop_first"
                    if upper_hit
                    else "stop_loss_first"
                )
                break
            if upper_hit:
                actions[index] = ACTION_BUY
                outcomes[index] = "take_profit_first"
                break

    return pd.DataFrame(
        {
            "oracle_action": actions,
            "barrier_outcome": outcomes,
            "entry_price": entry_prices,
            "label_valid": valid_targets,
        },
        index=frame.index,
    )


def apply_partition_horizon_mask(
    label_valid: NDArray[np.bool_],
    *,
    row_count: int,
    train_end: int,
    validation_end: int,
    horizon: int,
) -> NDArray[np.bool_]:
    """Exclude labels whose future window crosses a chronological split."""
    if (
        label_valid.ndim != 1
        or len(label_valid) != row_count
        or horizon < 1
        or not 0 < train_end < validation_end < row_count
    ):
        raise ValueError("oracle_split_horizon_configuration_invalid")

    indices = np.arange(row_count, dtype=np.int64)
    train_valid = (indices < train_end) & (indices + horizon < train_end)
    validation_valid = (
        (indices >= train_end)
        & (indices < validation_end)
        & (indices + horizon < validation_end)
    )
    holdout_valid = (indices >= validation_end) & (indices + horizon < row_count)
    return label_valid & (train_valid | validation_valid | holdout_valid)


def build_causal_features(frame: pd.DataFrame) -> tuple[NDArray[np.float32], list[str]]:
    """Build point-in-time OHLCV, RSI, volatility and trend features."""
    _validate_ohlcv_frame(frame)
    open_price = frame["open"].astype(np.float64)
    high = frame["high"].astype(np.float64)
    low = frame["low"].astype(np.float64)
    close = frame["close"].astype(np.float64)
    volume = frame["volume"].astype(np.float64)

    previous_close = close.shift(1).fillna(close.iloc[0])
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_fraction = true_range.rolling(14, min_periods=1).mean() / close

    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    losses = -delta.clip(upper=0.0).rolling(14, min_periods=14).mean()
    relative_strength = gains / losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((losses == 0.0) & (gains > 0.0), 100.0).fillna(50.0)

    log_volume = pd.Series(
        np.log1p(volume.clip(lower=0.0)), index=frame.index, dtype=np.float64
    )
    rolling_volume_mean = log_volume.rolling(24, min_periods=1).mean()
    rolling_volume_std = log_volume.rolling(24, min_periods=2).std()
    volume_z = (log_volume - rolling_volume_mean) / rolling_volume_std.replace(
        0.0, np.nan
    )
    typical_price = (high + low + close) / 3.0
    rolling_volume_sum = volume.rolling(24, min_periods=1).sum().replace(0.0, np.nan)
    vwap = (typical_price * volume).rolling(
        24, min_periods=1
    ).sum() / rolling_volume_sum
    mean_20 = close.rolling(20, min_periods=1).mean()
    std_20 = close.rolling(20, min_periods=2).std().replace(0.0, np.nan)

    feature_frame = pd.DataFrame(
        {
            "return_1": close.pct_change(1, fill_method=None),
            "return_3": close.pct_change(3, fill_method=None),
            "return_6": close.pct_change(6, fill_method=None),
            "return_24": close.pct_change(24, fill_method=None),
            "candle_body_fraction": (close - open_price) / open_price,
            "candle_range_fraction": (high - low) / close,
            "upper_wick_fraction": (high - np.maximum(open_price, close)) / close,
            "lower_wick_fraction": (np.minimum(open_price, close) - low) / close,
            "rsi_14_scaled": rsi / 100.0,
            "atr_fraction_14": atr_fraction,
            "bollinger_z_20": (close - mean_20) / std_20,
            "sma_distance_6": close / close.rolling(6, min_periods=1).mean() - 1.0,
            "sma_distance_24": close / close.rolling(24, min_periods=1).mean() - 1.0,
            "sma_distance_72": close / close.rolling(72, min_periods=1).mean() - 1.0,
            "sma_distance_168": close / close.rolling(168, min_periods=1).mean() - 1.0,
            "vwap_distance_24": close / vwap - 1.0,
            "log_volume_z_24": volume_z,
        },
        index=frame.index,
    )
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = feature_frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("oracle_feature_matrix_non_finite")
    return values, [str(column) for column in feature_frame.columns]


def _validate_oracle_configuration(
    horizon: int, take_profit_fraction: float, stop_loss_fraction: float
) -> None:
    if isinstance(horizon, bool) or not 1 <= horizon <= 240:
        raise ValueError("oracle_horizon_out_of_range")
    fractions = np.asarray([take_profit_fraction, stop_loss_fraction], dtype=np.float64)
    if (
        not np.isfinite(fractions).all()
        or np.any(fractions <= 0.0)
        or np.any(fractions >= 0.5)
    ):
        raise ValueError("oracle_barrier_fraction_out_of_range")


def _validate_ohlcv_frame(frame: pd.DataFrame) -> None:
    missing = set(_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"oracle_market_columns_missing:{','.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("oracle_market_frame_empty")
    numeric_columns = ("open", "high", "low", "close", "volume")
    numeric = frame.loc[:, list(numeric_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("oracle_market_frame_non_finite")
    if np.any(numeric[:, :4] <= 0.0) or np.any(numeric[:, 4] < 0.0):
        raise ValueError("oracle_market_frame_price_or_volume_invalid")
    if np.any(frame["high"].to_numpy() < frame["low"].to_numpy()):
        raise ValueError("oracle_market_frame_high_below_low")
    if len(frame) < 3:
        raise ValueError("oracle_market_frame_too_short")


def _split_boundaries(row_count: int, window: int, horizon: int) -> tuple[int, int]:
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    if (
        train_end <= window + horizon
        or validation_end - train_end <= window + horizon
        or row_count - validation_end <= window + horizon
    ):
        raise ValueError("oracle_training_splits_too_short")
    return train_end, validation_end


def _prepare_sequences(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    window: int,
    horizon: int,
) -> dict[str, Any]:
    row_count = len(frame)
    train_end, validation_end = _split_boundaries(row_count, window, horizon)
    labels = labels.copy()
    labels["label_valid"] = apply_partition_horizon_mask(
        labels["label_valid"].to_numpy(dtype=np.bool_),
        row_count=row_count,
        train_end=train_end,
        validation_end=validation_end,
        horizon=horizon,
    )

    actions = labels["oracle_action"].to_numpy(dtype=np.int64)
    features, feature_names = build_causal_features(frame)
    scaler_mean = features[:train_end].mean(axis=0, dtype=np.float64)
    scaler_scale = features[:train_end].std(axis=0, dtype=np.float64)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    normalized = ((features - scaler_mean) / scaler_scale).astype(np.float32)
    sequences, sequence_labels, target_indices = make_sequences(
        normalized, actions, window
    )
    target_valid = labels["label_valid"].to_numpy(dtype=np.bool_)[target_indices]
    train_mask = (target_indices < train_end) & target_valid
    validation_mask = (
        (target_indices >= train_end) & (target_indices < validation_end) & target_valid
    )
    holdout_mask = (target_indices >= validation_end) & target_valid
    if not train_mask.any() or not validation_mask.any() or not holdout_mask.any():
        raise ValueError("oracle_sequence_split_empty")
    train_labels = sequence_labels[train_mask]
    class_weights = _compute_class_weights(train_labels)
    return {
        "features": features,
        "feature_names": feature_names,
        "sequences": sequences,
        "sequence_labels": sequence_labels,
        "target_indices": target_indices,
        "target_valid": target_valid,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "holdout_mask": holdout_mask,
        "train_labels": train_labels,
        "class_weights": class_weights,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "train_end": train_end,
        "validation_end": validation_end,
        "row_count": row_count,
    }


def _snapshot_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _predict_actions(
    torch: Any,
    lstm: Any,
    classifier: Any,
    sequences: NDArray[np.float32],
) -> NDArray[np.int64]:
    if len(sequences) == 0:
        raise ValueError("oracle_prediction_split_empty")
    lstm.eval()
    classifier.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(sequences, dtype=torch.float32)
        outputs, _ = lstm(tensor)
        logits = classifier(outputs[:, -1, :])
        predictions = torch.argmax(logits, dim=1)
    return predictions.detach().cpu().numpy().astype(np.int64, copy=False)


def _train_lstm(
    torch: Any, prepared: dict[str, Any], *, epochs: int, seed: int
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    sequences = prepared["sequences"]
    train_mask = prepared["train_mask"]
    validation_mask = prepared["validation_mask"]
    train_features = torch.as_tensor(sequences[train_mask], dtype=torch.float32)
    train_targets = torch.as_tensor(
        prepared["sequence_labels"][train_mask], dtype=torch.long
    )
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_targets),
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )
    feature_count = int(prepared["features"].shape[1])
    lstm = torch.nn.LSTM(
        input_size=feature_count,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        dropout=0.15,
    )
    classifier = torch.nn.Linear(64, 3)
    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(classifier.parameters()),
        lr=0.001,
    )
    loss_weights = torch.as_tensor(prepared["class_weights"], dtype=torch.float32)
    validation_sequences = sequences[validation_mask]
    validation_targets = prepared["sequence_labels"][validation_mask]
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0
    best_lstm_state: dict[str, Any] = {}
    best_classifier_state: dict[str, Any] = {}

    for epoch in range(epochs):
        lstm.train()
        classifier.train()
        for batch_features, batch_targets in data_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = lstm(batch_features)
            logits = classifier(outputs[:, -1, :])
            loss = _weighted_cross_entropy_loss(
                torch, logits, batch_targets, loss_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lstm.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        completed_epochs = epoch + 1
        predictions = _predict_actions(torch, lstm, classifier, validation_sequences)
        validation_score = classification_metrics(
            validation_targets.astype(np.int64, copy=False), predictions
        )["macro_f1"]
        score = float(validation_score)
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = completed_epochs
            best_lstm_state = _snapshot_state(lstm)
            best_classifier_state = _snapshot_state(classifier)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break

    if not best_lstm_state or not best_classifier_state:
        raise RuntimeError("oracle_lstm_validation_checkpoint_missing")
    lstm.load_state_dict(best_lstm_state)
    classifier.load_state_dict(best_classifier_state)
    return {
        "lstm": lstm,
        "classifier": classifier,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "best_validation_macro_f1": float(best_score),
        "feature_count": feature_count,
        "hidden_size": 64,
        "requested_epochs": epochs,
    }


def _economics_for_predictions(
    actions: NDArray[np.int64],
    target_indices: NDArray[np.int64],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Simulate next-open Spot long/flat fills with costs and final forced close."""
    if len(actions) != len(target_indices) or len(actions) == 0:
        raise ValueError("oracle_economic_input_alignment_invalid")
    if np.any(target_indices + 1 >= len(frame)):
        raise ValueError("oracle_economic_execution_price_unavailable")
    opens = frame["open"].to_numpy(dtype=np.float64)
    execution_prices = opens[target_indices + 1]
    closed_actions = np.concatenate(
        [
            actions.astype(np.int64, copy=False),
            np.asarray([ACTION_SELL], dtype=np.int64),
        ]
    )
    closed_prices = np.concatenate(
        [execution_prices, np.asarray([execution_prices[-1]], dtype=np.float64)]
    )
    return simulate_economics(
        closed_actions,
        closed_prices,
        fee_bps_per_side=_FEE_BPS_PER_SIDE,
        slippage_bps_per_side=_SLIPPAGE_BPS_PER_SIDE,
    )


def _evaluate_partition(
    *,
    name: str,
    mask: NDArray[np.bool_],
    prepared: dict[str, Any],
    predictions: NDArray[np.int64],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    target_indices = prepared["target_indices"][mask]
    truth = prepared["sequence_labels"][mask].astype(np.int64, copy=False)
    partition_predictions = predictions.astype(np.int64, copy=False)
    effective_actions = mask_spot_actions(partition_predictions)
    return {
        "name": name,
        "row_start_inclusive": int(target_indices[0]),
        "row_end_exclusive": int(target_indices[-1] + 1),
        "sample_count": int(len(truth)),
        "classification_vs_oracle": classification_metrics(
            truth, partition_predictions
        ),
        "oracle_action_distribution": action_distribution(truth),
        "model_raw_action_distribution": action_distribution(partition_predictions),
        "model_spot_action_distribution": action_distribution(effective_actions),
        "economic_metrics_after_costs": _economics_for_predictions(
            partition_predictions, target_indices, frame
        ),
    }


def _build_economic_gate(
    validation: dict[str, Any], holdout: dict[str, Any]
) -> dict[str, Any]:
    validation_economics = validation["economic_metrics_after_costs"]
    holdout_economics = holdout["economic_metrics_after_costs"]
    validation_pf = validation_economics["profit_factor"]
    holdout_pf = holdout_economics["profit_factor"]
    validation_pf_passed = (
        validation_pf is not None and float(validation_pf) > _MINIMUM_PROFIT_FACTOR
    )
    holdout_pf_passed = (
        holdout_pf is not None and float(holdout_pf) > _MINIMUM_PROFIT_FACTOR
    )
    validation_dd_passed = (
        float(validation_economics["max_drawdown_fraction"]) <= _MAXIMUM_DRAWDOWN
    )
    holdout_dd_passed = (
        float(holdout_economics["max_drawdown_fraction"]) <= _MAXIMUM_DRAWDOWN
    )
    validation_signals = sum(
        validation["model_raw_action_distribution"][name]["count"]
        for name in ("BUY", "SELL")
    )
    holdout_signals = sum(
        holdout["model_raw_action_distribution"][name]["count"]
        for name in ("BUY", "SELL")
    )
    signal_count = int(validation_signals + holdout_signals)
    signal_density_passed = signal_count > _MINIMUM_OOS_SIGNAL_COUNT
    passed = all(
        [
            validation_pf_passed,
            holdout_pf_passed,
            validation_dd_passed,
            holdout_dd_passed,
            signal_density_passed,
        ]
    )
    return {
        "passed": passed,
        "minimum_profit_factor": _MINIMUM_PROFIT_FACTOR,
        "profit_factor_comparison": "strictly_greater_than_minimum",
        "maximum_allowed_drawdown_fraction": _MAXIMUM_DRAWDOWN,
        "minimum_oos_model_signal_count": _MINIMUM_OOS_SIGNAL_COUNT,
        "signal_count_comparison": "strictly_greater_than_minimum",
        "oos_model_signal_count": signal_count,
        "validation_profit_factor_passed": validation_pf_passed,
        "holdout_profit_factor_passed": holdout_pf_passed,
        "validation_drawdown_passed": validation_dd_passed,
        "holdout_drawdown_passed": holdout_dd_passed,
        "signal_density_passed": signal_density_passed,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_training(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    commit_sha: str,
    epochs: int = _DEFAULT_EPOCHS,
    window: int = _DEFAULT_WINDOW,
    horizon: int = _DEFAULT_HORIZON,
    take_profit_fraction: float = _DEFAULT_TAKE_PROFIT,
    stop_loss_fraction: float = _DEFAULT_STOP_LOSS,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    if isinstance(epochs, bool) or not 1 <= epochs <= 200:
        raise ValueError("oracle_training_epochs_out_of_range")
    if isinstance(window, bool) or not 8 <= window <= 512:
        raise ValueError("oracle_sequence_window_out_of_range")
    _validate_oracle_configuration(horizon, take_profit_fraction, stop_loss_fraction)
    frame, source_manifest = load_public_market_data(
        dataset_path, manifest_path, commit_sha
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = generate_oracle_labels(
        frame,
        horizon=horizon,
        take_profit_fraction=take_profit_fraction,
        stop_loss_fraction=stop_loss_fraction,
    )
    prepared = _prepare_sequences(frame, labels, window=window, horizon=horizon)
    label_valid = apply_partition_horizon_mask(
        labels["label_valid"].to_numpy(dtype=np.bool_),
        row_count=len(frame),
        train_end=int(prepared["train_end"]),
        validation_end=int(prepared["validation_end"]),
        horizon=horizon,
    )
    labels["label_valid"] = label_valid
    label_actions = labels["oracle_action"].to_numpy(dtype=np.int64)
    label_export = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].astype(str),
            "oracle_action": label_actions,
            "oracle_action_name": [
                _ACTION_NAMES[int(value)] for value in label_actions
            ],
            "barrier_outcome": labels["barrier_outcome"].astype(str),
            "entry_price": labels["entry_price"],
            "label_valid": label_valid,
        }
    )
    label_csv_path = output_dir / "oracle-teacher-dataset.csv"
    label_export.to_csv(label_csv_path, index=False)
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "private_api_calls": 0,
    }
    label_manifest = {
        "schema_version": "alix.oracle-teacher-manifest.v1",
        "commit_sha": commit_sha,
        "dataset_row_count": int(len(frame)),
        "source_csv_sha256": source_manifest["csv_sha256"],
        "teacher_csv_sha256": hashlib.sha256(label_csv_path.read_bytes()).hexdigest(),
        "label_horizon_candles": int(horizon),
        "entry_reference": "open_of_next_candle",
        "take_profit_fraction": float(take_profit_fraction),
        "stop_loss_fraction": float(stop_loss_fraction),
        "same_candle_double_hit_policy": "SELL_STOP_FIRST_CONSERVATIVE",
        "features_use_future_data": False,
        "feature_future_lookahead_candles": 0,
        "boundary_crossing_labels_excluded": True,
        "train_end_exclusive": int(prepared["train_end"]),
        "validation_end_exclusive": int(prepared["validation_end"]),
        "final_holdout_end_exclusive": int(len(frame)),
        "safety": safety,
    }
    _write_json(output_dir / "oracle-teacher-manifest.json", label_manifest)
    _write_json(
        output_dir / "oracle-label-stage-report.json",
        {
            "commit_sha": commit_sha,
            "label_distribution_valid_targets": action_distribution(
                label_actions[label_valid]
            ),
            "valid_target_count": int(np.count_nonzero(label_valid)),
            "invalid_target_count": int(
                len(label_valid) - np.count_nonzero(label_valid)
            ),
            "feature_future_lookahead_candles": 0,
            "features_use_future_data": False,
            "label_horizon_candles": int(horizon),
            "boundary_crossing_labels_excluded": True,
            "safety": safety,
        },
    )

    torch: Any = importlib.import_module("torch")
    training = _train_lstm(torch, prepared, epochs=epochs, seed=seed)
    validation_mask = prepared["validation_mask"]
    holdout_mask = prepared["holdout_mask"]
    validation_predictions = _predict_actions(
        torch,
        training["lstm"],
        training["classifier"],
        prepared["sequences"][validation_mask],
    )
    holdout_predictions = _predict_actions(
        torch,
        training["lstm"],
        training["classifier"],
        prepared["sequences"][holdout_mask],
    )
    validation = _evaluate_partition(
        name="validation",
        mask=validation_mask,
        prepared=prepared,
        predictions=validation_predictions,
        frame=frame,
    )
    holdout = _evaluate_partition(
        name="holdout",
        mask=holdout_mask,
        prepared=prepared,
        predictions=holdout_predictions,
        frame=frame,
    )
    economic_gate = _build_economic_gate(validation, holdout)

    model_path = output_dir / "oracle-lstm-state.pt"
    torch.save(
        {
            "architecture": "two-layer LSTM plus linear 3-class head",
            "lstm_state_dict": training["lstm"].state_dict(),
            "classifier_state_dict": training["classifier"].state_dict(),
            "input_features": prepared["feature_names"],
            "window_candles": int(window),
            "hidden_size": int(training["hidden_size"]),
            "action_map": _ACTION_NAMES,
            "initialized_from_scratch": True,
            "training_commit_sha": commit_sha,
        },
        model_path,
    )
    _write_json(
        output_dir / "oracle-feature-scaler.json",
        {
            "feature_names": prepared["feature_names"],
            "mean": [float(value) for value in prepared["scaler_mean"]],
            "scale": [float(value) for value in prepared["scaler_scale"]],
            "fit_rows": int(prepared["train_end"]),
            "fit_policy": "chronological_training_partition_only",
        },
    )
    train_target_indices = prepared["target_indices"][prepared["train_mask"]]
    validation_target_indices = prepared["target_indices"][validation_mask]
    holdout_target_indices = prepared["target_indices"][holdout_mask]
    report = {
        "schema_version": "alix.oracle-teacher-lstm-report.v1",
        "commit_sha": commit_sha,
        "dataset": {
            "provider": source_manifest["provider"],
            "source_csv_sha256": source_manifest["csv_sha256"],
            "row_count": int(len(frame)),
            "interval": source_manifest.get("interval", "1h"),
            "train_rows": int(prepared["train_end"]),
            "validation_rows": int(prepared["validation_end"] - prepared["train_end"]),
            "holdout_rows": int(len(frame) - prepared["validation_end"]),
        },
        "oracle": {
            "method": "Triple-Barrier forward-label oracle",
            "label_horizon_candles": int(horizon),
            "entry_reference": "open_of_next_candle",
            "take_profit_fraction": float(take_profit_fraction),
            "stop_loss_fraction": float(stop_loss_fraction),
            "tie_policy": "SELL_STOP_FIRST_CONSERVATIVE",
            "valid_target_count": int(np.count_nonzero(label_valid)),
            "invalid_target_count": int(len(label_valid) - np.count_nonzero(label_valid)),
            "action_distribution": action_distribution(label_actions[label_valid]),
            "train_action_distribution": action_distribution(prepared["train_labels"]),
            "validation_action_distribution": action_distribution(
                prepared["sequence_labels"][validation_mask]
            ),
            "holdout_action_distribution": action_distribution(
                prepared["sequence_labels"][holdout_mask]
            ),
            "boundary_crossing_labels_excluded": True,
            "validation_labels_used_for_checkpoint_selection": True,
            "holdout_used_for_training_or_model_selection": False,
        },
        "features": {
            "names": prepared["feature_names"],
            "count": int(len(prepared["feature_names"])),
            "future_data_used": False,
            "future_lookahead_candles": 0,
            "scaler_fit_partition": "training_only",
            "sequence_ends_at_prediction_candle": True,
        },
        "splits": {
            "train_start_inclusive": 0,
            "train_end_exclusive": int(prepared["train_end"]),
            "validation_start_inclusive": int(prepared["train_end"]),
            "validation_end_exclusive": int(prepared["validation_end"]),
            "holdout_start_inclusive": int(prepared["validation_end"]),
            "holdout_end_exclusive": int(len(frame)),
            "label_horizon_purge_candles": int(horizon),
            "train_valid_target_rows": int(len(train_target_indices)),
            "validation_valid_target_rows": int(len(validation_target_indices)),
            "holdout_valid_target_rows": int(len(holdout_target_indices)),
        },
        "model": {
            "architecture": "two-layer CPU LSTM plus linear 3-class head",
            "initialized_from_scratch": True,
            "window_candles": int(window),
            "input_feature_count": int(training["feature_count"]),
            "input_features": prepared["feature_names"],
            "hidden_size": int(training["hidden_size"]),
            "requested_epochs": int(training["requested_epochs"]),
            "completed_epochs": int(training["completed_epochs"]),
            "best_validation_epoch": int(training["best_epoch"]),
            "best_validation_macro_f1": float(training["best_validation_macro_f1"]),
            "seed": int(seed),
            "class_weighting": {
                "derived_from": "chronological_training_partition_only",
                "loss": "per-sample weighted cross entropy",
                "weights_by_action": {
                    name: float(prepared["class_weights"][action])
                    for action, name in _ACTION_NAMES.items()
                },
                "train_class_distribution": action_distribution(
                    prepared["train_labels"]
                ),
                "validation_or_holdout_used_for_weights": False,
            },
            "state_artifact": model_path.name,
        },
        "validation": validation,
        "holdout": holdout,
        "economic_gate": economic_gate,
        "economic_simulation": {
            "market": "Spot long/flat; SELL closes an open long and never shorts",
            "execution_timing": "signals on candle close execute at next candle open",
            "fee_bps_per_side": _FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": _SLIPPAGE_BPS_PER_SIDE,
            "forced_liquidation": "at final evaluated next-open price with exit costs",
            "metrics": [
                "net_return_fraction",
                "max_drawdown_fraction",
                "profit_factor",
                "closed_trade_count",
            ],
        },
        "safety": {
            **safety,
            "execution_api_implemented": False,
        },
        "decision": "RESEARCH_ONLY_NOT_PROMOTED",
    }
    _write_json(output_dir / "oracle-teacher-lstm-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an offline LSTM to predict Triple-Barrier oracle labels."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--window", type=int, default=_DEFAULT_WINDOW)
    parser.add_argument("--horizon", type=int, default=_DEFAULT_HORIZON)
    parser.add_argument(
        "--take-profit-fraction", type=float, default=_DEFAULT_TAKE_PROFIT
    )
    parser.add_argument("--stop-loss-fraction", type=float, default=_DEFAULT_STOP_LOSS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    arguments = parser.parse_args()
    run_training(
        arguments.dataset,
        arguments.manifest,
        arguments.output_dir,
        commit_sha=str(arguments.commit_sha),
        epochs=int(arguments.epochs),
        window=int(arguments.window),
        horizon=int(arguments.horizon),
        take_profit_fraction=float(arguments.take_profit_fraction),
        stop_loss_fraction=float(arguments.stop_loss_fraction),
        seed=int(arguments.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
