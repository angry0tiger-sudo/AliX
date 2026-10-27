"""Train a fresh CPU LSTM to imitate causal teacher labels; never execute orders."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.rl_research.institutional_teacher import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    generate_teacher_labels,
)

FloatArray = NDArray[np.float32]
Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
_ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}
_REQUIRED_SOURCE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest_must_be_json_object")
    return cast(dict[str, Any], payload)


def load_public_market_data(
    dataset_path: Path, manifest_path: Path, commit_sha: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate provenance, hashes and closed safety invariants before use."""
    manifest = _read_json_object(manifest_path)
    expected_safety: dict[str, Any] = {
        "provider": "Nobitex public UDF history API",
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
        "commit_sha": commit_sha,
    }
    for key, expected in expected_safety.items():
        if manifest.get(key) != expected:
            raise ValueError(f"market_manifest_safety_or_identity_mismatch:{key}")
    observed_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if observed_hash != manifest.get("csv_sha256"):
        raise ValueError("market_csv_sha256_mismatch")

    frame = pd.read_csv(dataset_path)
    if "date" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename(columns={"date": "timestamp"})
    missing = set(_REQUIRED_SOURCE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing_market_columns:{','.join(sorted(missing))}")
    if len(frame) != manifest.get("row_count") or frame.empty:
        raise ValueError("market_csv_row_count_mismatch")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("market_timestamps_invalid")
    return frame.loc[:, list(_REQUIRED_SOURCE_COLUMNS)].copy(), manifest


def build_feature_matrix(
    frame: pd.DataFrame, teacher: pd.DataFrame
) -> tuple[FloatArray, list[str]]:
    """Create point-in-time OHLCV features and causal higher-timeframe context."""
    if len(frame) != len(teacher):
        raise ValueError("feature_teacher_row_count_mismatch")
    open_price = frame["open"].astype(np.float64)
    high = frame["high"].astype(np.float64)
    low = frame["low"].astype(np.float64)
    close = frame["close"].astype(np.float64)
    volume = frame["volume"].astype(np.float64)
    typical = (high + low + close) / 3.0
    rolling_volume = volume.rolling(24, min_periods=1).sum().replace(0.0, np.nan)
    vwap = (typical * volume).rolling(24, min_periods=1).sum() / rolling_volume
    log_volume = pd.Series(np.log1p(volume.clip(lower=0.0)), index=frame.index)
    volume_std = log_volume.rolling(24, min_periods=2).std().replace(0.0, np.nan)
    volume_z = (log_volume - log_volume.rolling(24, min_periods=1).mean()) / volume_std

    features = pd.DataFrame(
        {
            "return_1": close.pct_change(1, fill_method=None),
            "return_3": close.pct_change(3, fill_method=None),
            "return_6": close.pct_change(6, fill_method=None),
            "return_24": close.pct_change(24, fill_method=None),
            "candle_body_fraction": (close - open_price) / open_price,
            "candle_range_fraction": (high - low) / close,
            "upper_wick_fraction": (high - np.maximum(open_price, close)) / close,
            "lower_wick_fraction": (np.minimum(open_price, close) - low) / close,
            "volume_z_24": volume_z,
            "vwap_distance_24": close / vwap - 1.0,
            "sma_distance_6": close / close.rolling(6, min_periods=1).mean() - 1.0,
            "sma_distance_24": close / close.rolling(24, min_periods=1).mean() - 1.0,
            "sma_distance_72": close / close.rolling(72, min_periods=1).mean() - 1.0,
            "sma_distance_168": close / close.rolling(168, min_periods=1).mean() - 1.0,
            "htf_bias_4h": teacher["higher_timeframe_bias_4h"].astype(np.float32),
            "htf_bias_1d": teacher["higher_timeframe_bias_1d"].astype(np.float32),
            "bos_reference_distance": (
                close - teacher["bos_reference_high"].astype(np.float64)
            )
            / close,
        },
        index=frame.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    names = [str(column) for column in features.columns]
    values = features.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("feature_matrix_contains_non_finite_values")
    return values, names


def make_sequences(
    features: FloatArray, actions: IntArray, window: int
) -> tuple[FloatArray, IntArray, IntArray]:
    """Return sliding windows and the original candle index for every target."""
    if isinstance(window, bool) or window < 2:
        raise ValueError("sequence_window_out_of_range")
    if features.ndim != 2 or actions.ndim != 1 or len(features) != len(actions):
        raise ValueError("sequence_inputs_shape_mismatch")
    if len(features) <= window:
        raise ValueError("not_enough_rows_for_sequence_window")
    target_indices = np.arange(window - 1, len(features), dtype=np.int64)
    sequences = np.empty(
        (len(target_indices), window, features.shape[1]), dtype=np.float32
    )
    for sequence_index, target_index in enumerate(target_indices):
        start = int(target_index) - window + 1
        stop = int(target_index) + 1
        sequences[sequence_index] = features[start:stop]
    return sequences, actions[target_indices].astype(np.int64), target_indices


def action_distribution(actions: IntArray) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    total = int(len(actions))
    for action, name in _ACTION_NAMES.items():
        count = int(np.count_nonzero(actions == action))
        result[name] = {
            "count": count,
            "fraction": float(count / total) if total else 0.0,
        }
    return result


def classification_metrics(truth: IntArray, predictions: IntArray) -> dict[str, Any]:
    if truth.shape != predictions.shape or truth.ndim != 1:
        raise ValueError("classification_metric_shape_mismatch")
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual in range(3):
        for predicted in range(3):
            matrix[actual, predicted] = int(
                np.count_nonzero((truth == actual) & (predictions == predicted))
            )
    class_metrics: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for action, name in _ACTION_NAMES.items():
        true_positive = int(matrix[action, action])
        false_positive = int(matrix[:, action].sum()) - true_positive
        false_negative = int(matrix[action, :].sum()) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        class_metrics[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    accuracy = float(np.mean(truth == predictions)) if len(truth) else 0.0
    return {
        "sample_count": int(len(truth)),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "class_metrics": class_metrics,
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }


def _validate_economic_shapes(actions: IntArray, close_prices: Float64Array) -> None:
    if (
        actions.ndim != 1
        or close_prices.ndim != 1
        or len(actions) != len(close_prices)
        or len(actions) < 2
    ):
        raise ValueError("economic_simulation_shape_mismatch")


def _validate_economic_prices_and_costs(
    close_prices: Float64Array, fee_bps_per_side: float, slippage_bps_per_side: float
) -> None:
    if not np.isfinite(close_prices).all() or np.any(close_prices <= 0):
        raise ValueError("economic_simulation_input_invalid")
    if fee_bps_per_side < 0 or slippage_bps_per_side < 0:
        raise ValueError("economic_simulation_input_invalid")


def _target_positions(actions: IntArray) -> NDArray[np.int8]:
    positions = np.zeros(len(actions), dtype=np.int8)
    position = 0
    for index, action in enumerate(actions):
        if int(action) == ACTION_BUY:
            position = 1
        elif int(action) == ACTION_SELL:
            position = -1
        elif int(action) != ACTION_HOLD:
            raise ValueError("economic_simulation_action_invalid")
        positions[index] = position
    return positions


def _net_period_returns(
    positions: NDArray[np.int8],
    close_prices: Float64Array,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> tuple[Float64Array, IntArray]:
    turnover = np.abs(
        np.diff(np.concatenate((np.array([0], dtype=np.int8), positions)))
    ).astype(np.int64)
    log_returns = np.diff(np.log(close_prices))
    transaction_cost = (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
    net_returns = (
        positions[:-1].astype(np.float64) * log_returns
        - turnover[:-1].astype(np.float64) * transaction_cost
    )
    return net_returns, turnover


def _economic_scorecard(
    net_returns: Float64Array,
    turnover: IntArray,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> dict[str, Any]:
    equity = np.concatenate((np.array([1.0]), np.exp(np.cumsum(net_returns))))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    gross_profit = float(net_returns[net_returns > 0.0].sum())
    gross_loss = float(-net_returns[net_returns < 0.0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else None
    return {
        "net_return_fraction": float(equity[-1] - 1.0),
        "max_drawdown_fraction": float(max(0.0, -float(drawdown.min()))),
        "profit_factor": profit_factor,
        "positive_period_fraction": (
            float(np.mean(net_returns > 0.0)) if len(net_returns) else 0.0
        ),
        "evaluated_periods": int(len(net_returns)),
        "position_change_count": int(np.count_nonzero(turnover[:-1])),
        "fee_bps_per_side": float(fee_bps_per_side),
        "slippage_bps_per_side": float(slippage_bps_per_side),
    }


def simulate_economics(
    actions: IntArray,
    close_prices: Float64Array,
    *,
    fee_bps_per_side: float = 10.0,
    slippage_bps_per_side: float = 5.0,
) -> dict[str, Any]:
    """Offline long/short target-position simulation; no execution integration."""
    _validate_economic_shapes(actions, close_prices)
    _validate_economic_prices_and_costs(
        close_prices, fee_bps_per_side, slippage_bps_per_side
    )
    positions = _target_positions(actions)
    net_returns, turnover = _net_period_returns(
        positions, close_prices, fee_bps_per_side, slippage_bps_per_side
    )
    return _economic_scorecard(
        net_returns, turnover, fee_bps_per_side, slippage_bps_per_side
    )


def _predict_actions(
    torch: Any, lstm: Any, classifier: Any, features: FloatArray
) -> IntArray:
    lstm.eval()
    classifier.eval()
    tensor = torch.tensor(features, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        output, _ = lstm(tensor)
        logits = classifier(output[:, -1, :])
        predictions = logits.argmax(dim=1).cpu().numpy()
    return np.asarray(predictions, dtype=np.int64)


def _validate_training_parameters(epochs: int, window: int) -> None:
    if isinstance(epochs, bool) or not 1 <= epochs <= 100:
        raise ValueError("epochs_out_of_range")
    if isinstance(window, bool) or not 8 <= window <= 256:
        raise ValueError("sequence_window_out_of_range")


def _write_teacher_dataset(
    frame: pd.DataFrame,
    source_manifest: dict[str, Any],
    output_dir: Path,
    commit_sha: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    teacher = generate_teacher_labels(
        frame,
        higher_timeframe="4h+1D",
        ema_span=12,
        swing_window=20,
        volume_window=60,
        cooldown_candles=5,
    )
    teacher_path = output_dir / "teacher-dataset.csv"
    teacher.to_csv(teacher_path, index=False)
    teacher_manifest: dict[str, Any] = {
        "schema_version": "alix.imitation-teacher-dataset.v1",
        "commit_sha": commit_sha,
        "source_provider": source_manifest["provider"],
        "source_csv_sha256": source_manifest["csv_sha256"],
        "teacher_csv_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "row_count": int(len(teacher)),
        "teacher_timeframe_mode": "4h+1D_confluence",
        "teacher_action_distribution": action_distribution(
            teacher["teacher_action"].to_numpy(dtype=np.int64)
        ),
        "teacher_label_source": "causal OHLCV heuristic; not ground-truth institutional order flow",
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    manifest_path = output_dir / "teacher-dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(teacher_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return teacher, teacher_manifest


def _validate_split_boundaries(
    row_count: int, window: int, train_end: int, validation_end: int
) -> None:
    if (
        train_end <= window
        or validation_end <= train_end
        or row_count <= validation_end
    ):
        raise ValueError("insufficient_rows_for_chronological_splits")


def _validate_split_masks(
    train_mask: NDArray[np.bool_],
    validation_mask: NDArray[np.bool_],
    holdout_mask: NDArray[np.bool_],
) -> None:
    if not train_mask.any() or not validation_mask.any() or not holdout_mask.any():
        raise ValueError("chronological_sequence_split_empty")


def _compute_class_weights(train_actions: IntArray) -> Float64Array:
    class_counts = np.bincount(train_actions, minlength=3).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("teacher_train_split_missing_action_class")
    class_weights = np.minimum(np.sqrt(len(train_actions) / (3.0 * class_counts)), 8.0)
    return class_weights / class_weights.mean()


def _prepare_training_data(
    frame: pd.DataFrame, teacher: pd.DataFrame, window: int
) -> dict[str, Any]:
    actions = teacher["teacher_action"].to_numpy(dtype=np.int64)
    features, feature_names = build_feature_matrix(frame, teacher)
    row_count = len(frame)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    _validate_split_boundaries(row_count, window, train_end, validation_end)

    scaler_mean = features[:train_end].mean(axis=0, dtype=np.float64)
    scaler_scale = features[:train_end].std(axis=0, dtype=np.float64)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    normalized = ((features - scaler_mean) / scaler_scale).astype(np.float32)
    sequences, sequence_actions, target_indices = make_sequences(
        normalized, actions, window
    )
    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    holdout_mask = target_indices >= validation_end
    _validate_split_masks(train_mask, validation_mask, holdout_mask)
    train_actions = sequence_actions[train_mask]
    class_weights = _compute_class_weights(train_actions)
    return {
        "features": features,
        "feature_names": feature_names,
        "actions": actions,
        "row_count": row_count,
        "train_end": train_end,
        "validation_end": validation_end,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "sequences": sequences,
        "sequence_actions": sequence_actions,
        "target_indices": target_indices,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "holdout_mask": holdout_mask,
        "train_actions": train_actions,
        "class_weights": class_weights,
    }


def _create_training_state(torch: Any, prepared: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(20261027)
    torch.set_num_threads(1)
    sequences = prepared["sequences"]
    train_mask = prepared["train_mask"]
    validation_mask = prepared["validation_mask"]
    holdout_mask = prepared["holdout_mask"]
    train_features = torch.tensor(sequences[train_mask], dtype=torch.float32)
    train_targets = torch.tensor(prepared["train_actions"], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_targets),
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )
    validation_features = sequences[validation_mask]
    validation_targets = prepared["sequence_actions"][validation_mask]
    holdout_features = sequences[holdout_mask]
    holdout_targets = prepared["sequence_actions"][holdout_mask]
    feature_count = int(prepared["features"].shape[1])
    hidden_size = 64
    lstm = torch.nn.LSTM(
        input_size=feature_count,
        hidden_size=hidden_size,
        num_layers=2,
        batch_first=True,
        dropout=0.15,
    )
    classifier = torch.nn.Linear(hidden_size, 3)
    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(classifier.parameters()), lr=0.001
    )
    loss_weights = torch.tensor(prepared["class_weights"], dtype=torch.float32)
    return {
        "lstm": lstm,
        "classifier": classifier,
        "optimizer": optimizer,
        "loss_weights": loss_weights,
        "train_loader": train_loader,
        "validation_features": validation_features,
        "validation_targets": validation_targets,
        "holdout_features": holdout_features,
        "holdout_targets": holdout_targets,
        "hidden_size": hidden_size,
        "feature_count": feature_count,
    }


def _snapshot_model_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _fit_lstm(
    torch: Any,
    state: dict[str, Any],
    epochs: int,
) -> dict[str, Any]:
    lstm = state["lstm"]
    classifier = state["classifier"]
    optimizer = state["optimizer"]
    loss_weights = state["loss_weights"]
    train_loader = state["train_loader"]
    validation_features = state["validation_features"]
    validation_targets = state["validation_targets"]
    best_validation_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_lstm_state: dict[str, Any] = {}
    best_classifier_state: dict[str, Any] = {}
    completed_epochs = 0

    for epoch in range(epochs):
        lstm.train()
        classifier.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = lstm(batch_features)
            logits = classifier(outputs[:, -1, :])
            loss = torch.nn.functional.cross_entropy(
                logits, batch_targets, weight=loss_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lstm.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        completed_epochs = epoch + 1
        predictions = _predict_actions(torch, lstm, classifier, validation_features)
        metrics = classification_metrics(validation_targets, predictions)
        score = float(metrics["macro_f1"])
        if score > best_validation_macro_f1 + 1e-9:
            best_validation_macro_f1 = score
            best_epoch = completed_epochs
            best_lstm_state = _snapshot_model_state(lstm)
            best_classifier_state = _snapshot_model_state(classifier)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break

    if not best_lstm_state or not best_classifier_state:
        raise RuntimeError("no_validation_checkpoint_was_selected")
    lstm.load_state_dict(best_lstm_state)
    classifier.load_state_dict(best_classifier_state)
    state["best_validation_macro_f1"] = best_validation_macro_f1
    state["best_epoch"] = best_epoch
    state["completed_epochs"] = completed_epochs
    return state


def _evaluate_training_state(
    torch: Any,
    frame: pd.DataFrame,
    prepared: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    validation_targets = state["validation_targets"]
    holdout_targets = state["holdout_targets"]
    validation_predictions = _predict_actions(
        torch, state["lstm"], state["classifier"], state["validation_features"]
    )
    holdout_predictions = _predict_actions(
        torch, state["lstm"], state["classifier"], state["holdout_features"]
    )
    close_prices = frame["close"].to_numpy(dtype=np.float64)
    target_indices = prepared["target_indices"]
    validation_close = close_prices[target_indices[prepared["validation_mask"]]]
    holdout_close = close_prices[target_indices[prepared["holdout_mask"]]]
    return {
        "validation_predictions": validation_predictions,
        "holdout_predictions": holdout_predictions,
        "validation_classification": classification_metrics(
            validation_targets, validation_predictions
        ),
        "holdout_classification": classification_metrics(
            holdout_targets, holdout_predictions
        ),
        "validation_teacher_economics": simulate_economics(
            validation_targets, validation_close
        ),
        "validation_model_economics": simulate_economics(
            validation_predictions, validation_close
        ),
        "holdout_teacher_economics": simulate_economics(holdout_targets, holdout_close),
        "holdout_model_economics": simulate_economics(
            holdout_predictions, holdout_close
        ),
        "validation_model_actions": action_distribution(validation_predictions),
        "holdout_teacher_actions": action_distribution(holdout_targets),
        "holdout_model_actions": action_distribution(holdout_predictions),
    }


def _save_model_artifacts(
    torch: Any,
    output_dir: Path,
    commit_sha: str,
    window: int,
    prepared: dict[str, Any],
    state: dict[str, Any],
) -> str:
    model_path = output_dir / "lstm-behavioral-cloning-state.pt"
    torch.save(
        {
            "architecture": "two-layer LSTM plus linear 3-class head",
            "lstm_state_dict": state["lstm"].state_dict(),
            "classifier_state_dict": state["classifier"].state_dict(),
            "input_features": prepared["feature_names"],
            "window": window,
            "hidden_size": state["hidden_size"],
            "action_map": _ACTION_NAMES,
            "initialized_from_scratch": True,
            "training_commit_sha": commit_sha,
        },
        model_path,
    )
    scaler_document = {
        "feature_names": prepared["feature_names"],
        "mean": [float(value) for value in prepared["scaler_mean"]],
        "scale": [float(value) for value in prepared["scaler_scale"]],
        "fit_rows": int(prepared["train_end"]),
        "fit_policy": "training_split_only",
    }
    (output_dir / "feature-scaler.json").write_text(
        json.dumps(scaler_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_path.name


def _build_training_report(
    commit_sha: str,
    source_manifest: dict[str, Any],
    teacher_manifest: dict[str, Any],
    teacher: pd.DataFrame,
    prepared: dict[str, Any],
    state: dict[str, Any],
    evaluation: dict[str, Any],
    model_artifact: str,
) -> dict[str, Any]:
    row_count = int(prepared["row_count"])
    train_end = int(prepared["train_end"])
    validation_end = int(prepared["validation_end"])
    holdout_distribution = evaluation["holdout_model_actions"]
    return {
        "schema_version": "alix.imitation-learning-report.v1",
        "commit_sha": commit_sha,
        "dataset": {
            "provider": source_manifest["provider"],
            "source_csv_sha256": source_manifest["csv_sha256"],
            "teacher_csv_sha256": teacher_manifest["teacher_csv_sha256"],
            "row_count": row_count,
            "train_rows": train_end,
            "validation_rows": validation_end - train_end,
            "holdout_rows": row_count - validation_end,
            "train_end_timestamp": str(teacher.iloc[train_end - 1]["timestamp"]),
            "validation_end_timestamp": str(
                teacher.iloc[validation_end - 1]["timestamp"]
            ),
            "holdout_start_timestamp": str(teacher.iloc[validation_end]["timestamp"]),
        },
        "teacher": {
            "kind": "causal_ohlcv_heuristic",
            "higher_timeframe_mode": "4h+1D_confluence",
            "ground_truth_claim": False,
            "action_distribution": action_distribution(prepared["actions"]),
        },
        "model": {
            "architecture": "two-layer CPU LSTM plus linear head",
            "initialized_from_scratch": True,
            "window_candles": int(prepared["sequences"].shape[1]),
            "input_feature_count": int(state["feature_count"]),
            "input_features": prepared["feature_names"],
            "requested_epochs": int(state["requested_epochs"]),
            "completed_epochs": int(state["completed_epochs"]),
            "best_validation_epoch": int(state["best_epoch"]),
            "best_validation_macro_f1": float(state["best_validation_macro_f1"]),
            "state_artifact": model_artifact,
        },
        "validation": {
            "classification": evaluation["validation_classification"],
            "teacher_economics": evaluation["validation_teacher_economics"],
            "model_economics": evaluation["validation_model_economics"],
            "model_action_distribution": evaluation["validation_model_actions"],
        },
        "holdout": {
            "classification_vs_teacher": evaluation["holdout_classification"],
            "teacher_economics": evaluation["holdout_teacher_economics"],
            "model_economics": evaluation["holdout_model_economics"],
            "teacher_action_distribution": evaluation["holdout_teacher_actions"],
            "model_action_distribution": holdout_distribution,
            "hold_collapse": holdout_distribution["HOLD"]["fraction"] >= 0.95,
        },
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "real_order_attempts": 0,
            "exchange_calls": 0,
            "private_api_calls": 0,
            "execution_api_implemented": False,
        },
        "decision": "RESEARCH_ONLY_NOT_PROMOTED",
    }


def run_training(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    commit_sha: str,
    epochs: int = 20,
    window: int = 48,
) -> dict[str, Any]:
    _validate_training_parameters(epochs, window)
    frame, source_manifest = load_public_market_data(
        dataset_path, manifest_path, commit_sha
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher, teacher_manifest = _write_teacher_dataset(
        frame, source_manifest, output_dir, commit_sha
    )
    prepared = _prepare_training_data(frame, teacher, window)
    torch: Any = importlib.import_module("torch")
    state = _create_training_state(torch, prepared)
    state["requested_epochs"] = epochs
    state = _fit_lstm(torch, state, epochs)
    evaluation = _evaluate_training_state(torch, frame, prepared, state)
    model_artifact = _save_model_artifacts(
        torch, output_dir, commit_sha, window, prepared, state
    )
    report = _build_training_report(
        commit_sha,
        source_manifest,
        teacher_manifest,
        teacher,
        prepared,
        state,
        evaluation,
        model_artifact,
    )
    report_path = output_dir / "imitation-learning-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "teacher_dataset_rows": int(prepared["row_count"]),
                "teacher_action_distribution": teacher_manifest[
                    "teacher_action_distribution"
                ],
                "validation": report["validation"],
                "holdout": report["holdout"],
                "safety": report["safety"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fresh LSTM by offline behavioral cloning."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window", type=int, default=48)
    args = parser.parse_args()
    run_training(
        args.dataset,
        args.manifest,
        args.output_dir,
        commit_sha=str(args.commit_sha),
        epochs=int(args.epochs),
        window=int(args.window),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
