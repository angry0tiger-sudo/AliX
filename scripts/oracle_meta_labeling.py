"""Research-only out-of-sample profitability filter for Triple-Barrier LSTM signals."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scripts.oracle_teacher_lstm import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    _ACTION_NAMES,
    build_causal_features,
    generate_oracle_labels,
)
from scripts.run_imitation_learning_agent import (
    _compute_class_weights,
    _weighted_cross_entropy_loss,
    action_distribution,
    classification_metrics,
    load_public_market_data,
    make_sequences,
)

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]
_HORIZON = 24
_TAKE_PROFIT = 0.03
_STOP_LOSS = 0.015
_WINDOW = 48
_EPOCHS = 20
_SEED = 20261028
_FEE_BPS = 25.0
_SLIPPAGE_BPS = 20.0
_FEE_BPS_PER_SIDE = _FEE_BPS
_SLIPPAGE_BPS_PER_SIDE = _SLIPPAGE_BPS
_ROUND_TRIP_COST = 2.0 * (_FEE_BPS + _SLIPPAGE_BPS) / 10_000.0
_MINIMUM_PF = 1.5
_MAXIMUM_DD = 0.15
_MINIMUM_OOS_SIGNALS = 800
_THRESHOLDS = tuple(round(float(v), 2) for v in np.arange(0.30, 0.801, 0.05))
_META_FEATURE_NAMES = (
    "lstm_probability_buy",
    "lstm_probability_sell",
    "lstm_probability_hold",
    "atr_fraction_14",
    "sma_distance_6",
    "sma_distance_24",
    "sma_distance_72",
    "sma_distance_168",
)
_PARTITIONS = ("base_train", "base_validation", "meta_train", "validation", "holdout")


def _chronological_boundaries(row_count: int) -> dict[str, int]:
    boundaries = {
        "base_train_end": int(row_count * 0.50),
        "base_validation_end": int(row_count * 0.55),
        "meta_train_end": int(row_count * 0.70),
        "validation_end": int(row_count * 0.85),
        "holdout_end": row_count,
    }
    previous = 0
    for value in boundaries.values():
        if value - previous <= _WINDOW + _HORIZON:
            raise ValueError("meta_labeling_temporal_partition_too_short")
        previous = value
    return boundaries


def _partition_masks(
    target_indices: IntArray,
    label_valid: NDArray[np.bool_],
    boundaries: dict[str, int],
    horizon: int = _HORIZON,
) -> dict[str, NDArray[np.bool_]]:
    if len(target_indices) != len(label_valid):
        raise ValueError("meta_partition_alignment_invalid")
    starts = {
        "base_train": 0,
        "base_validation": boundaries["base_train_end"],
        "meta_train": boundaries["base_validation_end"],
        "validation": boundaries["meta_train_end"],
        "holdout": boundaries["validation_end"],
    }
    ends = {
        "base_train": boundaries["base_train_end"],
        "base_validation": boundaries["base_validation_end"],
        "meta_train": boundaries["meta_train_end"],
        "validation": boundaries["validation_end"],
        "holdout": boundaries["holdout_end"],
    }
    return {
        name: (
            label_valid
            & (target_indices >= starts[name])
            & (target_indices < ends[name])
            & (target_indices + horizon < ends[name])
        )
        for name in _PARTITIONS
    }


def _standardize_from_training(
    features: Float32Array, train_end: int
) -> tuple[Float32Array, FloatArray, FloatArray]:
    mean = features[:train_end].mean(axis=0, dtype=np.float64)
    scale = features[:train_end].std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = ((features.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("meta_lstm_normalized_features_non_finite")
    return normalized, mean, scale


def _predict_probabilities(
    torch: Any,
    lstm: Any,
    classifier: Any,
    sequences: Float32Array,
    batch_size: int = 512,
) -> Float32Array:
    if len(sequences) == 0:
        raise ValueError("meta_lstm_prediction_split_empty")
    lstm.eval()
    classifier.eval()
    chunks: list[Float32Array] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.as_tensor(
                sequences[start : start + batch_size], dtype=torch.float32
            )
            outputs, _ = lstm(batch)
            logits = classifier(outputs[:, -1, :])
            values = logits.softmax(dim=1).cpu().numpy().astype(np.float32)
            chunks.append(cast(Float32Array, values))
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(sequences), 3) or not np.isfinite(result).all():
        raise ValueError("meta_lstm_probability_shape_or_finiteness_invalid")
    return result


def _train_base_epoch(
    torch: Any,
    lstm: Any,
    classifier: Any,
    loader: Any,
    optimizer: Any,
    loss_weights: Any,
) -> None:
    lstm.train()
    classifier.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs, _ = lstm(batch_x)
        loss = _weighted_cross_entropy_loss(
            torch, classifier(outputs[:, -1, :]), batch_y, loss_weights
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(lstm.parameters()) + list(classifier.parameters()), max_norm=1.0
        )
        optimizer.step()


def _base_validation_score(
    torch: Any,
    lstm: Any,
    classifier: Any,
    x_validation: Float32Array,
    y_validation: IntArray,
) -> float:
    probabilities = _predict_probabilities(torch, lstm, classifier, x_validation)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    return float(classification_metrics(y_validation, predictions)["macro_f1"])


def _capture_base_checkpoint(
    lstm: Any, classifier: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    lstm_state = {
        key: value.detach().clone() for key, value in lstm.state_dict().items()
    }
    classifier_state = {
        key: value.detach().clone() for key, value in classifier.state_dict().items()
    }
    return lstm_state, classifier_state


def _fit_base_lstm(
    torch: Any,
    sequences: Float32Array,
    labels: IntArray,
    train_mask: NDArray[np.bool_],
    validation_mask: NDArray[np.bool_],
    *,
    feature_count: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("meta_lstm_train_or_validation_partition_empty")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x_train = torch.as_tensor(sequences[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(labels[train_mask], dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )
    lstm = torch.nn.LSTM(
        input_size=feature_count,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        dropout=0.15,
    )
    classifier = torch.nn.Linear(64, 3)
    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(classifier.parameters()), lr=0.001
    )
    class_weights = _compute_class_weights(labels[train_mask])
    loss_weights = torch.as_tensor(class_weights, dtype=torch.float32)
    x_validation = sequences[validation_mask]
    y_validation = labels[validation_mask]
    best_score = -1.0
    best_epoch = 0
    completed_epochs = 0
    stale_epochs = 0
    best_lstm: dict[str, Any] = {}
    best_classifier: dict[str, Any] = {}

    for epoch in range(epochs):
        _train_base_epoch(torch, lstm, classifier, loader, optimizer, loss_weights)
        completed_epochs = epoch + 1
        score = _base_validation_score(
            torch, lstm, classifier, x_validation, y_validation
        )
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = completed_epochs
            best_lstm, best_classifier = _capture_base_checkpoint(lstm, classifier)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break
    if not best_lstm or not best_classifier:
        raise RuntimeError("meta_lstm_validation_checkpoint_missing")
    lstm.load_state_dict(best_lstm)
    classifier.load_state_dict(best_classifier)
    return {
        "lstm": lstm,
        "classifier": classifier,
        "class_weights": [float(value) for value in class_weights],
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "best_validation_macro_f1": float(best_score),
        "hidden_size": 64,
        "feature_count": feature_count,
    }


def _meta_feature_matrix(
    probabilities: Float32Array,
    raw_features: Float32Array,
    feature_names: list[str],
) -> tuple[Float32Array, list[str]]:
    market_names = _META_FEATURE_NAMES[3:]
    indexes = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in market_names if name not in indexes]
    if missing:
        raise ValueError(f"meta_required_market_features_missing:{','.join(missing)}")
    market_features = raw_features[:, [indexes[name] for name in market_names]]
    result = np.column_stack(
        [probabilities[:, 1], probabilities[:, 2], probabilities[:, 0], market_features]
    ).astype(np.float32, copy=False)
    if result.shape[1] != len(_META_FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError("meta_feature_matrix_shape_or_finiteness_invalid")
    return result, list(_META_FEATURE_NAMES)


def _profitability_targets(
    frame: pd.DataFrame,
    oracle_labels: pd.DataFrame,
    *,
    horizon: int = _HORIZON,
    take_profit: float = _TAKE_PROFIT,
    stop_loss: float = _STOP_LOSS,
) -> tuple[IntArray, FloatArray]:
    outcomes = oracle_labels["oracle_action"].to_numpy(dtype=np.int64)
    entries = oracle_labels["entry_price"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    valid = oracle_labels["label_valid"].to_numpy(dtype=np.bool_)
    valid &= np.arange(len(frame), dtype=np.int64) + horizon < len(frame)
    targets = np.zeros(len(frame), dtype=np.int64)
    returns = np.full(len(frame), np.nan, dtype=np.float64)
    for index in np.flatnonzero(valid):
        if outcomes[index] == ACTION_BUY:
            gross_return = take_profit
        elif outcomes[index] == ACTION_SELL:
            gross_return = -stop_loss
        else:
            gross_return = float(closes[index + horizon] / entries[index] - 1.0)
        returns[index] = gross_return - _ROUND_TRIP_COST
        targets[index] = int(returns[index] > 0.0)
    return targets, returns


def _fit_meta_model(
    features: Float32Array,
    target_labels: IntArray,
    *,
    seed: int,
) -> Any:
    if len(features) < 20 or len(np.unique(target_labels)) != 2:
        raise ValueError("meta_train_requires_at_least_20_buy_signals_and_both_classes")
    if len(features) != len(target_labels):
        raise ValueError("meta_train_target_alignment_invalid")
    sklearn_ensemble: Any = importlib.import_module("sklearn.ensemble")
    model = sklearn_ensemble.RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, target_labels)
    if 1 not in model.classes_:
        raise RuntimeError("meta_classifier_missing_positive_profitability_class")
    return model


def _predict_profit_probability(model: Any, features: Float32Array) -> FloatArray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float64)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features)[:, positive_column].astype(np.float64)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("meta_probability_out_of_bounds")
    return cast(FloatArray, probabilities)


def _binary_metrics(truth: IntArray, predictions: IntArray) -> dict[str, Any]:
    if len(truth) != len(predictions) or len(truth) == 0:
        return {"sample_count": int(len(truth)), "macro_f1": None}
    tp = int(np.sum((truth == 1) & (predictions == 1)))
    fp = int(np.sum((truth == 0) & (predictions == 1)))
    fn = int(np.sum((truth == 1) & (predictions == 0)))
    tn = int(np.sum((truth == 0) & (predictions == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1_positive = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    neg_precision = tn / max(1, tn + fn)
    neg_recall = tn / max(1, tn + fp)
    f1_negative = (
        2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        if neg_precision + neg_recall
        else 0.0
    )
    return {
        "sample_count": len(truth),
        "accuracy": float((tp + tn) / len(truth)),
        "precision_profitable": float(precision),
        "recall_profitable": float(recall),
        "macro_f1": float((f1_positive + f1_negative) / 2.0),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _open_position(index: int, price: float, equity: float) -> dict[str, float | int]:
    fill = price * (1.0 + _SLIPPAGE_BPS_PER_SIDE / 10_000.0)
    quantity = equity * (1.0 - _FEE_BPS_PER_SIDE / 10_000.0) / fill
    return {
        "entry_index": index,
        "entry_price": fill,
        "entry_equity": equity,
        "quantity": quantity,
    }


def _gap_exit(
    price: float, position: dict[str, float | int]
) -> tuple[str, float] | None:
    entry = float(position["entry_price"])
    if price <= entry * (1.0 - _STOP_LOSS):
        return "GAP_STOP", price
    if price >= entry * (1.0 + _TAKE_PROFIT):
        return "GAP_TAKE_PROFIT", price
    return None


def _intrabar_exit(
    high: float, low: float, position: dict[str, float | int]
) -> tuple[str, float] | None:
    entry = float(position["entry_price"])
    stop_price = entry * (1.0 - _STOP_LOSS)
    target_price = entry * (1.0 + _TAKE_PROFIT)
    stop_hit = low <= stop_price
    target_hit = high >= target_price
    if stop_hit:
        return ("SAME_CANDLE_STOP_FIRST" if target_hit else "STOP_LOSS", stop_price)
    if target_hit:
        return "TAKE_PROFIT", target_price
    return None


def _close_position(
    position: dict[str, float | int], price: float, reason: str, index: int
) -> tuple[float, dict[str, Any]]:
    fill = price * (1.0 - _SLIPPAGE_BPS_PER_SIDE / 10_000.0)
    proceeds = float(position["quantity"]) * fill * (1.0 - _FEE_BPS_PER_SIDE / 10_000.0)
    start_equity = float(position["entry_equity"])
    pnl = proceeds - start_equity
    return proceeds, {
        "entry_index": int(position["entry_index"]),
        "exit_index": int(index),
        "exit_reason": reason,
        "net_pnl_fraction": float(pnl / start_equity) if start_equity > 0 else 0.0,
        "net_pnl_equity_units": float(pnl),
    }


def _mark_to_market(
    cash: float, position: dict[str, float | int] | None, close: float
) -> float:
    if position is None:
        return cash
    liquidation_price = close * (1.0 - _SLIPPAGE_BPS_PER_SIDE / 10_000.0)
    return (
        float(position["quantity"])
        * liquidation_price
        * (1.0 - _FEE_BPS_PER_SIDE / 10_000.0)
    )


def _validate_simulation_inputs(
    frame: pd.DataFrame,
    actions: IntArray,
    profit_probabilities: FloatArray,
    threshold: float,
    start: int,
    end: int,
) -> None:
    if len(actions) != len(frame) or len(profit_probabilities) != len(frame):
        raise ValueError("meta_economic_simulation_configuration_invalid")
    if not 0 <= start < end <= len(frame):
        raise ValueError("meta_economic_simulation_configuration_invalid")
    if not 0 <= threshold <= 1:
        raise ValueError("meta_economic_simulation_configuration_invalid")


def _settle_pending_exit(
    cash: float,
    position: dict[str, float | int] | None,
    pending_exit: bool,
    index: int,
    opens: FloatArray,
    trades: list[dict[str, Any]],
) -> tuple[float, dict[str, float | int] | None, bool]:
    if position is None or not pending_exit:
        return cash, position, pending_exit
    cash, trade = _close_position(
        position, float(opens[index]), "LSTM_SELL_EXIT", index
    )
    trades.append(trade)
    return cash, None, False


def _settle_gap_exit(
    cash: float,
    position: dict[str, float | int] | None,
    index: int,
    opens: FloatArray,
    trades: list[dict[str, Any]],
) -> tuple[float, dict[str, float | int] | None]:
    if position is None:
        return cash, None
    gap = _gap_exit(float(opens[index]), position)
    if gap is None:
        return cash, position
    reason, price = gap
    cash, trade = _close_position(position, price, reason, index)
    trades.append(trade)
    return cash, None


def _activate_pending_entry(
    position: dict[str, float | int] | None,
    pending_entry: bool,
    index: int,
    opens: FloatArray,
    cash: float,
) -> tuple[dict[str, float | int] | None, bool]:
    if position is None and pending_entry:
        return _open_position(index, float(opens[index]), cash), False
    return position, False


def _manage_open_position(
    cash: float,
    position: dict[str, float | int] | None,
    index: int,
    end: int,
    horizon: int,
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    actions: IntArray,
    trades: list[dict[str, Any]],
) -> tuple[float, dict[str, float | int] | None, bool]:
    if position is None:
        return cash, None, False
    barrier = _intrabar_exit(float(highs[index]), float(lows[index]), position)
    if barrier is not None:
        reason, price = barrier
        cash, trade = _close_position(position, price, reason, index)
        trades.append(trade)
        return cash, None, False
    if index - int(position["entry_index"]) >= horizon - 1:
        cash, trade = _close_position(
            position, float(closes[index]), "TIME_BARRIER", index
        )
        trades.append(trade)
        return cash, None, False
    pending_exit = actions[index] == ACTION_SELL and index + 1 < end
    return cash, position, bool(pending_exit)


def _buy_signal_approved(
    index: int,
    end: int,
    position: dict[str, float | int] | None,
    actions: IntArray,
    probabilities: FloatArray,
    threshold: float,
) -> bool:
    return (
        position is None
        and index + 1 < end
        and actions[index] == ACTION_BUY
        and probabilities[index] >= threshold
    )


def _force_close_partition(
    cash: float,
    position: dict[str, float | int] | None,
    end: int,
    closes: FloatArray,
    curve: list[float],
    trades: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    if position is None:
        return cash, curve
    cash, trade = _close_position(
        position, float(closes[end - 1]), "FORCED_PARTITION_CLOSE", end - 1
    )
    trades.append(trade)
    curve[-1] = cash
    return cash, curve


def _summarize_simulation(
    cash: float,
    curve: list[float],
    trades: list[dict[str, Any]],
    horizon: int,
    *,
    include_equity_curve: bool = False,
) -> dict[str, Any]:
    equity_curve = np.asarray(curve, dtype=np.float64)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = np.divide(
        peaks - equity_curve, peaks, out=np.zeros_like(equity_curve), where=peaks > 0
    )
    pnls = np.asarray(
        [trade["net_pnl_equity_units"] for trade in trades], dtype=np.float64
    )
    gross_profit = float(pnls[pnls > 0].sum()) if np.any(pnls > 0) else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if np.any(pnls < 0) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    summary: dict[str, Any] = {
        "net_return_fraction": float(cash - 1.0),
        "max_drawdown_fraction": float(drawdowns.max(initial=0.0)),
        "profit_factor": profit_factor,
        "profit_factor_status": (
            "defined" if profit_factor is not None else "undefined_without_losses"
        ),
        "closed_trade_count": len(trades),
        "winning_trade_count": int(np.count_nonzero(pnls > 0)),
        "losing_trade_count": int(np.count_nonzero(pnls < 0)),
        "win_rate": float(np.mean(pnls > 0)) if len(pnls) else 0.0,
        "gross_profit_equity_units": gross_profit,
        "gross_loss_equity_units": gross_loss,
        "fee_bps_per_side": _FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": _SLIPPAGE_BPS_PER_SIDE,
        "take_profit_fraction": _TAKE_PROFIT,
        "stop_loss_fraction": _STOP_LOSS,
        "maximum_holding_candles": horizon,
        "exit_policy": "SELL_SIGNAL_OR_TRIPLE_BARRIER_OR_HORIZON",
        "market_mode": "SPOT_LONG_FLAT_NO_SHORTS",
        "trades": trades,
    }
    if include_equity_curve:
        summary["equity_curve"] = equity_curve.tolist()
    return summary


def _simulate_long_flat(
    frame: pd.DataFrame,
    actions: IntArray,
    profit_probabilities: FloatArray,
    *,
    threshold: float,
    start: int,
    end: int,
    horizon: int = _HORIZON,
    include_equity_curve: bool = False,
) -> dict[str, Any]:
    """Simulate gated Spot entries with stop/target/horizon and model SELL exits."""
    _validate_simulation_inputs(
        frame, actions, profit_probabilities, threshold, start, end
    )
    opens = frame["open"].to_numpy(dtype=np.float64)
    highs = frame["high"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    cash = 1.0
    position: dict[str, float | int] | None = None
    pending_entry = False
    pending_exit = False
    trades: list[dict[str, Any]] = []
    curve: list[float] = [1.0]

    for index in range(start, end):
        cash, position, pending_exit = _settle_pending_exit(
            cash, position, pending_exit, index, opens, trades
        )
        cash, position = _settle_gap_exit(cash, position, index, opens, trades)
        position, pending_entry = _activate_pending_entry(
            position, pending_entry, index, opens, cash
        )
        cash, position, pending_exit = _manage_open_position(
            cash, position, index, end, horizon, highs, lows, closes, actions, trades
        )
        pending_entry = _buy_signal_approved(
            index, end, position, actions, profit_probabilities, threshold
        )
        curve.append(_mark_to_market(cash, position, float(closes[index])))

    cash, curve = _force_close_partition(cash, position, end, closes, curve, trades)
    return _summarize_simulation(
        cash, curve, trades, horizon, include_equity_curve=include_equity_curve
    )


def _choose_threshold(
    frame: pd.DataFrame,
    actions: IntArray,
    profit_probabilities: FloatArray,
    *,
    start: int,
    end: int,
) -> tuple[float, list[dict[str, Any]], str]:
    candidates: list[dict[str, Any]] = []
    eligible: list[tuple[float, float, float, float]] = []
    for threshold in _THRESHOLDS:
        metrics = _simulate_long_flat(
            frame,
            actions,
            profit_probabilities,
            threshold=threshold,
            start=start,
            end=end,
        )
        candidates.append({"threshold": threshold, **metrics})
        if metrics["closed_trade_count"] >= 5:
            pf = metrics["profit_factor"]
            capped_pf = min(float(pf), 5.0) if pf is not None else 0.0
            excess_dd = max(0.0, float(metrics["max_drawdown_fraction"]) - _MAXIMUM_DD)
            score = capped_pf + float(metrics["net_return_fraction"]) - 5.0 * excess_dd
            eligible.append(
                (
                    score,
                    float(metrics["net_return_fraction"]),
                    -float(metrics["max_drawdown_fraction"]),
                    threshold,
                )
            )
    if not eligible:
        return 0.50, candidates, "no_threshold_reached_5_closed_trades;default_0.50"
    return (
        float(max(eligible)[3]),
        candidates,
        "maximize_capped_validation_PF_plus_net_return_minus_5x_excess_drawdown;min_5_trades",
    )


def _partition_classification_report(
    truth: IntArray, base: IntArray, gated: IntArray
) -> dict[str, Any]:
    if len(truth) == 0:
        raise ValueError("meta_classification_partition_empty")
    return {
        "sample_count": int(len(truth)),
        "oracle_action_distribution": action_distribution(truth),
        "base_lstm_action_distribution": action_distribution(base),
        "meta_filtered_action_distribution": action_distribution(gated),
        "base_lstm_classification": classification_metrics(truth, base),
        "meta_filtered_classification": classification_metrics(truth, gated),
    }


def _binary_report_for_buy_positions(
    positions: NDArray[np.int64],
    base_actions: IntArray,
    meta_probabilities: FloatArray,
    profit_targets: IntArray,
    threshold: float,
) -> dict[str, Any]:
    buy_positions = positions[base_actions[positions] == ACTION_BUY]
    if len(buy_positions) == 0:
        return {"sample_count": 0, "macro_f1": None}
    truth = profit_targets[buy_positions]
    pred = (meta_probabilities[buy_positions] >= threshold).astype(np.int64)
    return {
        **_binary_metrics(truth, pred),
        "actual_profitability_distribution": {
            "profitable": int(np.count_nonzero(truth == 1)),
            "unprofitable": int(np.count_nonzero(truth == 0)),
        },
        "meta_accepted_buy_count": int(np.count_nonzero(pred == 1)),
        "meta_rejected_buy_count": int(np.count_nonzero(pred == 0)),
    }


def _validate_safety(report: dict[str, Any]) -> None:
    expected = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "private_api_calls": 0,
    }
    if any(report["safety"].get(key) != value for key, value in expected.items()):
        raise RuntimeError("meta_labeling_safety_invariant_breached")
    if report["decision"] != "RESEARCH_ONLY_NOT_PROMOTED":
        raise RuntimeError("meta_labeling_research_promotion_invariant_breached")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _return_moments(returns: FloatArray) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("meta_overfit_returns_invalid")
    count = int(values.size)
    if count < 3:
        return {
            "sample_count": count,
            "mean_return": float(np.mean(values)) if count else None,
            "sample_std": None,
            "observed_sharpe": None,
            "skewness": None,
            "kurtosis_pearson": None,
            "status": "insufficient_track_record",
        }
    sample_std = float(np.std(values, ddof=1))
    if sample_std <= 1e-12:
        return {
            "sample_count": count,
            "mean_return": float(np.mean(values)),
            "sample_std": sample_std,
            "observed_sharpe": None,
            "skewness": None,
            "kurtosis_pearson": None,
            "status": "zero_variance_track_record",
        }
    mean_return = float(np.mean(values))
    population_std = float(np.std(values, ddof=0))
    centered = values - mean_return
    standardized = centered / population_std
    return {
        "sample_count": count,
        "mean_return": mean_return,
        "sample_std": sample_std,
        "observed_sharpe": mean_return / sample_std,
        "skewness": float(np.mean(standardized**3)),
        "kurtosis_pearson": float(np.mean(standardized**4)),
        "status": "estimated",
    }


def _deflated_sharpe_ratio(
    trade_returns: FloatArray, trial_count: int, confidence_level: float = 0.95
) -> dict[str, Any]:
    if trial_count < 1 or not 0.5 < confidence_level < 1.0:
        raise ValueError("meta_dsr_configuration_invalid")
    moments = _return_moments(trade_returns)
    result: dict[str, Any] = {
        "probability": None,
        "candidate_threshold_trials": trial_count,
        "confidence_level_for_reference": confidence_level,
        "formula": "Deflated Sharpe Ratio with expected maximum Sharpe across tested thresholds",
        "return_frequency": "closed_trade; nonannualized",
        **moments,
    }
    observed_sharpe = moments["observed_sharpe"]
    if observed_sharpe is None:
        result["status"] = str(moments["status"])
        result["expected_max_sharpe_under_null"] = None
        result["sharpe_standard_error"] = None
        return result
    skewness = float(moments["skewness"])
    kurtosis = float(moments["kurtosis_pearson"])
    variance_factor = (
        1.0
        - skewness * float(observed_sharpe)
        + ((kurtosis - 1.0) / 4.0) * float(observed_sharpe) ** 2
    )
    if variance_factor <= 0.0:
        result["probability"] = None
        result["status"] = "invalid_sharpe_sampling_variance"
        result["expected_max_sharpe_under_null"] = None
        result["sharpe_standard_error"] = None
        return result
    standard_error = math.sqrt(variance_factor / (int(moments["sample_count"]) - 1))
    normal = NormalDist()
    if trial_count == 1:
        expected_max_sharpe = 0.0
    else:
        euler_mascheroni = 0.5772156649015329
        expected_max_sharpe = standard_error * (
            (1.0 - euler_mascheroni) * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + euler_mascheroni * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        )
    z_score = (float(observed_sharpe) - expected_max_sharpe) / standard_error
    result["probability"] = float(normal.cdf(z_score))
    result["expected_max_sharpe_under_null"] = float(expected_max_sharpe)
    result["sharpe_standard_error"] = float(standard_error)
    result["status"] = "estimated"
    return result


def _minimum_track_record_length(
    trade_returns: FloatArray, confidence_level: float = 0.95
) -> dict[str, Any]:
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("meta_mintrl_confidence_invalid")
    moments = _return_moments(trade_returns)
    result: dict[str, Any] = {
        "minimum_track_record_length_trades": None,
        "confidence_level": confidence_level,
        "reference_sharpe": 0.0,
        "formula": "1 + (1 - skew*SR + ((kurtosis_pearson - 1)/4)*SR^2) * (z_confidence/SR)^2",
        **moments,
    }
    observed_sharpe = moments["observed_sharpe"]
    if observed_sharpe is None:
        result["status"] = str(moments["status"])
        return result
    if float(observed_sharpe) <= 0.0:
        result["status"] = "no_finite_track_length_for_nonpositive_observed_sharpe"
        return result
    skewness = float(moments["skewness"])
    kurtosis = float(moments["kurtosis_pearson"])
    variance_factor = (
        1.0
        - skewness * float(observed_sharpe)
        + ((kurtosis - 1.0) / 4.0) * float(observed_sharpe) ** 2
    )
    if variance_factor <= 0.0:
        result["status"] = "invalid_sharpe_sampling_variance"
        return result
    z_confidence = NormalDist().inv_cdf(confidence_level)
    minimum_length = (
        1.0 + variance_factor * (z_confidence / float(observed_sharpe)) ** 2
    )
    result["minimum_track_record_length_trades"] = int(math.ceil(minimum_length))
    result["additional_trades_required"] = max(
        0,
        int(result["minimum_track_record_length_trades"])
        - int(moments["sample_count"]),
    )
    result["status"] = "estimated"
    return result


def _sharpe_ratio(returns: FloatArray) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("meta_pbo_return_window_invalid")
    deviation = float(np.std(values, ddof=1))
    if deviation <= 1e-12:
        return 0.0
    return float(np.mean(values) / deviation)


def _rank_percentile(value: float, candidates: list[float]) -> float:
    if not candidates:
        raise ValueError("meta_pbo_rank_population_empty")
    less = sum(
        1
        for candidate in candidates
        if candidate < value
        and not math.isclose(candidate, value, rel_tol=1e-12, abs_tol=1e-12)
    )
    equal = sum(
        1
        for candidate in candidates
        if math.isclose(candidate, value, rel_tol=1e-12, abs_tol=1e-12)
    )
    return float((less + 0.5 * equal) / len(candidates))


def _prepare_pbo_blocks(
    threshold_return_streams: dict[float, FloatArray],
    block_count: int,
    purge_candles: int,
) -> tuple[list[float], dict[float, list[FloatArray]]]:
    thresholds = sorted(threshold_return_streams)
    if (
        len(thresholds) < 2
        or block_count < 4
        or block_count % 2 != 0
        or purge_candles < 0
    ):
        raise ValueError("meta_pbo_configuration_invalid")
    lengths = {len(threshold_return_streams[threshold]) for threshold in thresholds}
    if len(lengths) != 1:
        raise ValueError("meta_pbo_return_stream_lengths_mismatch")
    stream_length = next(iter(lengths))
    if stream_length < block_count * (2 * purge_candles + 2):
        raise ValueError("meta_pbo_track_record_too_short_for_purge")
    blocks_by_threshold = {
        threshold: _purged_return_blocks(
            threshold_return_streams[threshold], block_count, purge_candles
        )
        for threshold in thresholds
    }
    return thresholds, blocks_by_threshold


def _purged_return_blocks(
    returns: FloatArray, block_count: int, purge_candles: int
) -> list[FloatArray]:
    stream = np.asarray(returns, dtype=np.float64)
    if stream.ndim != 1 or not np.isfinite(stream).all():
        raise ValueError("meta_pbo_returns_non_finite")
    blocks: list[FloatArray] = []
    for block in np.array_split(stream, block_count):
        if len(block) <= 2 * purge_candles + 1:
            raise ValueError("meta_pbo_block_too_short_after_purge")
        trimmed = block[purge_candles : len(block) - purge_candles]
        blocks.append(np.asarray(trimmed, dtype=np.float64))
    return blocks


def _score_threshold_blocks(
    blocks: list[FloatArray], selected_blocks: tuple[int, ...]
) -> float:
    combined = np.concatenate([blocks[index] for index in selected_blocks])
    return _sharpe_ratio(combined)


def _evaluate_pbo_split(
    thresholds: list[float],
    blocks_by_threshold: dict[float, list[FloatArray]],
    in_sample_blocks: tuple[int, ...],
    out_sample_blocks: tuple[int, ...],
) -> dict[str, Any]:
    in_scores = {
        threshold: _score_threshold_blocks(
            blocks_by_threshold[threshold], in_sample_blocks
        )
        for threshold in thresholds
    }
    out_scores = {
        threshold: _score_threshold_blocks(
            blocks_by_threshold[threshold], out_sample_blocks
        )
        for threshold in thresholds
    }
    best_in_sample_score = max(in_scores.values())
    winners = [
        threshold
        for threshold in thresholds
        if math.isclose(
            in_scores[threshold],
            best_in_sample_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ]
    winning_oos_ranks = [
        _rank_percentile(out_scores[threshold], list(out_scores.values()))
        for threshold in winners
    ]
    selected_oos_rank = float(np.mean(winning_oos_ranks))
    return {
        "in_sample_blocks": list(in_sample_blocks),
        "out_of_sample_blocks": list(out_sample_blocks),
        "in_sample_winner_thresholds": winners,
        "selected_oos_rank_percentile": selected_oos_rank,
        "overfit": selected_oos_rank < 0.5,
    }


def _probability_backtest_overfitting(
    threshold_return_streams: dict[float, FloatArray],
    *,
    block_count: int = 6,
    purge_candles: int = _HORIZON,
) -> dict[str, Any]:
    thresholds, blocks_by_threshold = _prepare_pbo_blocks(
        threshold_return_streams, block_count, purge_candles
    )
    split_results: list[dict[str, Any]] = []
    block_ids = tuple(range(block_count))
    for in_sample_blocks in itertools.combinations(block_ids, block_count // 2):
        in_sample_set = set(in_sample_blocks)
        out_sample_blocks = tuple(
            index for index in block_ids if index not in in_sample_set
        )
        split_results.append(
            _evaluate_pbo_split(
                thresholds,
                blocks_by_threshold,
                in_sample_blocks,
                out_sample_blocks,
            )
        )
    overfit_count = sum(bool(result["overfit"]) for result in split_results)
    return {
        "probability": float(overfit_count / len(split_results)),
        "cscv_split_count": len(split_results),
        "block_count": block_count,
        "purge_candles_each_block_edge": purge_candles,
        "candidate_thresholds": thresholds,
        "performance_metric": "nonannualized Sharpe ratio of per-candle marked-to-market equity returns",
        "method": "Combinatorially Symmetric Cross-Validation on existing Validation OOS forecasts",
        "definition": "fraction of splits where the in-sample winner has below-median out-of-sample rank",
        "oos_rank_percentiles": [
            float(result["selected_oos_rank_percentile"]) for result in split_results
        ],
        "splits": split_results,
        "status": "estimated",
    }


def _load_diagnostic_artifact(
    artifact_dir: Path, expected_source_sha: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    report_path = artifact_dir / "oracle-meta-labeling-report.json"
    manifest_path = artifact_dir / "oracle-meta-labeling-manifest.json"
    market_path = artifact_dir / "public-market-data.csv"
    predictions_path = artifact_dir / "meta-out-of-sample-predictions.csv"
    required_paths = (report_path, manifest_path, market_path, predictions_path)
    if any(not path.is_file() for path in required_paths):
        raise ValueError("meta_overfit_required_artifact_missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        report.get("commit_sha") != expected_source_sha
        or manifest.get("commit_sha") != expected_source_sha
    ):
        raise ValueError("meta_overfit_source_commit_sha_mismatch")
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "private_api_calls": 0,
    }
    _validate_artifact_safety(report, manifest, safety)
    _validate_artifact_market_hash(report, manifest, market_path)
    frame = pd.read_csv(market_path)
    predictions = pd.read_csv(predictions_path)
    expected_rows = int(manifest.get("row_count", -1))
    reported_rows = int(report.get("dataset", {}).get("row_count", -1))
    if len(frame) != expected_rows or reported_rows != len(frame):
        raise ValueError("meta_overfit_market_data_row_count_mismatch")
    return report, frame, predictions


def _validate_artifact_safety(
    report: dict[str, Any], manifest: dict[str, Any], expected_safety: dict[str, Any]
) -> None:
    for name, payload in (("report", report), ("manifest", manifest)):
        safety = payload.get("safety", {})
        if any(safety.get(key) != value for key, value in expected_safety.items()):
            raise ValueError(f"meta_overfit_{name}_safety_invariant_breached")


def _validate_artifact_market_hash(
    report: dict[str, Any], manifest: dict[str, Any], market_path: Path
) -> None:
    actual_csv_sha = hashlib.sha256(market_path.read_bytes()).hexdigest()
    expected_csv_sha = str(manifest.get("source_csv_sha256", ""))
    report_csv_sha = str(report.get("dataset", {}).get("csv_sha256", ""))
    if (
        not expected_csv_sha
        or actual_csv_sha != expected_csv_sha
        or actual_csv_sha != report_csv_sha
    ):
        raise ValueError("meta_overfit_market_data_hash_mismatch")


def _reconstruct_oos_prediction_arrays(
    frame: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[IntArray, FloatArray, IntArray]:
    needed_columns = {
        "row_index",
        "base_lstm_action",
        "meta_profit_probability",
        "partition",
    }
    if not needed_columns.issubset(predictions.columns):
        raise ValueError("meta_overfit_prediction_schema_invalid")
    row_indices = predictions["row_index"].to_numpy(dtype=np.int64)
    if (
        len(row_indices) == 0
        or predictions["row_index"].duplicated().any()
        or np.any(row_indices < 0)
        or np.any(row_indices >= len(frame))
    ):
        raise ValueError("meta_overfit_prediction_row_alignment_invalid")
    action_lookup = {"BUY": ACTION_BUY, "HOLD": ACTION_HOLD, "SELL": ACTION_SELL}
    mapped_actions = predictions["base_lstm_action"].map(action_lookup)
    if mapped_actions.isna().any():
        raise ValueError("meta_overfit_prediction_action_invalid")
    actions = np.full(len(frame), ACTION_HOLD, dtype=np.int64)
    actions[row_indices] = mapped_actions.to_numpy(dtype=np.int64)
    probabilities = np.zeros(len(frame), dtype=np.float64)
    probability_values = predictions["meta_profit_probability"].to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(probability_values).all()
        or np.any(probability_values < 0.0)
        or np.any(probability_values > 1.0)
    ):
        raise ValueError("meta_overfit_prediction_probability_invalid")
    probabilities[row_indices] = probability_values
    partitions = predictions["partition"].astype(str).to_numpy()
    validation_indices = np.sort(row_indices[partitions == "validation"])
    if len(validation_indices) == 0 or np.any(np.diff(validation_indices) != 1):
        raise ValueError("meta_overfit_validation_oos_partition_invalid")
    return actions, probabilities, validation_indices


def _thresholds_from_report(report: dict[str, Any]) -> list[float]:
    thresholds_raw = report.get("threshold_selection", {}).get(
        "candidate_thresholds", []
    )
    thresholds = sorted({float(threshold) for threshold in thresholds_raw})
    if len(thresholds) < 2 or any(
        not 0.0 <= threshold <= 1.0 for threshold in thresholds
    ):
        raise ValueError("meta_overfit_threshold_candidates_invalid")
    return thresholds


def _trade_returns_from_report(report: dict[str, Any]) -> FloatArray:
    validation_metrics = report.get("validation", {}).get(
        "meta_filtered_economic_metrics_after_costs", {}
    )
    trades = validation_metrics.get("trades", [])
    return np.asarray(
        [float(trade["net_pnl_fraction"]) for trade in trades], dtype=np.float64
    )


def _validation_threshold_return_streams(
    frame: pd.DataFrame,
    actions: IntArray,
    probabilities: FloatArray,
    thresholds: list[float],
    validation_indices: IntArray,
    horizon: int,
) -> dict[float, FloatArray]:
    validation_start = int(validation_indices[0])
    validation_end = int(validation_indices[-1]) + 1
    if len(validation_indices) < 6 * (2 * horizon + 2):
        raise ValueError("meta_overfit_validation_oos_partition_invalid")
    streams: dict[float, FloatArray] = {}
    for threshold in thresholds:
        simulated = _simulate_long_flat(
            frame,
            actions,
            probabilities,
            threshold=threshold,
            start=validation_start,
            end=validation_end,
            horizon=horizon,
            include_equity_curve=True,
        )
        curve = np.asarray(simulated["equity_curve"], dtype=np.float64)
        if curve.size != validation_end - validation_start + 1 or np.any(
            curve[:-1] <= 0.0
        ):
            raise ValueError("meta_overfit_validation_equity_curve_invalid")
        streams[threshold] = np.asarray(curve[1:] / curve[:-1] - 1.0, dtype=np.float64)
    return streams


def _build_overfit_diagnostics(
    trade_returns: FloatArray,
    threshold_return_streams: dict[float, FloatArray],
    source_sha: str,
    code_sha: str,
) -> dict[str, Any]:
    dsr = _deflated_sharpe_ratio(trade_returns, len(threshold_return_streams))
    mintrl = _minimum_track_record_length(trade_returns)
    pbo = _probability_backtest_overfitting(
        threshold_return_streams, block_count=6, purge_candles=_HORIZON
    )
    dsr_probability = dsr.get("probability")
    pbo_probability = pbo.get("probability")
    dsr_failed = dsr_probability is None or float(dsr_probability) < 0.5
    pbo_failed = pbo_probability is None or float(pbo_probability) > 0.5
    return {
        "status": "OVERFIT_WARNING"
        if dsr_failed or pbo_failed
        else "NO_OVERFIT_WARNING",
        "source_artifact_commit_sha": source_sha,
        "diagnostics_code_commit_sha": code_sha,
        "evaluation_scope": "existing Validation OOS predictions only; no model refit; Holdout excluded",
        "threshold_selection_performed": False,
        "candidate_threshold_count": len(threshold_return_streams),
        "validation_closed_trade_count": len(trade_returns),
        "dsr_below_0_5": dsr_failed,
        "pbo_above_0_5": pbo_failed,
        "validation_results_trustworthy": not (dsr_failed or pbo_failed),
        "dsr": dsr,
        "pbo": pbo,
        "minimum_track_record_length": mintrl,
        "blocking_reason": (
            "DSR below 0.5 or not estimable"
            if dsr_failed
            else "PBO above 0.5 or not estimable"
            if pbo_failed
            else None
        ),
    }


def _run_existing_oos_diagnostics(
    artifact_dir: Path, expected_source_sha: str, diagnostics_code_sha: str
) -> dict[str, Any]:
    report, frame, predictions = _load_diagnostic_artifact(
        artifact_dir, expected_source_sha
    )
    actions, probabilities, validation_indices = _reconstruct_oos_prediction_arrays(
        frame, predictions
    )
    thresholds = _thresholds_from_report(report)
    trade_returns = _trade_returns_from_report(report)
    horizon = int(
        report.get("economic_simulation", {}).get("maximum_holding_candles", _HORIZON)
    )
    threshold_return_streams = _validation_threshold_return_streams(
        frame,
        actions,
        probabilities,
        thresholds,
        validation_indices,
        horizon,
    )
    diagnostics = _build_overfit_diagnostics(
        trade_returns,
        threshold_return_streams,
        expected_source_sha,
        diagnostics_code_sha,
    )
    report["overfit_diagnostics"] = diagnostics
    _validate_safety(report)
    _write_json(artifact_dir / "oracle-meta-labeling-report.json", report)
    _write_json(artifact_dir / "oracle-meta-labeling-diagnostics.json", diagnostics)
    return diagnostics


def _validate_meta_training_configuration(
    epochs: int, window: int, horizon: int, take_profit: float, stop_loss: float
) -> None:
    if not 1 <= epochs <= 200:
        raise ValueError("meta_labeling_training_configuration_out_of_range")
    if not 8 <= window <= 512:
        raise ValueError("meta_labeling_training_configuration_out_of_range")
    if not 1 <= horizon <= 240:
        raise ValueError("meta_labeling_training_configuration_out_of_range")
    if not 0 < take_profit < 0.5:
        raise ValueError("meta_labeling_barrier_fraction_out_of_range")
    if not 0 < stop_loss < 0.5:
        raise ValueError("meta_labeling_barrier_fraction_out_of_range")


def _profit_factor_passed(value: Any) -> bool:
    return value is not None and float(value) > _MINIMUM_PF


def _drawdown_passed(metrics: dict[str, Any]) -> bool:
    return float(metrics["max_drawdown_fraction"]) <= _MAXIMUM_DD


def _build_economic_gate(
    validation: dict[str, Any],
    holdout: dict[str, Any],
    validation_actions: IntArray,
    holdout_actions: IntArray,
) -> dict[str, Any]:
    validation_economics = validation
    holdout_economics = holdout
    validation_pf = validation_economics["profit_factor"]
    holdout_pf = holdout_economics["profit_factor"]
    signal_count = int(
        np.count_nonzero(validation_actions == ACTION_BUY)
        + np.count_nonzero(validation_actions == ACTION_SELL)
        + np.count_nonzero(holdout_actions == ACTION_BUY)
        + np.count_nonzero(holdout_actions == ACTION_SELL)
    )
    validation_pf_passed = _profit_factor_passed(validation_pf)
    holdout_pf_passed = _profit_factor_passed(holdout_pf)
    validation_dd_passed = _drawdown_passed(validation_economics)
    holdout_dd_passed = _drawdown_passed(holdout_economics)
    signal_count_passed = signal_count > _MINIMUM_OOS_SIGNALS
    return {
        "passed": all(
            [
                validation_pf_passed,
                holdout_pf_passed,
                validation_dd_passed,
                holdout_dd_passed,
                signal_count_passed,
            ]
        ),
        "minimum_profit_factor": _MINIMUM_PF,
        "profit_factor_comparison": "strictly_greater_than_minimum",
        "maximum_allowed_drawdown_fraction": _MAXIMUM_DD,
        "minimum_oos_signal_count": _MINIMUM_OOS_SIGNALS,
        "oos_signal_count": signal_count,
        "validation_profit_factor_passed": validation_pf_passed,
        "holdout_profit_factor_passed": holdout_pf_passed,
        "validation_drawdown_passed": validation_dd_passed,
        "holdout_drawdown_passed": holdout_dd_passed,
        "signal_count_passed": signal_count_passed,
    }


def _validate_required_partition_masks(masks: dict[str, NDArray[np.bool_]]) -> None:
    if any(not masks[name].any() for name in _PARTITIONS):
        raise ValueError("meta_labeling_required_partition_has_no_valid_labels")


def _write_meta_csv_outputs(output_dir: Path, context: dict[str, Any]) -> None:
    frame = context["frame"]
    labels = context["labels"]
    targets_by_row = context["targets_by_row"]
    returns_by_row = context["returns_by_row"]
    meta_train_features = context["meta_train_features"]
    meta_feature_names = context["meta_feature_names"]
    meta_train_targets = context["meta_train_targets"]
    target_indices = context["target_indices"]
    meta_train_positions = context["meta_train_positions"]
    probabilities = context["probabilities"]
    base_actions = context["base_actions"]
    meta_probabilities = context["meta_probabilities"]
    raw_gated_actions = context["raw_gated_actions"]
    masks = context["masks"]
    sequence_returns = context["sequence_returns"]
    targets = context["targets"]

    labels_out = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].astype(str),
            "oracle_action": labels["oracle_action"].to_numpy(dtype=np.int64),
            "oracle_action_name": [
                _ACTION_NAMES[int(v)] for v in labels["oracle_action"]
            ],
            "barrier_outcome": labels["barrier_outcome"].astype(str),
            "entry_price": labels["entry_price"].to_numpy(dtype=np.float64),
            "label_valid": labels["label_valid"].to_numpy(dtype=bool),
            "profitable_after_costs": targets_by_row,
            "estimated_forward_net_return_fraction": returns_by_row,
        }
    )
    labels_out.to_csv(output_dir / "oracle-meta-label-targets.csv", index=False)
    training_csv = pd.DataFrame(meta_train_features, columns=meta_feature_names)
    training_csv["profitable_after_costs"] = meta_train_targets
    training_csv["source_row"] = target_indices[meta_train_positions]
    training_csv.to_csv(output_dir / "meta-training-dataset.csv", index=False)

    sequence_report = pd.DataFrame(
        {
            "row_index": target_indices,
            "timestamp": frame["timestamp"].iloc[target_indices].astype(str).to_numpy(),
            "base_lstm_probability_buy": probabilities[:, 1],
            "base_lstm_probability_sell": probabilities[:, 2],
            "base_lstm_probability_hold": probabilities[:, 0],
            "base_lstm_action": [_ACTION_NAMES[int(v)] for v in base_actions],
            "meta_profit_probability": meta_probabilities,
            "meta_filtered_action": [_ACTION_NAMES[int(v)] for v in raw_gated_actions],
            "partition": [
                next(
                    (name for name in _PARTITIONS if masks[name][pos]),
                    "purged_or_warmup",
                )
                for pos in range(len(target_indices))
            ],
            "profitable_after_costs_target": targets,
            "forward_net_return_fraction": sequence_returns,
        }
    )
    sequence_report.to_csv(
        output_dir / "meta-out-of-sample-predictions.csv", index=False
    )


def _write_meta_model_artifacts(output_dir: Path, context: dict[str, Any]) -> Path:
    torch: Any = context["torch"]
    base = context["base"]
    feature_names = context["feature_names"]
    window = context["window"]
    commit_sha = context["commit_sha"]
    model = context["meta_model"]
    scaler_mean = context["scaler_mean"]
    scaler_scale = context["scaler_scale"]
    boundaries = context["boundaries"]
    source_manifest = context["source_manifest"]
    count = context["count"]
    horizon = context["horizon"]
    take_profit = context["take_profit"]
    stop_loss = context["stop_loss"]

    model_path = output_dir / "oracle-meta-lstm-state.pt"
    torch.save(
        {
            "architecture": "two-layer LSTM plus linear 3-class head",
            "lstm_state_dict": base["lstm"].state_dict(),
            "classifier_state_dict": base["classifier"].state_dict(),
            "input_features": feature_names,
            "window_candles": int(window),
            "hidden_size": int(base["hidden_size"]),
            "action_map": _ACTION_NAMES,
            "trained_only_before_meta_partition": True,
            "training_commit_sha": commit_sha,
        },
        model_path,
    )
    meta_path = output_dir / "meta-profitability-random-forest.joblib"
    joblib_module: Any = importlib.import_module("joblib")
    joblib_module.dump(model, meta_path)
    _write_json(
        output_dir / "meta-feature-scaler.json",
        {
            "feature_names": feature_names,
            "mean": [float(v) for v in scaler_mean],
            "scale": [float(v) for v in scaler_scale],
            "fit_rows": int(boundaries["base_train_end"]),
            "fit_policy": "base_lstm_training_partition_only",
        },
    )
    safety_manifest = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
        "exchange_calls": 0,
        "private_api_calls": 0,
    }
    manifest = {
        "schema_version": "alix.oracle-meta-labeling-manifest.v1",
        "commit_sha": commit_sha,
        "source_provider": source_manifest["provider"],
        "source_csv_sha256": source_manifest["csv_sha256"],
        "row_count": int(count),
        "barrier_horizon_candles": int(horizon),
        "take_profit_fraction": float(take_profit),
        "stop_loss_fraction": float(stop_loss),
        "round_trip_fee_and_slippage_fraction": _ROUND_TRIP_COST,
        "base_lstm_never_trained_on_meta_rows": True,
        "meta_model_trained_only_on_oos_base_predictions": True,
        "validation_used_for_threshold_selection": True,
        "holdout_used_for_training_or_threshold_selection": False,
        "safety": safety_manifest,
    }
    _write_json(output_dir / "oracle-meta-labeling-manifest.json", manifest)
    return meta_path


def _build_meta_report(context: dict[str, Any]) -> dict[str, Any]:
    commit_sha = context["commit_sha"]
    source_manifest = context["source_manifest"]
    count = context["count"]
    boundaries = context["boundaries"]
    horizon = context["horizon"]
    take_profit = context["take_profit"]
    stop_loss = context["stop_loss"]
    targets_by_row = context["targets_by_row"]
    labels = context["labels"]
    masks = context["masks"]
    feature_names = context["feature_names"]
    window = context["window"]
    base = context["base"]
    epochs = context["epochs"]
    val_truth = context["val_truth"]
    val_base = context["val_base"]
    hold_truth = context["hold_truth"]
    hold_base = context["hold_base"]
    meta_train_positions = context["meta_train_positions"]
    meta_train_targets = context["meta_train_targets"]
    meta_feature_names = context["meta_feature_names"]
    val_meta_report = context["val_meta_report"]
    hold_meta_report = context["hold_meta_report"]
    threshold = context["threshold"]
    threshold_candidates = context["threshold_candidates"]
    threshold_reason = context["threshold_reason"]
    validation_economics = context["validation_economics"]
    holdout_economics = context["holdout_economics"]
    val_actions_report = context["val_actions_report"]
    hold_actions_report = context["hold_actions_report"]
    row_actions = context["row_actions"]
    row_meta_probabilities = context["row_meta_probabilities"]
    economic_gate = context["economic_gate"]
    frame = context["frame"]
    meta_path = context["meta_path"]

    report = {
        "schema_version": "alix.oracle-meta-labeling-report.v1",
        "commit_sha": commit_sha,
        "dataset": {
            "provider": source_manifest["provider"],
            "csv_sha256": source_manifest["csv_sha256"],
            "row_count": int(count),
            "interval": source_manifest.get("interval", "1h"),
        },
        "partitioning": {
            "base_train": [0, boundaries["base_train_end"]],
            "base_validation": [
                boundaries["base_train_end"],
                boundaries["base_validation_end"],
            ],
            "meta_train": [
                boundaries["base_validation_end"],
                boundaries["meta_train_end"],
            ],
            "validation": [boundaries["meta_train_end"], boundaries["validation_end"]],
            "holdout": [boundaries["validation_end"], count],
            "percentages": {
                "base_train": 50,
                "base_validation": 5,
                "meta_train": 15,
                "validation": 15,
                "holdout": 15,
            },
            "label_horizon_purge_candles": int(horizon),
            "cross_partition_labels_excluded": True,
        },
        "oracle": {
            "method": "Triple-Barrier",
            "horizon_candles": int(horizon),
            "take_profit_fraction": float(take_profit),
            "stop_loss_fraction": float(stop_loss),
            "same_candle_double_hit_policy": "stop_loss_first_conservative",
            "label_target": "positive_forward_return_after_round_trip_costs",
            "round_trip_fee_and_slippage_fraction": _ROUND_TRIP_COST,
            "profitable_label_distribution_valid_only": {
                "profitable": int(
                    np.count_nonzero(
                        targets_by_row[labels["label_valid"].to_numpy(dtype=bool)] == 1
                    )
                ),
                "unprofitable": int(
                    np.count_nonzero(
                        targets_by_row[labels["label_valid"].to_numpy(dtype=bool)] == 0
                    )
                ),
            },
        },
        "base_lstm": {
            "architecture": "two-layer CPU LSTM plus linear 3-class head",
            "trained_only_on_base_train_partition": True,
            "training_rows": int(np.count_nonzero(masks["base_train"])),
            "early_stopping_rows": int(np.count_nonzero(masks["base_validation"])),
            "meta_train_rows_not_seen_by_base_lstm": int(
                np.count_nonzero(masks["meta_train"])
            ),
            "input_features": feature_names,
            "window_candles": int(window),
            "hidden_size": int(base["hidden_size"]),
            "requested_epochs": int(epochs),
            "completed_epochs": int(base["completed_epochs"]),
            "best_epoch": int(base["best_epoch"]),
            "best_base_validation_macro_f1": float(base["best_validation_macro_f1"]),
            "class_weights": base["class_weights"],
            "validation": _partition_classification_report(
                val_truth, val_base, val_base
            ),
            "holdout": _partition_classification_report(
                hold_truth, hold_base, hold_base
            ),
        },
        "meta_model": {
            "algorithm": "RandomForestClassifier",
            "estimators": 250,
            "max_depth": 5,
            "min_samples_leaf": 8,
            "class_weight": "balanced_subsample",
            "features": meta_feature_names,
            "training_population": "out-of-sample base LSTM BUY signals in meta_train only",
            "training_signal_count": int(len(meta_train_positions)),
            "training_positive_count": int(np.count_nonzero(meta_train_targets == 1)),
            "training_negative_count": int(np.count_nonzero(meta_train_targets == 0)),
            "validation_binary_metrics": val_meta_report,
            "holdout_binary_metrics": hold_meta_report,
            "model_artifact": meta_path.name,
        },
        "threshold_selection": {
            "candidate_thresholds": list(_THRESHOLDS),
            "selected_threshold": float(threshold),
            "selected_on": "validation_economics_only",
            "selection_reason": threshold_reason,
            "validation_candidate_economics": threshold_candidates,
        },
        "validation": {
            "row_start_inclusive": boundaries["meta_train_end"],
            "row_end_exclusive": boundaries["validation_end"],
            "base_vs_oracle": val_actions_report,
            "meta_profitability": val_meta_report,
            "unfiltered_economic_metrics_after_costs": _simulate_long_flat(
                frame,
                row_actions,
                row_meta_probabilities,
                threshold=0.0,
                start=boundaries["meta_train_end"],
                end=boundaries["validation_end"],
                horizon=horizon,
            ),
            "meta_filtered_economic_metrics_after_costs": validation_economics,
            "macro_f1_after_action_filter": val_actions_report[
                "meta_filtered_classification"
            ]["macro_f1"],
        },
        "holdout": {
            "row_start_inclusive": boundaries["validation_end"],
            "row_end_exclusive": count,
            "base_vs_oracle": hold_actions_report,
            "meta_profitability": hold_meta_report,
            "unfiltered_economic_metrics_after_costs": _simulate_long_flat(
                frame,
                row_actions,
                row_meta_probabilities,
                threshold=0.0,
                start=boundaries["validation_end"],
                end=count,
                horizon=horizon,
            ),
            "meta_filtered_economic_metrics_after_costs": holdout_economics,
            "macro_f1_after_action_filter": hold_actions_report[
                "meta_filtered_classification"
            ]["macro_f1"],
        },
        "economic_gate": economic_gate,
        "economic_simulation": {
            "market_mode": "Spot long/flat; SELL only exits and never shorts",
            "signals_at_candle_close_execute_at_next_candle_open": True,
            "sell_exit_is_never_filtered": True,
            "take_profit_fraction": float(take_profit),
            "stop_loss_fraction": float(stop_loss),
            "maximum_holding_candles": int(horizon),
            "same_candle_stop_and_target_policy": "stop_first",
            "fee_bps_per_side": _FEE_BPS,
            "slippage_bps_per_side": _SLIPPAGE_BPS,
            "forced_partition_close": True,
            "equity_curve_marked_to_market": True,
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
    return report


def run_meta_labeling(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    commit_sha: str,
    epochs: int = _EPOCHS,
    window: int = _WINDOW,
    horizon: int = _HORIZON,
    take_profit: float = _TAKE_PROFIT,
    stop_loss: float = _STOP_LOSS,
    seed: int = _SEED,
) -> dict[str, Any]:
    _validate_meta_training_configuration(
        epochs, window, horizon, take_profit, stop_loss
    )
    frame, source_manifest = load_public_market_data(
        dataset_path, manifest_path, commit_sha
    )
    count = len(frame)
    boundaries = _chronological_boundaries(count)
    labels = generate_oracle_labels(
        frame,
        horizon=horizon,
        take_profit_fraction=take_profit,
        stop_loss_fraction=stop_loss,
    )
    raw_features, feature_names = build_causal_features(frame)
    normalized, scaler_mean, scaler_scale = _standardize_from_training(
        raw_features, boundaries["base_train_end"]
    )
    sequences, sequence_labels, target_indices = make_sequences(
        normalized, labels["oracle_action"].to_numpy(dtype=np.int64), window
    )
    valid = labels["label_valid"].to_numpy(dtype=np.bool_)[target_indices]
    masks = _partition_masks(target_indices, valid, boundaries, horizon)
    _validate_required_partition_masks(masks)

    torch: Any = importlib.import_module("torch")
    base = _fit_base_lstm(
        torch,
        sequences,
        sequence_labels.astype(np.int64, copy=False),
        masks["base_train"],
        masks["base_validation"],
        feature_count=int(raw_features.shape[1]),
        epochs=epochs,
        seed=seed,
    )
    probabilities = _predict_probabilities(
        torch, base["lstm"], base["classifier"], sequences
    )
    base_actions = probabilities.argmax(axis=1).astype(np.int64, copy=False)
    meta_features, meta_feature_names = _meta_feature_matrix(
        probabilities, raw_features[target_indices], feature_names
    )
    targets_by_row, returns_by_row = _profitability_targets(
        frame, labels, horizon=horizon, take_profit=take_profit, stop_loss=stop_loss
    )
    targets = targets_by_row[target_indices]
    sequence_returns = returns_by_row[target_indices]
    meta_train_positions = np.flatnonzero(
        masks["meta_train"] & (base_actions == ACTION_BUY)
    )
    meta_train_features = meta_features[meta_train_positions]
    meta_train_targets = targets[meta_train_positions]
    model = _fit_meta_model(meta_train_features, meta_train_targets, seed=seed)

    meta_probabilities = np.zeros(len(target_indices), dtype=np.float64)
    candidates = (base_actions == ACTION_BUY) & (
        masks["meta_train"] | masks["validation"] | masks["holdout"]
    )
    meta_probabilities[candidates] = _predict_profit_probability(
        model, meta_features[candidates]
    )
    row_actions = np.full(count, ACTION_HOLD, dtype=np.int64)
    row_meta_probabilities = np.zeros(count, dtype=np.float64)
    active_positions = masks["meta_train"] | masks["validation"] | masks["holdout"]
    active_rows = target_indices[active_positions]
    row_actions[active_rows] = base_actions[active_positions]
    row_meta_probabilities[active_rows] = meta_probabilities[active_positions]

    validation_positions = np.flatnonzero(masks["validation"])
    holdout_positions = np.flatnonzero(masks["holdout"])
    threshold, threshold_candidates, threshold_reason = _choose_threshold(
        frame,
        row_actions,
        row_meta_probabilities,
        start=boundaries["meta_train_end"],
        end=boundaries["validation_end"],
    )
    validation_economics = _simulate_long_flat(
        frame,
        row_actions,
        row_meta_probabilities,
        threshold=threshold,
        start=boundaries["meta_train_end"],
        end=boundaries["validation_end"],
        horizon=horizon,
    )
    holdout_economics = _simulate_long_flat(
        frame,
        row_actions,
        row_meta_probabilities,
        threshold=threshold,
        start=boundaries["validation_end"],
        end=count,
        horizon=horizon,
    )
    raw_gated_actions = base_actions.copy()
    reject_buys = (base_actions == ACTION_BUY) & (meta_probabilities < threshold)
    raw_gated_actions[reject_buys] = ACTION_HOLD
    val_base = base_actions[masks["validation"]]
    val_gated = raw_gated_actions[masks["validation"]]
    val_truth = sequence_labels[masks["validation"]].astype(np.int64)
    hold_base = base_actions[masks["holdout"]]
    hold_gated = raw_gated_actions[masks["holdout"]]
    hold_truth = sequence_labels[masks["holdout"]].astype(np.int64)
    val_actions_report = _partition_classification_report(
        val_truth, val_base, val_gated
    )
    hold_actions_report = _partition_classification_report(
        hold_truth, hold_base, hold_gated
    )
    val_meta_report = _binary_report_for_buy_positions(
        validation_positions, base_actions, meta_probabilities, targets, threshold
    )
    hold_meta_report = _binary_report_for_buy_positions(
        holdout_positions, base_actions, meta_probabilities, targets, threshold
    )

    economic_gate = _build_economic_gate(
        validation_economics,
        holdout_economics,
        val_gated,
        hold_gated,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    context: dict[str, Any] = {
        "commit_sha": commit_sha,
        "source_manifest": source_manifest,
        "count": count,
        "boundaries": boundaries,
        "horizon": horizon,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "frame": frame,
        "labels": labels,
        "targets_by_row": targets_by_row,
        "returns_by_row": returns_by_row,
        "target_indices": target_indices,
        "meta_train_positions": meta_train_positions,
        "meta_train_features": meta_train_features,
        "meta_train_targets": meta_train_targets,
        "meta_feature_names": meta_feature_names,
        "probabilities": probabilities,
        "base_actions": base_actions,
        "meta_probabilities": meta_probabilities,
        "raw_gated_actions": raw_gated_actions,
        "masks": masks,
        "sequence_returns": sequence_returns,
        "targets": targets,
        "torch": torch,
        "base": base,
        "feature_names": feature_names,
        "window": window,
        "epochs": epochs,
        "val_truth": val_truth,
        "val_base": val_base,
        "hold_truth": hold_truth,
        "hold_base": hold_base,
        "val_meta_report": val_meta_report,
        "hold_meta_report": hold_meta_report,
        "threshold": threshold,
        "threshold_candidates": threshold_candidates,
        "threshold_reason": threshold_reason,
        "validation_economics": validation_economics,
        "holdout_economics": holdout_economics,
        "val_actions_report": val_actions_report,
        "hold_actions_report": hold_actions_report,
        "row_actions": row_actions,
        "row_meta_probabilities": row_meta_probabilities,
        "economic_gate": economic_gate,
        "meta_model": model,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }
    _write_meta_csv_outputs(output_dir, context)
    context["meta_path"] = _write_meta_model_artifacts(output_dir, context)
    report = _build_meta_report(context)
    _validate_safety(report)
    _write_json(output_dir / "oracle-meta-labeling-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an OOS RandomForest filter for an oracle LSTM."
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--commit-sha")
    parser.add_argument("--epochs", type=int, default=_EPOCHS)
    parser.add_argument("--window", type=int, default=_WINDOW)
    parser.add_argument("--horizon", type=int, default=_HORIZON)
    parser.add_argument("--take-profit-fraction", type=float, default=_TAKE_PROFIT)
    parser.add_argument("--stop-loss-fraction", type=float, default=_STOP_LOSS)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--diagnostics-artifact-dir", type=Path)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--diagnostics-code-sha")
    args = parser.parse_args()
    if args.diagnostics_artifact_dir is not None:
        if not args.expected_source_sha or not args.diagnostics_code_sha:
            parser.error(
                "diagnostics mode requires --expected-source-sha and --diagnostics-code-sha"
            )
        result = _run_existing_oos_diagnostics(
            args.diagnostics_artifact_dir,
            expected_source_sha=str(args.expected_source_sha),
            diagnostics_code_sha=str(args.diagnostics_code_sha),
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if (
        args.dataset is None
        or args.manifest is None
        or args.output_dir is None
        or not args.commit_sha
    ):
        parser.error(
            "training mode requires --dataset, --manifest, --output-dir, and --commit-sha"
        )
    run_meta_labeling(
        args.dataset,
        args.manifest,
        args.output_dir,
        commit_sha=str(args.commit_sha),
        epochs=int(args.epochs),
        window=int(args.window),
        horizon=int(args.horizon),
        take_profit=float(args.take_profit_fraction),
        stop_loss=float(args.stop_loss_fraction),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
