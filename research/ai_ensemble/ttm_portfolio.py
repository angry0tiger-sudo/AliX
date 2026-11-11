"""Offline OOS entry-teacher replay with a TTM-only position-sizing overlay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from research.ai_ensemble.ensemble import StackingConfig, StackingEnsemble
from research.ai_ensemble.features import build_causal_features
from research.ai_ensemble.labels import build_net_profit_targets

PROBABILITY_THRESHOLD = 0.50
TRAINING_EMBARGO_BARS = 24
FIXED_POSITION_FRACTION = 0.10


def _aligned_teacher_arrays(ohlcv: DataFrame) -> dict[str, Any]:
    features = build_causal_features(ohlcv)
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    market_positions = np.flatnonzero(finite)
    if market_positions.size == 0:
        raise ValueError("ttm_portfolio_no_finite_teacher_features")
    feature_position = {
        int(market_index): model_index
        for model_index, market_index in enumerate(market_positions.tolist())
    }
    targets = build_net_profit_targets(ohlcv)
    labels = np.zeros(len(market_positions), dtype=int)
    label_ends = np.full(len(market_positions), -1, dtype=int)
    label_end_market = np.full(len(market_positions), -1, dtype=int)
    net_returns = np.zeros(len(market_positions), dtype=float)
    eligible: list[int] = []
    for model_index, raw_market_index in enumerate(market_positions.tolist()):
        target = targets.iloc[int(raw_market_index)]
        if (
            pd.isna(target["target"])
            or pd.isna(target["label_end_idx"])
            or pd.isna(target["net_return"])
        ):
            continue
        end_market_index = int(target["label_end_idx"])
        end_model_index = feature_position.get(end_market_index)
        if end_model_index is None or end_model_index < model_index:
            raise ValueError("ttm_portfolio_label_end_not_aligned")
        labels[model_index] = int(target["target"])
        label_ends[model_index] = end_model_index
        label_end_market[model_index] = end_market_index
        net_returns[model_index] = float(target["net_return"])
        eligible.append(model_index)
    return {
        "features": features.iloc[market_positions].reset_index(drop=True),
        "market_positions": market_positions,
        "labels": labels,
        "label_ends": label_ends,
        "label_end_market": label_end_market,
        "net_returns": net_returns,
        "eligible": eligible,
    }


def _indices_in_window(
    data: dict[str, Any],
    lower_market_index: int,
    upper_market_index: int,
    *,
    max_label_end_exclusive: int,
) -> list[int]:
    selected: list[int] = []
    positions = data["market_positions"]
    ends = data["label_end_market"]
    for model_index in data["eligible"]:
        market_index = int(positions[model_index])
        end_market_index = int(ends[model_index])
        if lower_market_index <= market_index < upper_market_index:
            if end_market_index < max_label_end_exclusive:
                selected.append(model_index)
    return selected


def _collect_split_trades(
    data: dict[str, Any],
    eligible_indices: list[int],
    probabilities: dict[int, float],
    split: str,
    split_end_market_index: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    blocked_until = -1
    positions = data["market_positions"]
    label_ends = data["label_ends"]
    label_end_market = data["label_end_market"]
    net_returns = data["net_returns"]
    for model_index in eligible_indices:
        if model_index <= blocked_until:
            continue
        probability = probabilities[model_index]
        if probability < PROBABILITY_THRESHOLD:
            continue
        end_model_index = int(label_ends[model_index])
        end_market_index = int(label_end_market[model_index])
        market_index = int(positions[model_index])
        if end_market_index >= split_end_market_index:
            continue
        if end_model_index < model_index or not np.isfinite(net_returns[model_index]):
            raise ValueError("ttm_portfolio_trade_window_invalid")
        events.append(
            {
                "split": split,
                "entry_market_index": market_index,
                "exit_market_index": end_market_index,
                "net_return": float(net_returns[model_index]),
                "entry_probability": probability,
            }
        )
        blocked_until = end_model_index
    return events


def build_oos_teacher_trade_events(
    ohlcv: DataFrame,
    *,
    training_end_market_index: int,
    validation_end_market_index: int,
) -> dict[str, Any]:
    """Fit one OHLCV-only teacher on 70% and freeze it for validation/holdout.

    Targets use the existing next-bar-open Triple-Barrier contract with 50 bps
    round-trip costs. Labels crossing a split boundary are excluded. The TTM
    forecast is not used by this function and cannot create entry signals.
    """
    if not 0 < training_end_market_index < validation_end_market_index < len(ohlcv):
        raise ValueError("ttm_portfolio_split_boundaries_invalid")
    data = _aligned_teacher_arrays(ohlcv)
    train_indices = _indices_in_window(
        data,
        0,
        training_end_market_index,
        max_label_end_exclusive=training_end_market_index - TRAINING_EMBARGO_BARS,
    )
    validation_indices = _indices_in_window(
        data,
        training_end_market_index,
        validation_end_market_index,
        max_label_end_exclusive=validation_end_market_index,
    )
    holdout_indices = _indices_in_window(
        data,
        validation_end_market_index,
        len(ohlcv),
        max_label_end_exclusive=len(ohlcv),
    )
    if len(train_indices) < 500 or not validation_indices or not holdout_indices:
        raise ValueError("ttm_portfolio_teacher_split_has_insufficient_samples")

    model = StackingEnsemble(
        StackingConfig(probability_threshold=PROBABILITY_THRESHOLD)
    )
    model.fit(
        data["features"],
        data["labels"].tolist(),
        data["label_ends"].tolist(),
        train_indices,
    )
    market_positions = data["market_positions"]
    oos_model_indices = np.flatnonzero(
        market_positions >= training_end_market_index
    ).tolist()
    oos_probabilities = model.predict_proba(data["features"].iloc[oos_model_indices])
    probability_by_index = {
        int(model_index): float(probability)
        for model_index, probability in zip(
            oos_model_indices, oos_probabilities.tolist()
        )
    }
    validation_trades = _collect_split_trades(
        data,
        validation_indices,
        probability_by_index,
        "validation",
        validation_end_market_index,
    )
    holdout_trades = _collect_split_trades(
        data,
        holdout_indices,
        probability_by_index,
        "holdout",
        len(ohlcv),
    )
    return {
        "training_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "holdout_rows": len(holdout_indices),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "round_trip_cost_bps": 50,
        "uses_chronos_features": False,
        "uses_ttm_for_entry_signals": False,
        "validation_trades": validation_trades,
        "holdout_trades": holdout_trades,
    }


def attach_asof_risk_multipliers(
    trades: list[dict[str, Any]],
    forecast_market_indices: np.ndarray,
    risk_multipliers: np.ndarray,
) -> list[dict[str, Any]]:
    """Attach the latest forecast available at or before each teacher entry."""
    origins = np.asarray(forecast_market_indices, dtype=np.int64)
    multipliers = np.asarray(risk_multipliers, dtype=np.float64)
    if origins.ndim != 1 or multipliers.shape != origins.shape or origins.size == 0:
        raise ValueError("ttm_portfolio_risk_forecast_alignment_invalid")
    if not np.all(np.diff(origins) > 0):
        raise ValueError("ttm_portfolio_risk_origins_not_strictly_increasing")
    if (
        not np.isfinite(multipliers).all()
        or np.any(multipliers <= 0.0)
        or np.any(multipliers > 1.0)
    ):
        raise ValueError("ttm_portfolio_risk_multiplier_out_of_range")
    attached: list[dict[str, Any]] = []
    for trade in trades:
        entry_index = int(trade["entry_market_index"])
        origin_index = int(np.searchsorted(origins, entry_index, side="right") - 1)
        if origin_index < 0:
            raise ValueError("ttm_portfolio_entry_has_no_prior_risk_forecast")
        attached.append({**trade, "risk_multiplier": float(multipliers[origin_index])})
    return attached


def evaluate_sized_portfolio(
    trades: list[dict[str, Any]],
    *,
    risk_managed: bool,
    fixed_position_fraction: float = FIXED_POSITION_FRACTION,
) -> dict[str, float | int | None]:
    """Compound non-overlapping trades at fixed or TTM-scaled equity exposure."""
    if (
        not np.isfinite(fixed_position_fraction)
        or not 0.0 < fixed_position_fraction <= 0.10
    ):
        raise ValueError("ttm_portfolio_fixed_position_fraction_out_of_range")
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": None,
            "net_profit_factor": None,
            "max_drawdown": None,
            "net_return": 0.0,
            "ending_equity": 1.0,
            "average_position_fraction": None,
            "average_risk_multiplier": None,
        }
    equity = 1.0
    peak_equity = 1.0
    maximum_drawdown = 0.0
    positive_pnl = 0.0
    negative_pnl = 0.0
    wins = 0
    exposures: list[float] = []
    multipliers: list[float] = []
    for trade in trades:
        net_return = float(trade["net_return"])
        multiplier = float(trade["risk_multiplier"]) if risk_managed else 1.0
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("ttm_portfolio_trade_return_invalid")
        if not np.isfinite(multiplier) or not 0.0 < multiplier <= 1.0:
            raise ValueError("ttm_portfolio_trade_multiplier_invalid")
        exposure = fixed_position_fraction * multiplier
        pnl = equity * exposure * net_return
        if pnl > 0.0:
            positive_pnl += pnl
            wins += 1
        elif pnl < 0.0:
            negative_pnl -= pnl
        equity += pnl
        if equity <= 0.0:
            raise ValueError("ttm_portfolio_equity_depleted")
        peak_equity = max(peak_equity, equity)
        maximum_drawdown = max(maximum_drawdown, 1.0 - equity / peak_equity)
        exposures.append(exposure)
        multipliers.append(multiplier)
    profit_factor = positive_pnl / negative_pnl if negative_pnl > 0.0 else None
    return {
        "trade_count": len(trades),
        "win_rate": float(wins / len(trades)),
        "net_profit_factor": float(profit_factor)
        if profit_factor is not None
        else None,
        "max_drawdown": float(maximum_drawdown),
        "net_return": float(equity - 1.0),
        "ending_equity": float(equity),
        "average_position_fraction": float(np.mean(exposures)),
        "average_risk_multiplier": float(np.mean(multipliers)),
    }
