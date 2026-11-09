"""Triple-barrier targets converted to realized profitability after 50 bps."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier

MAKER_FEE_BPS_PER_SIDE = 25
ROUND_TRIP_COST_BPS = 50
LABEL_HORIZON_CANDLES = 24
TAKE_PROFIT_ATR = 2.0
STOP_LOSS_ATR = 1.0


def build_net_profit_targets(
    candles: DataFrame,
    *,
    round_trip_cost_bps: int = ROUND_TRIP_COST_BPS,
) -> DataFrame:
    """Apply the existing 24-candle Triple Barrier contract and deduct fees.

    Entry is the next bar open. TP/SL exits use the existing barrier prices;
    horizon expiry is marked at the final close in the label window. A row is
    positive only when its realized return exceeds the full round-trip cost.
    """
    if round_trip_cost_bps != ROUND_TRIP_COST_BPS:
        raise ValueError("nobitex_round_trip_cost_must_be_50_bps")
    labels = apply_adaptive_triple_barrier(
        candles,
        horizon_candles=LABEL_HORIZON_CANDLES,
        take_profit_multiplier=TAKE_PROFIT_ATR,
        stop_loss_multiplier=STOP_LOSS_ATR,
        entry_reference="NEXT_BAR_OPEN",
        entry_slippage_bps=0.0,
    )
    cost_fraction = ROUND_TRIP_COST_BPS / 10_000.0
    rows: list[dict[str, object]] = []
    for index in range(len(labels)):
        record = labels.iloc[index]
        end_index = record["label_end_idx"]
        if pd.isna(end_index):
            rows.append(
                {
                    "target": np.nan,
                    "label_end_idx": np.nan,
                    "net_return": np.nan,
                    "gross_return": np.nan,
                    "exit_reason": record["exit_reason"],
                    "label": record["label"],
                }
            )
            continue
        final_index = int(end_index)
        entry_price = float(record["entry_price"])
        reason = str(record["exit_reason"])
        if reason == "TP_HIT":
            exit_price = float(record["take_profit_price"])
            if str(record["label"]) != "BUY":
                raise ValueError("triple_barrier_tp_label_mismatch")
        elif reason in {"SL_HIT", "COLLISION"}:
            exit_price = float(record["stop_loss_price"])
            if str(record["label"]) != "SELL":
                raise ValueError("triple_barrier_sl_label_mismatch")
        elif reason == "HORIZON_EXPIRY":
            exit_price = float(candles.iloc[final_index]["close"])
            if str(record["label"]) != "HOLD":
                raise ValueError("triple_barrier_horizon_label_mismatch")
        else:
            raise ValueError("triple_barrier_exit_reason_unknown")
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
            raise ValueError("triple_barrier_exit_price_invalid")
        gross_return = exit_price / entry_price - 1.0
        net_return = gross_return - cost_fraction
        rows.append(
            {
                "target": int(net_return > 0.0),
                "label_end_idx": final_index,
                "net_return": net_return,
                "gross_return": gross_return,
                "exit_reason": reason,
                "label": str(record["label"]),
            }
        )
    return DataFrame.from_records(rows)
