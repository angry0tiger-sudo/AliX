"""Binary BUY/NO_BUY target adapter for the V4-V Spot-only research path."""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandas import DataFrame

SOURCE_LABELS: Final[tuple[str, ...]] = ("BUY", "SELL", "HOLD")
BINARY_BUY_LABELS: Final[tuple[str, ...]] = ("BUY", "NO_BUY")
BINARY_TARGET_SCHEMA_VERSION: Final[str] = "alix.ml-core.v4v-binary-target.v1"
_REQUIRED_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "label_end_idx",
    "label",
    "exit_reason",
)


def apply_binary_buy_targets(labels: DataFrame) -> DataFrame:
    """Map BUY to BUY and merge SELL/HOLD into NO_BUY without losing provenance.

    Barrier outcome and maturity columns are preserved unchanged. The original
    three-class label is retained as source_label for audit and diagnostics.
    Unlabeled rows remain unlabeled; they are never coerced to NO_BUY.
    """
    missing = sorted(set(_REQUIRED_LABEL_COLUMNS) - set(labels.columns))
    if missing:
        raise ValueError(f"v4v_label_columns_missing:{','.join(missing)}")
    if labels.empty:
        raise ValueError("v4v_label_input_empty")

    source = labels["label"]
    observed = {str(value) for value in source.tolist() if not pd.isna(value)}
    unexpected = sorted(observed - set(SOURCE_LABELS))
    if unexpected:
        raise ValueError("v4v_source_label_values_invalid")

    result = labels.copy()
    result["source_label"] = source.astype(object)
    result["label"] = source.map(
        {
            "BUY": "BUY",
            "SELL": "NO_BUY",
            "HOLD": "NO_BUY",
        }
    ).astype(object)

    mapped_values = {
        str(value) for value in result["label"].tolist() if not pd.isna(value)
    }
    if not mapped_values.issubset(set(BINARY_BUY_LABELS)):
        raise ValueError("v4v_binary_label_values_invalid")
    if not result["source_label"].equals(source.astype(object)):
        raise ValueError("v4v_source_label_provenance_lost")
    return result
