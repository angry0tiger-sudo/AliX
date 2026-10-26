"""Unit tests for the Discrete(7) policy diagnostics."""

from __future__ import annotations

import math

from scripts.diagnose_rl_discrete7_policy import (
    _check_mask_contract,
    _cross_entropy_from_logits,
    _expected_masked_indices,
)
from services.rl_research.rl_trading_environment import (
    ACTION_NAMES,
    BUY_ACTION_INDICES,
    SELL_ACTION_INDICES,
)


def test_flat_position_masks_all_sell_logits_and_probabilities() -> None:
    raw = [0.3, 0.2, 0.1, 0.0, 0.4, 0.5, 0.6]
    masked = raw.copy()
    probabilities = [0.2, 0.2, 0.2, 0.4, 0.0, 0.0, 0.0]
    for index in SELL_ACTION_INDICES:
        masked[index] = -1e8
    record = {
        "observation_position_fraction": 0.0,
        "raw_logits": raw,
        "masked_logits": masked,
        "probabilities": probabilities,
    }
    assert _expected_masked_indices(0.0) == list(SELL_ACTION_INDICES)
    assert _check_mask_contract(record)["passed"] is True


def test_full_position_masks_all_buy_logits_and_probabilities() -> None:
    raw = [0.3, 0.2, 0.1, 0.0, 0.4, 0.5, 0.6]
    masked = raw.copy()
    probabilities = [0.4, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2]
    for index in BUY_ACTION_INDICES:
        masked[index] = -1e8
    record = {
        "observation_position_fraction": 1.0,
        "raw_logits": raw,
        "masked_logits": masked,
        "probabilities": probabilities,
    }
    assert _expected_masked_indices(1.0) == list(BUY_ACTION_INDICES)
    assert _check_mask_contract(record)["passed"] is True


def test_partial_position_does_not_mask_either_direction() -> None:
    raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    record = {
        "observation_position_fraction": 0.5,
        "raw_logits": raw,
        "masked_logits": raw.copy(),
        "probabilities": [1 / 7] * len(ACTION_NAMES),
    }
    assert _expected_masked_indices(0.5) == []
    assert _check_mask_contract(record)["passed"] is True


def test_mask_diagnostic_rejects_nonzero_probability_for_forbidden_action() -> None:
    raw = [0.3, 0.2, 0.1, 0.0, 0.4, 0.5, 0.6]
    masked = raw.copy()
    for index in SELL_ACTION_INDICES:
        masked[index] = -1e8
    probabilities = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    record = {
        "observation_position_fraction": 0.0,
        "raw_logits": raw,
        "masked_logits": masked,
        "probabilities": probabilities,
    }
    assert _check_mask_contract(record)["passed"] is False


def test_cross_entropy_is_nonnegative_and_prefers_target_logit() -> None:
    balanced = _cross_entropy_from_logits([0.0, 0.0, 0.0], target=0)
    confident = _cross_entropy_from_logits([8.0, 0.0, 0.0], target=0)
    assert math.isclose(balanced, math.log(3), rel_tol=1e-12)
    assert 0.0 <= confident < balanced
