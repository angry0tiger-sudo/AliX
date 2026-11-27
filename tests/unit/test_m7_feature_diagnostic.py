"""Mock-only tests for M7 feature availability diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.diagnose_m7_features import (
    inspect_beta_feature,
    inspect_volume_feature,
)


def test_beta_feature_accepts_finite_rolling_windows() -> None:
    returns = np.asarray([math.nan, *([0.01, -0.02] * 12)], dtype=float)
    value, reason = inspect_beta_feature(returns, returns.copy(), 24)
    assert reason is None
    assert value == pytest.approx(1.0)


def test_beta_feature_rejects_nonfinite_asset_window() -> None:
    asset_returns = np.asarray([math.nan, *([0.01, -0.02] * 12)], dtype=float)
    btc_returns = asset_returns.copy()
    asset_returns[5] = math.nan
    value, reason = inspect_beta_feature(asset_returns, btc_returns, 24)
    assert value is None
    assert reason == "asset_return_window_nonfinite"


def test_beta_feature_rejects_zero_btc_variance() -> None:
    asset_returns = np.asarray([math.nan, *([0.01, -0.02] * 12)], dtype=float)
    btc_returns = np.zeros(25, dtype=float)
    btc_returns[0] = math.nan
    value, reason = inspect_beta_feature(asset_returns, btc_returns, 24)
    assert value is None
    assert reason == "btc_variance_nonfinite_or_at_or_below_1e-12"


def test_volume_feature_rejects_zero_prior_median() -> None:
    volume = np.zeros(25, dtype=float)
    value, reason = inspect_volume_feature(volume, 24)
    assert value is None
    assert reason == "prior_24h_median_volume_nonpositive"


def test_volume_feature_accepts_positive_prior_median() -> None:
    volume = np.asarray([10.0] * 24 + [20.0], dtype=float)
    value, reason = inspect_volume_feature(volume, 24)
    assert reason is None
    assert value == pytest.approx(math.log(3.0))
