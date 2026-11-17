"""Configuration commitment and runtime parameter-drift tests for Regime R."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import research.regime_r.ngboost_cqr.adapters as adapter_module
from research.regime_r.ngboost_cqr.adapters import (
    OWNER_EXECUTION_ENV,
    HGBQCQRAdapter,
)
from research.regime_r.ngboost_cqr.contracts import ContractViolation

EXPECTED_CONFIG_SHA256 = (
    "92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa"
)


def _synthetic_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build independent deterministic synthetic fit and calibration arrays."""
    train_x = np.linspace(-1.0, 1.0, 64, dtype=np.float64).reshape(-1, 1)
    train_y = 0.003 * train_x[:, 0] + 0.001 * np.sin(train_x[:, 0])
    calibration_x = np.linspace(-0.9, 0.9, 32, dtype=np.float64).reshape(-1, 1)
    calibration_y = 0.003 * calibration_x[:, 0] + 0.001 * np.sin(calibration_x[:, 0])
    return train_x, train_y, calibration_x, calibration_y


def test_preregistration_config_sha_and_fitted_model_parameters_are_locked() -> None:
    raw_config = adapter_module.PREREGISTRATION_CONFIG_PATH.read_bytes()
    assert hashlib.sha256(raw_config).hexdigest() == EXPECTED_CONFIG_SHA256
    config = json.loads(raw_config.decode("utf-8"))
    common = config["model"]["common_parameters"]
    expected_quantiles = tuple(
        item["quantile"] for item in config["model"]["quantile_estimators"]
    )

    adapter = HGBQCQRAdapter(horizon="4h")
    train_x, train_y, calibration_x, calibration_y = _synthetic_arrays()
    with patch.dict(os.environ, {OWNER_EXECUTION_ENV: "true"}):
        adapter.fit(train_x, train_y, calibration_x, calibration_y)

    assert adapter.config_sha256 == EXPECTED_CONFIG_SHA256
    assert adapter.quantile_levels == expected_quantiles
    assert adapter._models is not None
    assert len(adapter._models) == 3
    for model, quantile in zip(adapter._models, expected_quantiles, strict=True):
        assert model.loss == common["loss"]
        assert model.learning_rate == common["learning_rate"]
        assert model.max_depth == common["max_depth"]
        assert model.max_iter == common["max_iter"]
        assert model.l2_regularization == common["l2_regularization"]
        assert model.min_samples_leaf == common["min_samples_leaf"]
        assert model.max_leaf_nodes == common["max_leaf_nodes"]
        assert model.max_bins == common["max_bins"]
        assert model.early_stopping is common["early_stopping"]
        assert model.random_state == common["random_state"]
        assert model.quantile == quantile


def test_any_single_byte_configuration_change_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_config = bytearray(adapter_module.PREREGISTRATION_CONFIG_PATH.read_bytes())
    raw_config[0] = ord("[") if raw_config[0] != ord("[") else ord("{")
    modified = tmp_path / "hgbq_cqr_preregistration.json"
    modified.write_bytes(raw_config)
    monkeypatch.setattr(adapter_module, "PREREGISTRATION_CONFIG_PATH", modified)

    with pytest.raises(
        ContractViolation,
        match="hgbq_preregistration_config_sha256_mismatch",
    ):
        HGBQCQRAdapter()


def test_missing_preregistration_config_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing-preregistration.json"
    monkeypatch.setattr(adapter_module, "PREREGISTRATION_CONFIG_PATH", missing)

    with pytest.raises(
        ContractViolation,
        match="hgbq_preregistration_config_unavailable",
    ):
        HGBQCQRAdapter()
