"""Tests for historical HMM training and tamper-evident model loading."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from services.strategy_selection.hmm_training import (
    FEATURE_NAMES,
    calculate_regime_features,
    fit_gaussian_hmm,
    fit_parameters_for_artifact,
    validate_hmm_holdout,
)
from services.strategy_selection.model_io import (
    MODEL_SCHEMA_VERSION,
    load_hmm_model_document,
)


def _feature_rows(count: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(72)
    labels = np.arange(count) % 3
    efficiency = np.choose(
        labels,
        [
            rng.normal(0.80, 0.05, count),
            rng.normal(0.20, 0.05, count),
            rng.normal(0.35, 0.08, count),
        ],
    )
    volatility = np.choose(
        labels,
        [
            rng.normal(0.004, 0.0008, count),
            rng.normal(0.003, 0.0005, count),
            rng.normal(0.018, 0.002, count),
        ],
    )
    return pd.DataFrame(
        {
            "directional_efficiency": np.clip(efficiency, 0.01, 0.99),
            "realized_volatility": np.clip(volatility, 0.0001, None),
        }
    ).loc[:, list(FEATURE_NAMES)]


def _artifact_document() -> str:
    model = {
        "revision": "hmm-test-v1",
        "trained_at": datetime(2026, 10, 1, tzinfo=timezone.utc).isoformat(),
        "owner_approved": False,
        "feature_schema_version": "market-regime-features.v1",
        "feature_names": ["directional_efficiency", "realized_volatility"],
        "regimes": ["trend", "range", "high_volatility"],
        "initial_probabilities": ["0.34", "0.33", "0.33"],
        "transition_matrix": [
            ["0.8", "0.1", "0.1"],
            ["0.1", "0.8", "0.1"],
            ["0.1", "0.1", "0.8"],
        ],
        "emissions": [
            {
                "regime": "trend",
                "means": ["0.8", "0.004"],
                "standard_deviations": ["0.1", "0.002"],
            },
            {
                "regime": "range",
                "means": ["0.2", "0.003"],
                "standard_deviations": ["0.1", "0.001"],
            },
            {
                "regime": "high_volatility",
                "means": ["0.3", "0.02"],
                "standard_deviations": ["0.1", "0.005"],
            },
        ],
        "max_model_age_seconds": 2592000,
        "max_observation_age_seconds": 7200,
        "minimum_confidence": "0.55",
        "minimum_posterior_margin": "0.10",
    }
    canonical = json.dumps(
        model, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    document = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "TRAINED_UNAPPROVED",
        "model_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "model": model,
        "dataset": {
            "provider": "Nobitex public UDF history API",
            "csv_sha256": "a" * 64,
            "row_count": 8760,
            "symbol": "BTCUSDT",
            "pair": "BTC/USDT",
            "timeframe": "1h",
        },
        "validation": {"passed": True, "status": "TECHNICAL_VALIDATION_PASSED"},
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
    }
    return json.dumps(document)


def test_calculate_regime_features_uses_only_causal_ohlcv_windows() -> None:
    close = np.exp(np.cumsum(np.random.default_rng(4).normal(0.0001, 0.003, 80))) * 100
    candles = pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(np.r_[close[0], close[:-1]], close) * 1.002,
            "low": np.minimum(np.r_[close[0], close[:-1]], close) * 0.998,
            "close": close,
            "volume": np.linspace(10, 20, len(close)),
        }
    )
    features = calculate_regime_features(candles)
    assert tuple(features.columns) == FEATURE_NAMES
    assert features.iloc[:12].isna().all().all()
    assert np.isfinite(features.iloc[13:].to_numpy()).all()


def test_hmm_training_and_holdout_validation_produce_finite_auditable_values() -> None:
    features = _feature_rows()
    fitted = fit_gaussian_hmm(features.iloc[:300], max_iter=12)
    assert len(fitted.initial_probabilities) == 3
    assert len(fitted.transition_matrix) == 3
    assert all(abs(sum(row) - 1.0) < 1e-8 for row in fitted.transition_matrix)
    assert all(value > 0 for value in fitted.state_occupancy)
    validated = validate_hmm_holdout(features.iloc[:300], features.iloc[300:])
    assert validated["passed"] is True
    assert validated["economic_acceptance"] == "NOT_EVALUATED"


def test_hmm_training_rejects_bad_schemas_and_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        fit_gaussian_hmm(pd.DataFrame({"wrong": [1.0] * 200}))
    invalid = _feature_rows(200)
    invalid.loc[0, "realized_volatility"] = np.nan
    with pytest.raises(ValueError, match="non_finite"):
        fit_gaussian_hmm(invalid)
    with pytest.raises(ValueError, match="sample_count"):
        fit_gaussian_hmm(_feature_rows(50))


def test_hmm_artifact_parameters_are_json_native_and_semantically_ordered() -> None:
    parameters = fit_parameters_for_artifact(_feature_rows(240))
    assert len(parameters["emissions"]) == 3
    assert [value["regime"] for value in parameters["emissions"]] == [
        "trend",
        "range",
        "high_volatility",
    ]
    assert parameters["fit_diagnostics"]["iterations"] >= 1


def test_model_loader_validates_digest_schema_and_release_flags() -> None:
    loaded = load_hmm_model_document(_artifact_document())
    assert loaded.revision == "hmm-test-v1"
    assert loaded.owner_approved is False
    raw = json.loads(_artifact_document())
    raw["model"]["minimum_confidence"] = "0.60"
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_hmm_model_document(json.dumps(raw))
    raw = json.loads(_artifact_document())
    raw["status"] = "PROMOTED"
    with pytest.raises(ValueError, match="status_not_loadable"):
        load_hmm_model_document(json.dumps(raw))
    raw = json.loads(_artifact_document())
    raw["economic_acceptance"] = "GO"
    with pytest.raises(ValueError, match="economic_gate"):
        load_hmm_model_document(json.dumps(raw))


def test_model_loader_rejects_invalid_dataset_and_validation_provenance() -> None:
    raw = json.loads(_artifact_document())
    raw["dataset"]["row_count"] = 8759
    with pytest.raises(ValueError, match="dataset_scope"):
        load_hmm_model_document(json.dumps(raw))
    raw = json.loads(_artifact_document())
    raw["validation"]["passed"] = False
    with pytest.raises(ValueError, match="validation_not_passed"):
        load_hmm_model_document(json.dumps(raw))


def test_model_loader_supports_explicit_technical_approval_without_economic_promotion() -> (
    None
):
    raw = json.loads(_artifact_document())
    raw["status"] = "OWNER_APPROVED_TECHNICAL_ONLY"
    raw["model"]["owner_approved"] = True
    canonical = json.dumps(
        raw["model"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    raw["model_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    loaded = load_hmm_model_document(json.dumps(raw))
    assert loaded.owner_approved is True
    assert raw["economic_acceptance"] == "NO-GO"
    assert raw["trading_authorization"] == "NO-GO"
