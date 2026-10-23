"""Tests for regime-gated signals, ATR exits and time-decay behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import user_data.strategies.AliXStrategy as strategy_module
from services.freqai_baseline.features import FEATURE_COLUMNS
from services.meta_labeling.meta_model import (
    META_MODEL_SCHEMA_VERSION,
    MetaLabelModel,
)
from user_data.strategies.AliXStrategy import AliXStrategy


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 102.0, 90.0, 100.0],
            "ema_20": [101.0, 101.0, 94.0, 101.0],
            "ema_50": [99.0, 100.0, 96.0, 100.0],
            "rsi_14": [40.0, 45.0, 25.0, 65.0],
            "bb_lower": [95.0, 96.0, 92.0, 97.0],
            "alix_data_quality_valid": [True, True, True, True],
            "alix_signal_bundle_approved": [True, True, True, True],
            "alix_regime": ["trend", "trend", "range", "range"],
            "alix_selected_strategy_id": [
                "trend_following",
                "trend_following",
                "mean_reversion",
                "mean_reversion",
            ],
            "alix_meta_probability": [0.75, 0.85, 0.91, 0.99],
        }
    )


def test_strategy_emits_only_approved_trend_and_mean_reversion_candidates() -> None:
    frame = _signal_frame()
    frame.loc[0, "close"] = 100.0  # previous close at/below EMA; next row crosses up
    frame.loc[0, "ema_20"] = 101.0
    frame.loc[0, "ema_50"] = 99.0
    result = AliXStrategy().populate_entry_trend(frame, {})
    assert result["enter_long"].tolist() == [0, 1, 1, 0]
    assert result.loc[1, "enter_tag"] == "hmm_trend_following_meta70"
    assert result.loc[2, "enter_tag"] == "hmm_mean_reversion_meta70"


def test_strategy_fails_closed_on_low_meta_score_and_missing_selection() -> None:
    frame = _signal_frame()
    frame.loc[1, "alix_meta_probability"] = 0.699
    frame.loc[2, "alix_selected_strategy_id"] = None
    result = AliXStrategy().populate_entry_trend(frame, {})
    assert result["enter_long"].sum() == 0


def test_technical_exit_is_available_without_ai_approval() -> None:
    frame = pd.DataFrame(
        {
            "close": [99.0, 100.0, 101.0],
            "ema_20": [100.0, 99.0, 100.0],
            "rsi_14": [40.0, 65.0, 50.0],
        }
    )
    result = AliXStrategy().populate_exit_trend(frame, {})
    assert result["exit_long"].tolist() == [1, 1, 0]
    assert result.loc[0, "exit_tag"] == "technical_trend_exit"
    assert result.loc[1, "exit_tag"] == "momentum_exit"


def test_dynamic_stop_uses_regime_atr_multiplier_and_fails_conservatively() -> None:
    strategy = AliXStrategy()
    analyzed = pd.DataFrame({"atr_14": [2.0], "alix_regime": ["trend"]})
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (analyzed, None)
    )
    trade = SimpleNamespace(open_rate=100.0)
    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    trend_stop = strategy.custom_stoploss("BTC/USDT", trade, now, 100.0, 0.0)
    assert trend_stop is not None
    assert abs(trend_stop - (-0.04)) < 1e-8
    analyzed.loc[0, "alix_regime"] = "range"
    range_stop = strategy.custom_stoploss("BTC/USDT", trade, now, 100.0, 0.0)
    assert range_stop is not None
    assert abs(range_stop - (-0.02)) < 1e-8
    analyzed.loc[0, "alix_regime"] = "high_volatility"
    volatile_stop = strategy.custom_stoploss("BTC/USDT", trade, now, 100.0, 0.0)
    assert volatile_stop is not None
    assert abs(volatile_stop - (-0.015)) < 1e-8
    assert strategy.custom_stoploss("BTC/USDT", trade, now, 0.0, 0.0) == -0.01


def test_custom_stop_handles_unavailable_data_and_invalid_atr() -> None:
    strategy = AliXStrategy()
    trade = SimpleNamespace(open_rate=100.0)
    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    assert strategy.custom_stoploss("BTC/USDT", trade, now, 100.0, 0.0) == -0.01
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (
            pd.DataFrame({"atr_14": [np.nan]}),
            None,
        )
    )
    assert strategy.custom_stoploss("BTC/USDT", trade, now, 100.0, 0.0) == -0.01


def test_time_decay_exit_only_closes_near_breakeven_after_twelve_candles() -> None:
    strategy = AliXStrategy()
    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    trade = SimpleNamespace(open_date_utc=now - timedelta(hours=13))
    assert (
        strategy.custom_exit("BTC/USDT", trade, now, 100.0, 0.001)
        == "time_decay_breakeven_12_candles"
    )
    assert strategy.custom_exit("BTC/USDT", trade, now, 100.0, -0.01) is None
    young_trade = SimpleNamespace(open_date_utc=now - timedelta(hours=11))
    assert strategy.custom_exit("BTC/USDT", young_trade, now, 100.0, 0.0) is None
    naive_trade = SimpleNamespace(open_date_utc=datetime(2026, 10, 23, 0, 0))
    assert strategy.custom_exit("BTC/USDT", naive_trade, now, 100.0, 0.0) is None


def test_populate_indicators_marks_model_or_selection_missing_as_no_trade() -> None:
    periods = 120
    close = np.linspace(100.0, 120.0, periods)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-10-01T00:30:00Z", periods=periods, freq="h"),
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(np.r_[close[0], close[:-1]], close) + 0.5,
            "low": np.minimum(np.r_[close[0], close[:-1]], close) - 0.5,
            "close": close,
            "volume": np.linspace(100, 200, periods),
        }
    )
    result = AliXStrategy().populate_indicators(frame, {"pair": "BTC/USDT"})
    assert result["alix_data_quality_valid"].all()
    assert (result["alix_regime"] == "NO_TRADE").all()
    assert (result["alix_signal_bundle_approved"] == False).all()  # noqa: E712


def test_meta_scoring_preserves_valid_tail_rows_after_nan_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_names = FEATURE_COLUMNS
    raw_model = {
        "schema_version": META_MODEL_SCHEMA_VERSION,
        "training_signal_requirement": "out_of_fold_only",
        "feature_names": list(feature_names),
        "means": [0.0] * len(feature_names),
        "scales": [1.0] * len(feature_names),
        "coefficients": [0.1] * len(feature_names),
        "intercept": 0.0,
        "regularization_c": 1.0,
        "max_iter": 1000,
    }
    model = MetaLabelModel.from_dict(raw_model)
    monkeypatch.setattr(strategy_module, "_load_meta_model", lambda: (model, True))
    frame = pd.DataFrame({name: [np.nan, 0.0, 0.1] for name in feature_names})

    result = strategy_module._score_meta_candidates(frame)

    assert not np.isfinite(cast(float, result.loc[0, "alix_meta_probability"]))
    assert result.loc[0, "alix_meta_model_approved"] == np.bool_(False)
    assert np.isfinite(result.loc[1:, "alix_meta_probability"]).all()
    assert result.loc[1:, "alix_meta_model_approved"].all()


def _write_meta_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_changes: dict[str, object] | None = None,
    approval_changes: dict[str, object] | None = None,
    invalid_model: bool = False,
) -> None:
    raw_model: dict[str, Any] = {
        "schema_version": META_MODEL_SCHEMA_VERSION,
        "training_signal_requirement": "out_of_fold_only",
        "feature_names": ["%-return_1"],
        "means": [0.0],
        "scales": [1.0],
        "coefficients": [0.5],
        "intercept": 0.0,
        "regularization_c": 1.0,
        "max_iter": 1000,
    }
    if invalid_model:
        raw_model["schema_version"] = "wrong-schema"
    model = MetaLabelModel.from_dict(raw_model) if not invalid_model else None
    digest = (
        hashlib.sha256(model.to_json().encode("utf-8")).hexdigest()
        if model is not None
        else "0" * 64
    )
    dataset_hash = "a" * 64
    manifest: dict[str, object] = {
        "model_sha256": digest,
        "dataset_sha256": dataset_hash,
        "dataset_rows": 8760,
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
    }
    approval: dict[str, object] = {
        "schema_version": "alix.model-technical-approval.v1",
        "model_type": "meta_label_logistic",
        "status": "OWNER_APPROVED_TECHNICAL_ONLY",
        "owner": "angry0tiger-sudo",
        "approval_id": "ALIX-TEST-META-APPROVAL",
        "model_sha256": digest,
        "dataset_sha256": dataset_hash,
        "approved_at": "2026-10-23T12:00:00+00:00",
        "economic_acceptance": "NO-GO",
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    manifest.update(manifest_changes or {})
    approval.update(approval_changes or {})
    model_path = tmp_path / "meta-model.json"
    manifest_path = tmp_path / "meta-manifest.json"
    approval_path = tmp_path / "meta-approval.json"
    model_path.write_text(json.dumps(raw_model), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    monkeypatch.setattr(strategy_module, "_META_MODEL_PATH", model_path)
    monkeypatch.setattr(strategy_module, "_META_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(strategy_module, "_META_APPROVAL_PATH", approval_path)


def test_meta_model_loader_fails_closed_on_missing_or_malformed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        strategy_module, "_META_MODEL_PATH", tmp_path / "missing-model.json"
    )
    monkeypatch.setattr(
        strategy_module, "_META_MANIFEST_PATH", tmp_path / "missing-manifest.json"
    )
    monkeypatch.setattr(
        strategy_module, "_META_APPROVAL_PATH", tmp_path / "missing-approval.json"
    )
    assert strategy_module._load_meta_model() == (None, False)

    _write_meta_artifacts(tmp_path, monkeypatch, invalid_model=True)
    assert strategy_module._load_meta_model() == (None, False)


def test_meta_model_loader_requires_manifest_and_closed_release_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_meta_artifacts(
        tmp_path,
        monkeypatch,
        manifest_changes={"dataset_rows": 8759},
    )
    assert strategy_module._load_meta_model() == (None, False)

    _write_meta_artifacts(
        tmp_path,
        monkeypatch,
        manifest_changes={"trading_authorization": "GO"},
    )
    assert strategy_module._load_meta_model() == (None, False)


@pytest.mark.parametrize(
    ("approval_change", "value"),
    [
        ("status", "NOT_APPROVED"),
        ("owner", "untrusted-owner"),
        ("approval_id", ""),
        ("model_sha256", "0" * 64),
        ("dataset_sha256", "0" * 64),
        ("approved_at", "not-a-timestamp"),
        ("approved_at", "2026-10-23T12:00:00"),
        ("order_permission", True),
        ("economic_acceptance", "GO"),
    ],
)
def test_meta_model_loader_requires_hash_pinned_technical_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_change: str,
    value: object,
) -> None:
    _write_meta_artifacts(
        tmp_path,
        monkeypatch,
        approval_changes={approval_change: value},
    )
    model, approved = strategy_module._load_meta_model()
    assert model is not None
    assert approved is False


def test_meta_model_loader_accepts_valid_technical_only_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_meta_artifacts(tmp_path, monkeypatch)
    model, approved = strategy_module._load_meta_model()
    assert model is not None
    assert approved is True
    assert strategy_module._load_meta_model()[1] is True
