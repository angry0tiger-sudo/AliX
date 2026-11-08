from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

import services.meta_labeling.v4aq as v4aq


def _assert_v4aq_finite(values: np.ndarray) -> None:
    if not bool(np.isfinite(values).all()):
        raise AssertionError("expected finite meta features")


def _assert_v4aq_missing(values: DataFrame) -> None:
    if not bool(values.isna().all(axis=None)):
        raise AssertionError("expected missing warm-up meta features")


def _bars(rows: int = 260) -> DataFrame:
    close = 100.0 + np.sin(np.arange(rows) / 9.0) * 5.0 + np.arange(rows) * 0.02
    return DataFrame(
        {
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _external(rows: int = 260) -> DataFrame:
    return DataFrame(
        {
            "funding_rate_zscore": np.sin(np.arange(rows) / 5.0),
            "funding_rate_ma": np.cos(np.arange(rows) / 7.0) * 0.001,
        }
    )


def test_donchian_meta_context_is_causal_and_preserves_funding_missing_rows() -> None:
    bars = _bars()
    external = _external()
    external.loc[0, "funding_rate_ma"] = np.nan
    features = v4aq.build_v4aq_meta_features(
        bars, external, strategy=v4aq.DONCHIAN_STRATEGY
    )
    assert tuple(features.columns) == v4aq.V4AQ_META_FEATURE_COLUMNS
    _assert_v4aq_missing(features.iloc[: v4aq.V4AQ_PRIMARY_LOOKBACK, :2])
    assert pd.isna(features.loc[0, "funding_rate_ma"])
    _assert_v4aq_finite(features.iloc[21:].to_numpy(dtype=float))
    changed = bars.copy()
    changed.loc[30:, ["high", "low", "close"]] *= 100.0
    changed_features = v4aq.build_v4aq_meta_features(
        changed, external, strategy=v4aq.DONCHIAN_STRATEGY
    )
    pd.testing.assert_frame_equal(features.iloc[:25], changed_features.iloc[:25])


def test_mean_reversion_context_respects_ema200_warmup_and_index() -> None:
    bars = _bars().set_axis(range(100, 360))
    external = _external().set_axis(range(100, 360))
    features = v4aq.build_v4aq_meta_features(
        bars, external, strategy=v4aq.MEAN_REVERSION_STRATEGY
    )
    assert features.index.equals(bars.index)
    assert features["primary_context_2"].iloc[:199].isna().all()
    assert features["primary_context_1"].iloc[14:].notna().all()
    assert np.isfinite(features.iloc[200:, :2].to_numpy(dtype=float)).all()


def test_builder_rejects_bad_alignment_schema_prices_and_strategy() -> None:
    bars = _bars()
    external = _external()
    with pytest.raises(ValueError, match="input_alignment_invalid"):
        v4aq.build_v4aq_meta_features(
            bars, external.iloc[:20], strategy=v4aq.DONCHIAN_STRATEGY
        )
    with pytest.raises(ValueError, match="ohlc_columns_missing"):
        v4aq.build_v4aq_meta_features(
            DataFrame({"close": [1.0]}),
            DataFrame(
                {
                    "funding_rate_zscore": [0.0],
                    "funding_rate_ma": [0.0],
                }
            ),
            strategy=v4aq.DONCHIAN_STRATEGY,
        )
    with pytest.raises(ValueError, match="external_feature_schema_invalid"):
        v4aq.build_v4aq_meta_features(
            bars,
            DataFrame({"unknown": np.ones(len(bars))}),
            strategy=v4aq.DONCHIAN_STRATEGY,
        )
    with pytest.raises(ValueError, match="strategy_not_supported"):
        v4aq.build_v4aq_meta_features(bars, external, strategy="OTHER")
    mismatched = external.set_axis(range(100, 100 + len(external)))
    with pytest.raises(ValueError, match="index_mismatch"):
        v4aq.build_v4aq_meta_features(bars, mismatched, strategy=v4aq.DONCHIAN_STRATEGY)
    invalid = bars.copy()
    invalid.loc[0, "high"] = np.nan
    with pytest.raises(ValueError, match="price_invalid:high"):
        v4aq.build_v4aq_meta_features(
            invalid, external, strategy=v4aq.DONCHIAN_STRATEGY
        )
    infinite = external.copy()
    infinite.loc[0, "funding_rate_zscore"] = np.inf
    with pytest.raises(ValueError, match="external_feature_infinite"):
        v4aq.build_v4aq_meta_features(bars, infinite, strategy=v4aq.DONCHIAN_STRATEGY)


def test_lightgbm_wrapper_enforces_schema_and_delegates_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = DataFrame(
        np.ones((40, len(v4aq.V4AQ_META_FEATURE_COLUMNS))),
        columns=v4aq.V4AQ_META_FEATURE_COLUMNS,
    )
    calibration = fit.iloc[:8].copy()
    labels = Series([0, 1] * 20, dtype=int)
    seen: dict[str, object] = {}

    def fake_fit(
        fit_features: DataFrame,
        fit_labels: Series,
        calibration_features: DataFrame,
        *,
        target_accept_fraction: float,
    ) -> tuple[object, dict[str, object]]:
        seen["fit_columns"] = tuple(fit_features.columns)
        seen["calibration_columns"] = tuple(calibration_features.columns)
        seen["target_accept_fraction"] = target_accept_fraction
        return object(), {"fit_rows": len(fit_features)}

    monkeypatch.setattr(v4aq, "fit_lightgbm_meta_filter", fake_fit)
    model, diagnostics = v4aq.fit_v4aq_lightgbm_meta_filter(
        fit, labels, calibration, target_accept_fraction=0.3
    )
    assert model is not None
    assert seen["fit_columns"] == v4aq.V4AQ_META_FEATURE_COLUMNS
    assert seen["calibration_columns"] == v4aq.V4AQ_META_FEATURE_COLUMNS
    assert seen["target_accept_fraction"] == 0.3
    assert diagnostics["feature_columns"] == list(v4aq.V4AQ_META_FEATURE_COLUMNS)
    assert diagnostics["external_features_used"] == [
        "funding_rate_zscore",
        "funding_rate_ma",
    ]


def test_wrapper_rejects_missing_features_and_misaligned_labels() -> None:
    frame = DataFrame(
        np.ones((40, len(v4aq.V4AQ_META_FEATURE_COLUMNS))),
        columns=v4aq.V4AQ_META_FEATURE_COLUMNS,
    )
    labels = Series([0, 1] * 20)
    invalid = frame.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="features_not_complete:fit"):
        v4aq.fit_v4aq_lightgbm_meta_filter(invalid, labels, frame.iloc[:8])
    with pytest.raises(ValueError, match="label_alignment_invalid"):
        v4aq.fit_v4aq_lightgbm_meta_filter(frame, labels.iloc[:-1], frame.iloc[:8])


def test_probability_wrapper_checks_schema_then_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = DataFrame(
        np.ones((3, len(v4aq.V4AQ_META_FEATURE_COLUMNS))),
        columns=v4aq.V4AQ_META_FEATURE_COLUMNS,
    )
    monkeypatch.setattr(
        v4aq,
        "predict_meta_probabilities",
        lambda model, features: np.array([0.2, 0.5, 0.8]),
    )
    assert v4aq.predict_v4aq_meta_probabilities(object(), frame).tolist() == [
        0.2,
        0.5,
        0.8,
    ]
    with pytest.raises(ValueError, match="schema_mismatch:prediction"):
        v4aq.predict_v4aq_meta_probabilities(object(), frame.iloc[:, :2])
