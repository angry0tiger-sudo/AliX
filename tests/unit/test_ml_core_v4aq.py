from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

import services.meta_labeling.v4aq as v4aq


def _bars(rows: int = 35) -> DataFrame:
    close = np.linspace(100.0, 120.0, rows)
    return DataFrame(
        {
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _external(rows: int = 35) -> DataFrame:
    return DataFrame(
        {
            "funding_rate_zscore": np.linspace(-1.0, 1.0, rows),
            "oi_change": np.linspace(0.0, 0.1, rows),
        }
    )


def _assert_v4aq_finite(values: np.ndarray) -> None:
    if not bool(np.isfinite(values).all()):
        raise AssertionError("expected finite V4-AQ meta features")


def _assert_v4aq_missing(values: np.ndarray) -> None:
    if not bool(pd.isna(values).all()):
        raise AssertionError("expected missing warm-up V4-AQ meta features")


def test_v4aq_meta_features_are_causal_and_preserve_external_missing_rows() -> None:
    bars = _bars()
    external = _external()
    external.loc[0, "oi_change"] = np.nan
    features = v4aq.build_v4aq_meta_features(bars, external)
    assert tuple(features.columns) == v4aq.V4AQ_META_FEATURE_COLUMNS
    _assert_v4aq_missing(
        features.iloc[: v4aq.V4AQ_PRIMARY_LOOKBACK].to_numpy(dtype=float)
    )
    _assert_v4aq_missing(np.asarray([features.loc[0, "oi_change"]], dtype=float))
    _assert_v4aq_finite(features.iloc[21:].to_numpy(dtype=float))
    changed = bars.copy()
    changed.loc[30:, ["high", "low", "close"]] *= 100.0
    changed_features = v4aq.build_v4aq_meta_features(changed, external)
    pd.testing.assert_frame_equal(
        features.iloc[:25],
        changed_features.iloc[:25],
    )


def test_v4aq_feature_builder_supports_non_range_indexes() -> None:
    bars = _bars().set_axis(range(100, 135))
    external = _external().set_axis(range(100, 135))
    features = v4aq.build_v4aq_meta_features(bars, external)
    assert features.index.equals(bars.index)


def test_v4aq_feature_builder_rejects_bad_alignment_and_schema() -> None:
    with pytest.raises(ValueError, match="input_alignment_invalid"):
        v4aq.build_v4aq_meta_features(_bars(), _external(20))
    with pytest.raises(ValueError, match="ohlc_columns_missing"):
        v4aq.build_v4aq_meta_features(DataFrame({"close": [1.0]}), _external(1))
    with pytest.raises(ValueError, match="external_feature_schema_invalid"):
        v4aq.build_v4aq_meta_features(_bars(), DataFrame({"unknown": np.ones(35)}))
    bars = _bars()
    external = _external().set_axis(range(35, 70))
    with pytest.raises(ValueError, match="external_feature_index_mismatch"):
        v4aq.build_v4aq_meta_features(bars, external)


def test_v4aq_feature_builder_rejects_invalid_prices_and_infinite_external() -> None:
    bars = _bars()
    bars.loc[0, "high"] = np.nan
    with pytest.raises(ValueError, match="price_invalid:high"):
        v4aq.build_v4aq_meta_features(bars, _external())
    external = _external()
    external.loc[0, "funding_rate_zscore"] = np.inf
    with pytest.raises(ValueError, match="external_feature_infinite"):
        v4aq.build_v4aq_meta_features(_bars(), external)


def test_lightgbm_wrapper_enforces_schema_and_delegates_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = DataFrame(
        {
            "primary_close_vs_donchian_high_pct": np.ones(40),
            "primary_donchian_channel_width_pct": np.ones(40),
            "funding_rate_zscore": np.ones(40),
            "oi_change": np.ones(40),
        }
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
        fit,
        labels,
        calibration,
        target_accept_fraction=0.3,
    )
    assert model is not None
    assert seen["fit_columns"] == v4aq.V4AQ_META_FEATURE_COLUMNS
    assert seen["calibration_columns"] == v4aq.V4AQ_META_FEATURE_COLUMNS
    assert seen["target_accept_fraction"] == 0.3
    assert diagnostics["feature_columns"] == list(v4aq.V4AQ_META_FEATURE_COLUMNS)
    assert diagnostics["external_features_used"] == ["funding_rate_zscore", "oi_change"]


def test_lightgbm_wrapper_rejects_missing_features_and_misaligned_labels() -> None:
    frame = DataFrame(
        np.ones((40, len(v4aq.V4AQ_META_FEATURE_COLUMNS))),
        columns=v4aq.V4AQ_META_FEATURE_COLUMNS,
    )
    labels = Series([0, 1] * 20)
    with pytest.raises(ValueError, match="features_not_complete:fit"):
        invalid = frame.copy()
        invalid.iloc[0, 0] = np.nan
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
    values = v4aq.predict_v4aq_meta_probabilities(object(), frame)
    assert values.tolist() == [0.2, 0.5, 0.8]
    with pytest.raises(ValueError, match="feature_schema_mismatch:prediction"):
        v4aq.predict_v4aq_meta_probabilities(object(), frame.iloc[:, :2])
