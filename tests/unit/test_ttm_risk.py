"""Contract tests for TTM-R2 risk-only evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.ai_ensemble.ttm_risk import (
    CONTEXT_LENGTH,
    TARGET_COLUMN,
    PREDICTION_HORIZON,
    ForecastContextOnlyDataset,
    forecast_dataset_indices,
    make_evaluation_origins,
    prepare_risk_target,
    qlike_loss,
    risk_forecast_gate,
    score_variance_forecasts,
)


def _ohlcv(rows: int = 2_000) -> pd.DataFrame:
    timestamps = pd.date_range("2022-01-01", periods=rows, freq="h", tz="UTC")
    returns = np.sin(np.arange(rows, dtype=np.float64) / 17.0) * 0.001
    close = 40_000.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": timestamps, "close": close})


def test_risk_target_is_causal_and_does_not_fill_first_return() -> None:
    data = _ohlcv()
    result = prepare_risk_target(data)
    assert len(result) == len(data) - 1
    assert result["return_variance"].iloc[0] > 0.0
    assert np.isfinite(result["log1p_return_variance"]).all()


def test_risk_target_rejects_gaps_and_duplicate_timestamps() -> None:
    data = _ohlcv()
    with pytest.raises(ValueError, match="continuous 1h"):
        prepare_risk_target(data.drop(index=100).reset_index(drop=True))
    duplicated = data.copy()
    duplicated.loc[101, "date"] = duplicated.loc[100, "date"]
    with pytest.raises(ValueError, match="unique"):
        prepare_risk_target(duplicated)


def test_risk_target_rejects_invalid_prices() -> None:
    data = _ohlcv()
    data.loc[10, "close"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_risk_target(data)


def test_split_origins_are_causal_non_overlapping_and_inside_splits() -> None:
    row_count = 31_800
    split = make_evaluation_origins(row_count)
    for origins, lower, upper in (
        (split["validation_origins"], split["train_end"], split["validation_end"]),
        (split["holdout_origins"], split["validation_end"], row_count),
    ):
        assert len(origins) >= 30
        assert all(origin >= CONTEXT_LENGTH - 1 for origin in origins)
        assert all(origin + 1 >= lower for origin in origins)
        assert all(origin + PREDICTION_HORIZON < upper for origin in origins)
        assert all(b - a == PREDICTION_HORIZON for a, b in zip(origins, origins[1:]))


def test_dataset_indices_end_context_at_the_registered_origin() -> None:
    origins = [511, 535, 559]
    indices = forecast_dataset_indices(origins)
    assert indices == [0, 24, 48]
    assert all(
        index + CONTEXT_LENGTH - 1 == origin for index, origin in zip(indices, origins)
    )


def test_forecast_dataset_hides_all_future_labels_from_model() -> None:
    samples = [
        {
            "past_values": np.asarray([1.0, 2.0]),
            "past_observed_mask": np.asarray([True, True]),
            "future_values": np.asarray([999.0, 1000.0]),
            "future_observed_mask": np.asarray([True, True]),
            "pad_mask": np.asarray([False, False]),
        },
        {
            "past_values": np.asarray([3.0, 4.0]),
            "future_values": np.asarray([888.0, 889.0]),
        },
    ]
    dataset = ForecastContextOnlyDataset(samples, [1, 0])
    first = dataset[0]
    assert set(first) == {"past_values"}
    assert np.array_equal(first["past_values"], np.asarray([3.0, 4.0]))
    second = dataset[1]
    assert set(second) == {"past_values", "past_observed_mask"}
    assert "future_values" not in second and "future_observed_mask" not in second


def test_forecast_dataset_rejects_non_mapping_samples() -> None:
    dataset = ForecastContextOnlyDataset([np.asarray([1.0, 2.0])], [0])
    with pytest.raises(TypeError, match="mappings"):
        dataset[0]


def test_predict_context_only_uses_loss_free_batched_model_inference() -> None:
    torch = pytest.importorskip("torch")

    from scripts.run_ttm_r2_risk_research import _predict_context_only

    class ContextDataset:
        def __len__(self) -> int:
            return 35

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "past_values": torch.full((CONTEXT_LENGTH, 1), float(index)),
                "past_observed_mask": torch.ones((CONTEXT_LENGTH, 1), dtype=torch.bool),
                "future_values": torch.full((PREDICTION_HORIZON, 1), -999.0),
            }

    class ForecastModel:
        def __init__(self) -> None:
            self.eval_called = False
            self.batch_sizes: list[int] = []

        def eval(self) -> "ForecastModel":
            self.eval_called = True
            return self

        def __call__(
            self,
            *,
            past_values: object,
            past_observed_mask: object,
            return_loss: bool,
            return_dict: bool,
        ) -> SimpleNamespace:
            assert return_loss is False
            assert return_dict is True
            assert isinstance(past_values, torch.Tensor)
            assert isinstance(past_observed_mask, torch.Tensor)
            self.batch_sizes.append(int(past_values.shape[0]))
            forecast = past_values[:, -1:, :].expand(-1, PREDICTION_HORIZON, -1)
            return SimpleNamespace(prediction_outputs=forecast)

    model = ForecastModel()
    origins = list(range(CONTEXT_LENGTH - 1, CONTEXT_LENGTH - 1 + 35))
    predictions = _predict_context_only(model, ContextDataset(), origins)

    assert model.eval_called is True
    assert model.batch_sizes == [16, 16, 3]
    assert predictions.shape == (35, PREDICTION_HORIZON, 1)
    np.testing.assert_array_equal(predictions[:, 0, 0], np.arange(35, dtype=float))


def test_predict_context_only_rejects_empty_origins() -> None:
    from scripts.run_ttm_r2_risk_research import _predict_context_only

    with pytest.raises(ValueError, match="at least one forecast origin"):
        _predict_context_only(object(), object(), [])


def test_inverse_forecast_variance_applies_training_only_positive_domain_floor() -> (
    None
):
    from scripts.run_ttm_r2_risk_research import _inverse_forecast_variance

    class IdentityPreprocessor:
        def inverse_scale_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame

    floor = 0.025
    raw_predictions = np.full((2, PREDICTION_HORIZON, 1), -0.5, dtype=np.float64)
    raw_predictions[1, :, 0] = 0.2

    forecast, bounded_target, floor_fraction = _inverse_forecast_variance(
        raw_predictions,
        IdentityPreprocessor(),
        origin_count=2,
        minimum_log1p_variance_target=floor,
    )

    assert np.all(bounded_target[0] == floor)
    assert np.all(bounded_target[1] == 0.2)
    np.testing.assert_allclose(forecast[0], PREDICTION_HORIZON * np.expm1(floor))
    np.testing.assert_allclose(forecast[1], PREDICTION_HORIZON * np.expm1(0.2))
    np.testing.assert_array_equal(floor_fraction, np.asarray([1.0, 0.0]))


def test_training_positive_target_floor_uses_only_training_prefix() -> None:
    from scripts.run_ttm_r2_risk_research import _training_positive_target_floor

    training_positive_targets = np.linspace(0.02, 0.08, 40, dtype=np.float64)
    frame = pd.DataFrame(
        {TARGET_COLUMN: [0.0, *training_positive_targets.tolist(), -999.0]}
    )
    train_end = len(training_positive_targets) + 1
    floor = _training_positive_target_floor(frame, train_end=train_end)

    assert 0.02 <= floor <= 0.08
    assert floor < 0.08

    frame.loc[len(frame) - 1, TARGET_COLUMN] = 1_000_000.0
    assert _training_positive_target_floor(frame, train_end=train_end) == floor


def test_qlike_rejects_non_positive_or_misaligned_values() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        qlike_loss(np.asarray([1.0, 2.0]), np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="equal non-zero shape"):
        qlike_loss(np.asarray([1.0]), np.asarray([1.0, 2.0]))


def test_score_and_gate_require_incremental_risk_forecast_value() -> None:
    actual = np.asarray([0.8, 1.2, 0.9, 1.1], dtype=np.float64)
    baseline = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    excellent = np.asarray([0.82, 1.18, 0.92, 1.08], dtype=np.float64)
    poor = np.asarray([1.4, 0.7, 1.5, 0.6], dtype=np.float64)
    good_metrics = score_variance_forecasts(actual, excellent, baseline)
    poor_metrics = score_variance_forecasts(actual, poor, baseline)
    assert good_metrics["maximum_position_multiplier"] <= 1.0
    assert risk_forecast_gate({**good_metrics, "sample_count": 30}) is True
    assert risk_forecast_gate({**poor_metrics, "sample_count": 30}) is False
    assert risk_forecast_gate({**good_metrics, "sample_count": 29}) is False
