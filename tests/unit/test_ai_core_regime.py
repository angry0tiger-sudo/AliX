from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ai_core.regime import infer_hmm_regimes
from services.strategy_selection.hmm_training import HMMFit

FEATURE_NAMES = ("directional_efficiency", "realized_volatility")


def fitted_parameters(**changes: object) -> HMMFit:
    values: dict[str, object] = {
        "initial_probabilities": (0.34, 0.33, 0.33),
        "transition_matrix": (
            (0.85, 0.10, 0.05),
            (0.10, 0.85, 0.05),
            (0.10, 0.10, 0.80),
        ),
        "means": ((0.85, 0.004), (0.20, 0.003), (0.35, 0.020)),
        "standard_deviations": (
            (0.10, 0.002),
            (0.10, 0.002),
            (0.12, 0.006),
        ),
        "mean_log_likelihood": -1.0,
        "iterations": 10,
        "state_occupancy": (0.4, 0.4, 0.2),
    }
    values.update(changes)
    return HMMFit(**values)  # type: ignore[arg-type]


def observations(rows: int = 20) -> DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "directional_efficiency": np.clip(
                0.72 + 0.1 * np.sin(index / 4.0), 0.0, 1.0
            ),
            "realized_volatility": 0.004 + 0.0005 * (1.0 + np.cos(index / 5.0)),
        }
    )


def test_hmm_inference_returns_normalized_causal_posteriors() -> None:
    source = observations(20)
    regimes, posterior = infer_hmm_regimes(source, fitted_parameters())
    assert len(regimes) == len(source)
    assert len(posterior) == len(source)
    assert set(regimes).issubset({"trend", "range", "high_volatility"})
    assert all(sum(row) == pytest.approx(1.0) for row in posterior)
    assert all(0.0 <= value <= 1.0 for row in posterior for value in row)


def test_hmm_inference_keeps_log_probabilities_stable_when_posteriors_underflow() -> None:
    fit = fitted_parameters(
        means=((0.70, 0.004), (0.0, 0.004), (1.0, 0.004)),
        standard_deviations=(
            (0.001, 0.001),
            (0.001, 0.001),
            (0.001, 0.001),
        ),
    )
    source = observations(8)
    with np.errstate(divide="raise", invalid="raise"):
        regimes, posterior = infer_hmm_regimes(source, fit)

    assert len(regimes) == len(source)
    assert all(np.isfinite(value) for row in posterior for value in row)
    assert any(value == 0.0 for row in posterior for value in row)
    assert all(sum(row) == pytest.approx(1.0) for row in posterior)


def test_hmm_forward_filter_is_invariant_to_future_observation_changes() -> None:
    fit = fitted_parameters()
    history = observations(15)
    future = observations(8)
    altered_future = future.copy()
    altered_future["directional_efficiency"] = (
        1.0 - altered_future["directional_efficiency"]
    )
    altered_future["realized_volatility"] = 0.08
    first_regimes, first_posterior = infer_hmm_regimes(history, fit)
    combined_a = pd.concat([history, future], ignore_index=True)
    combined_b = pd.concat([history, altered_future], ignore_index=True)
    all_regimes_a, all_posterior_a = infer_hmm_regimes(combined_a, fit)
    all_regimes_b, all_posterior_b = infer_hmm_regimes(combined_b, fit)
    assert all_regimes_a[: len(history)] == first_regimes
    assert all_regimes_b[: len(history)] == first_regimes
    assert all_posterior_a[: len(history)] == first_posterior
    assert all_posterior_b[: len(history)] == first_posterior


@pytest.mark.parametrize(
    ("frame", "reason"),
    (
        (pd.DataFrame({"wrong": [0.1], "other": [0.2]}), "feature_schema_mismatch"),
        (pd.DataFrame(columns=list(FEATURE_NAMES)), "observations_invalid"),
        (
            pd.DataFrame(
                {"directional_efficiency": [np.nan], "realized_volatility": [0.2]}
            ),
            "observations_invalid",
        ),
        (
            pd.DataFrame(
                {"directional_efficiency": [1.2], "realized_volatility": [0.2]}
            ),
            "observations_invalid",
        ),
        (
            pd.DataFrame(
                {"directional_efficiency": [0.5], "realized_volatility": [-0.2]}
            ),
            "observations_invalid",
        ),
    ),
)
def test_hmm_inference_rejects_invalid_feature_rows(
    frame: DataFrame, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        infer_hmm_regimes(frame, fitted_parameters())


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"initial_probabilities": (0.5, 0.5, 0.5)}, "probability_contract_invalid"),
        (
            {
                "transition_matrix": (
                    (0.8, 0.2, 0.0),
                    (0.1, 0.8, 0.1),
                    (0.1, 0.1, 0.8),
                )
            },
            "parameters_invalid",
        ),
        (
            {
                "means": (
                    (0.8, 0.01),
                    (0.2, 0.02),
                )
            },
            "parameter_shape_invalid",
        ),
        (
            {
                "standard_deviations": (
                    (0.1, 0.0),
                    (0.1, 0.002),
                    (0.12, 0.006),
                )
            },
            "parameters_invalid",
        ),
        (
            {
                "means": (
                    (0.8, float("nan")),
                    (0.2, 0.003),
                    (0.3, 0.02),
                )
            },
            "parameters_invalid",
        ),
    ),
)
def test_hmm_inference_fails_closed_on_invalid_model_parameters(
    changes: dict[str, object], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        infer_hmm_regimes(observations(5), fitted_parameters(**changes))
