"""Pinned LightGBM regression baseline for the FreqAI 2026.7 contract.

The model is a research predictor only. It has no access to exchange clients,
credentials, risk-policy writes, or order lifecycle methods.
"""

from __future__ import annotations

from typing import Any

from services.freqai_baseline.walkforward import BASELINE_MODEL_PARAMETERS

try:
    from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel  # type: ignore[import-not-found]
except ImportError:  # Quality CI intentionally has no Freqtrade runtime dependency.

    class BaseRegressionModel:  # type: ignore[no-redef]
        """Static-analysis fallback; runtime resolves the pinned Freqtrade class."""

        pass


class AliXBaselineRegressor(BaseRegressionModel):  # type: ignore[misc]
    """Deterministic LightGBM regression baseline following Freqtrade's fit contract."""

    def fit(self, data_dictionary: dict[str, Any], dk: Any, **kwargs: Any) -> Any:
        # Defer the optional runtime imports so contract tests stay offline/lightweight.
        from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

        configured = getattr(self, "model_training_parameters", {})
        parameters = dict(BASELINE_MODEL_PARAMETERS)
        if isinstance(configured, dict):
            parameters.update(configured)

        # The Phase 4 baseline must be deterministic across runs and single-threaded.
        parameters.update(
            {
                "objective": "regression",
                "random_state": 42,
                "n_jobs": 1,
                "deterministic": True,
                "force_col_wise": True,
                "verbosity": -1,
            }
        )
        estimator = LGBMRegressor(**parameters)

        fit_kwargs: dict[str, Any] = {
            "X": data_dictionary["train_features"],
            "y": data_dictionary["train_labels"],
            "sample_weight": data_dictionary.get("train_weights"),
        }
        test_features = data_dictionary.get("test_features")
        test_labels = data_dictionary.get("test_labels")
        if test_features is not None and test_labels is not None and len(test_labels):
            fit_kwargs["eval_set"] = [(test_features, test_labels)]
            eval_weights = data_dictionary.get("test_weights")
            if eval_weights is not None:
                fit_kwargs["eval_sample_weight"] = [eval_weights]

        estimator.fit(**fit_kwargs)
        return estimator
