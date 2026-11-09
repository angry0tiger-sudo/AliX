"""Three-state diagonal Gaussian HMM with causal forward-filtered posteriors."""

from __future__ import annotations

import numpy as np


class CausalGaussianHMM:
    """A small deterministic Gaussian HMM; no backward-smoothed outputs are exposed."""

    def __init__(
        self,
        n_states: int = 3,
        *,
        max_iter: int = 30,
        tolerance: float = 1e-4,
        covariance_floor: float = 0.05,
        random_state: int = 17,
    ) -> None:
        if n_states != 3:
            raise ValueError("regime_model_requires_exactly_three_states")
        if max_iter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("hmm_iteration_configuration_invalid")
        if not np.isfinite(covariance_floor) or covariance_floor <= 0:
            raise ValueError("hmm_covariance_floor_invalid")
        self.n_states = n_states
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.covariance_floor = covariance_floor
        self.random_state = random_state
        self.is_fitted_ = False
        self.location_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.initial_probabilities_: np.ndarray | None = None
        self.transition_matrix_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.state_order_: np.ndarray | None = None
        self.last_state_probability_: np.ndarray | None = None

    @staticmethod
    def _matrix(values: object, *, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(f"{name}_must_be_nonempty_2d")
        if not bool(np.isfinite(matrix).all()):
            raise ValueError(f"{name}_must_be_finite")
        return matrix

    @staticmethod
    def _normalize(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
        totals = values.sum(axis=axis, keepdims=axis is not None)
        if bool((totals <= 0).any()):
            raise ValueError("hmm_probability_normalization_failed")
        return values / totals

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            not self.is_fitted_
            or self.location_ is None
            or self.scale_ is None
            or self.initial_probabilities_ is None
            or self.transition_matrix_ is None
            or self.means_ is None
            or self.variances_ is None
            or self.state_order_ is None
        ):
            raise RuntimeError("hmm_model_not_fitted")
        return self.location_, self.scale_, self.means_, self.variances_

    def _emissions(self, observations: np.ndarray) -> np.ndarray:
        _, _, means, variances = self._require_fitted()
        delta = observations[:, None, :] - means[None, :, :]
        log_probability = -0.5 * (
            np.log(2.0 * np.pi * variances)[None, :, :] + (delta * delta) / variances[None, :, :]
        ).sum(axis=2)
        shifted = log_probability - log_probability.max(axis=1, keepdims=True)
        return np.exp(shifted)

    def _forward_backward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.initial_probabilities_ is None or self.transition_matrix_ is None:
            raise RuntimeError("hmm_parameters_not_initialized")
        emissions = self._emissions(observations)
        row_count = observations.shape[0]
        alpha = np.zeros((row_count, self.n_states), dtype=float)
        scales = np.zeros(row_count, dtype=float)
        first = self.initial_probabilities_ * emissions[0]
        scales[0] = max(float(first.sum()), 1e-300)
        alpha[0] = first / scales[0]
        for index in range(1, row_count):
            raw = (alpha[index - 1] @ self.transition_matrix_) * emissions[index]
            scales[index] = max(float(raw.sum()), 1e-300)
            alpha[index] = raw / scales[index]
        beta = np.ones((row_count, self.n_states), dtype=float)
        for index in range(row_count - 2, -1, -1):
            raw = self.transition_matrix_ @ (emissions[index + 1] * beta[index + 1])
            beta[index] = raw / max(float(scales[index + 1]), 1e-300)
        gamma = self._normalize(alpha * beta, axis=1)
        transition_counts = np.zeros((self.n_states, self.n_states), dtype=float)
        for index in range(row_count - 1):
            raw = (
                alpha[index, :, None]
                * self.transition_matrix_
                * (emissions[index + 1] * beta[index + 1])[None, :]
            )
            total = max(float(raw.sum()), 1e-300)
            transition_counts += raw / total
        return gamma, transition_counts

    def fit(self, values: object) -> CausalGaussianHMM:
        observations = self._matrix(values, name="hmm_training_observations")
        if observations.shape[0] < 45 or observations.shape[1] < 2:
            raise ValueError("hmm_training_sample_or_feature_count_too_small")
        self.location_ = observations.mean(axis=0)
        raw_scale = observations.std(axis=0)
        self.scale_ = np.where(raw_scale > 1e-12, raw_scale, 1.0)
        normalized = (observations - self.location_) / self.scale_
        sorted_indices = np.argsort(normalized[:, 0])
        quantile_positions = (
            max(0, int(len(sorted_indices) * 0.17)),
            max(0, int(len(sorted_indices) * 0.50)),
            max(0, int(len(sorted_indices) * 0.83)),
        )
        self.means_ = np.vstack(
            [normalized[sorted_indices[position]] for position in quantile_positions]
        )
        base_variance = np.maximum(
            normalized.var(axis=0), self.covariance_floor
        )
        self.variances_ = np.tile(base_variance, (self.n_states, 1))
        self.initial_probabilities_ = np.full(self.n_states, 1.0 / self.n_states)
        self.transition_matrix_ = np.full(
            (self.n_states, self.n_states), 0.03 / (self.n_states - 1)
        )
        np.fill_diagonal(self.transition_matrix_, 0.94)
        previous_means = self.means_.copy()
        for _ in range(self.max_iter):
            gamma, transition_counts = self._forward_backward(normalized)
            occupancy = np.maximum(gamma.sum(axis=0), 1e-8)
            self.initial_probabilities_ = self._normalize(gamma[0])
            transition_counts += 1e-5
            self.transition_matrix_ = self._normalize(transition_counts, axis=1)
            updated_means = (gamma.T @ normalized) / occupancy[:, None]
            updated_variances = np.zeros_like(self.variances_)
            for state in range(self.n_states):
                residual = normalized - updated_means[state]
                weighted = gamma[:, state, None] * (residual * residual)
                updated_variances[state] = np.maximum(
                    weighted.sum(axis=0) / occupancy[state],
                    self.covariance_floor,
                )
            self.means_ = updated_means
            self.variances_ = updated_variances
            movement = float(np.linalg.norm(self.means_ - previous_means))
            previous_means = self.means_.copy()
            if movement <= self.tolerance:
                break
        self.state_order_ = np.argsort(self.means_[:, 0])
        self.is_fitted_ = True
        filtered = self.filter(observations)
        self.last_state_probability_ = filtered[-1].copy()
        return self

    def filter(
        self,
        values: object,
        *,
        initial_state_probabilities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return P(state_t | observations_0..t), never smoothed P(state_t | all)."""
        location, scale, _, _ = self._require_fitted()
        observations = self._matrix(values, name="hmm_filter_observations")
        if observations.shape[1] != len(location):
            raise ValueError("hmm_filter_feature_count_mismatch")
        normalized = (observations - location) / scale
        emissions = self._emissions(normalized)
        assert self.initial_probabilities_ is not None
        assert self.transition_matrix_ is not None
        assert self.state_order_ is not None
        if initial_state_probabilities is None:
            prior = self.initial_probabilities_.copy()
        else:
            supplied = np.asarray(initial_state_probabilities, dtype=float)
            if supplied.shape != (self.n_states,) or not bool(np.isfinite(supplied).all()):
                raise ValueError("hmm_initial_state_probabilities_invalid")
            if bool((supplied < 0).any()) or float(supplied.sum()) <= 0:
                raise ValueError("hmm_initial_state_probabilities_invalid")
            sorted_prior = self._normalize(supplied)
            prior = np.empty(self.n_states, dtype=float)
            prior[self.state_order_] = sorted_prior
        filtered = np.zeros((len(normalized), self.n_states), dtype=float)
        for index in range(len(normalized)):
            raw = prior * emissions[index]
            total = max(float(raw.sum()), 1e-300)
            posterior = raw / total
            filtered[index] = posterior[self.state_order_]
            prior = posterior @ self.transition_matrix_
        return filtered

    def next_prior_probabilities(self) -> np.ndarray:
        """Return the next-step prior in DOWN/RANGE/UP state order."""
        self._require_fitted()
        assert self.transition_matrix_ is not None
        assert self.state_order_ is not None
        if self.last_state_probability_ is None:
            raise RuntimeError("hmm_training_filter_state_missing")
        previous = np.empty(self.n_states, dtype=float)
        previous[self.state_order_] = self.last_state_probability_
        next_prior = previous @ self.transition_matrix_
        return self._normalize(next_prior[self.state_order_])
