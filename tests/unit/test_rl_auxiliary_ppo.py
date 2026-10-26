from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from services.rl_research.rl_trading_environment import (
    ACTION_BUY,
    ACTION_NAMES,
    BUY_ACTION_INDICES,
    SELL_ACTION_INDICES,
    OBSERVATION_FIELDS,
)

from scripts.rl_auxiliary_ppo import (
    DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS,
    DEFAULT_AUXILIARY_LOSS_WEIGHT,
    auxiliary_loss_curriculum_report,
    auxiliary_loss_weight,
)


def test_auxiliary_loss_weight_anneals_linearly_to_zero() -> None:
    assert auxiliary_loss_weight(0) == pytest.approx(0.1)
    assert auxiliary_loss_weight(250_000) == pytest.approx(0.05)
    assert auxiliary_loss_weight(500_000) == pytest.approx(0.0)
    assert auxiliary_loss_weight(1_000_000) == pytest.approx(0.0)


def test_auxiliary_loss_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="timestep_must_be_integer"):
        auxiliary_loss_weight(True)
    with pytest.raises(ValueError, match="timestep_must_be_non_negative"):
        auxiliary_loss_weight(-1)
    with pytest.raises(ValueError, match="weight_must_be_non_negative"):
        auxiliary_loss_weight(0, -0.1)
    with pytest.raises(ValueError, match="anneal_steps_must_be_positive"):
        auxiliary_loss_weight(0, 0.1, 0)


def test_auxiliary_loss_report_keeps_guidance_out_of_evaluation() -> None:
    report = auxiliary_loss_curriculum_report(1_000_000)
    assert report["method"] == "recurrent_policy_direction_cross_entropy"
    assert report["initial_weight"] == DEFAULT_AUXILIARY_LOSS_WEIGHT
    assert report["anneal_steps"] == DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS
    assert report["effective_weight_at_training_end"] == pytest.approx(0.0)
    assert report["anneal_completed"] is True
    assert report["validation_enabled"] is False
    assert report["final_holdout_enabled"] is False
    assert report["target_classes"] == {
        str(index): name for index, name in enumerate(ACTION_NAMES)
    }


def test_auxiliary_loss_recurrent_ppo_training_and_reload(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    dummy_vec_env = getattr(
        importlib.import_module("stable_baselines3.common.vec_env"), "DummyVecEnv"
    )

    from scripts.rl_auxiliary_ppo import build_auxiliary_recurrent_ppo
    from services.rl_research.rl_trading_environment import (
        BTCUSDTTradingEnv,
        MarketObservationScaler,
    )

    rows = 48
    timestamps = 1_700_000_000_000 + np.arange(rows, dtype=np.float64) * 60_000
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.1
    data = np.column_stack(
        (
            timestamps,
            close,
            close + 1.0,
            close - 1.0,
            close,
            np.ones(rows, dtype=np.float64),
        )
    ).astype(np.float64)
    scaler = MarketObservationScaler.fit(data)
    env = dummy_vec_env(
        [
            lambda: BTCUSDTTradingEnv(
                data,
                scaler=scaler,
                initial_equity=1_000.0,
                fee_bps_per_side=0.0,
                slippage_bps_per_side=0.0,
                min_remaining_steps=1,
                signal_guidance_reward_scale=0.0,
            )
        ]
    )
    algorithm = build_auxiliary_recurrent_ppo()
    model = algorithm(
        algorithm.policy_class,
        env,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=3e-4,
        ent_coef=0.05,
        seed=19,
        device="cpu",
        verbose=0,
        auxiliary_loss_weight=0.1,
        auxiliary_loss_anneal_steps=16,
        policy_kwargs={
            "lstm_hidden_size": 16,
            "n_lstm_layers": 1,
            "net_arch": [16],
        },
    )
    try:
        torch = importlib.import_module("torch")
        position_index = OBSERVATION_FIELDS.index("current_position_fraction")
        hidden = torch.zeros((1, 1, 16), dtype=torch.float32)
        episode_starts = torch.ones((1,), dtype=torch.float32)
        for position_fraction, forbidden_actions in (
            (0.0, SELL_ACTION_INDICES),
            (1.0, BUY_ACTION_INDICES),
        ):
            observation = np.zeros((1, len(OBSERVATION_FIELDS)), dtype=np.float32)
            observation[0, position_index] = position_fraction
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            distribution, _ = model.policy.get_distribution(
                observation_tensor,
                (hidden.clone(), hidden.clone()),
                episode_starts,
            )
            action_distribution = distribution.distribution
            for forbidden_action in forbidden_actions:
                assert action_distribution.probs[0, forbidden_action].item() == 0.0
            action, _ = model.predict(observation[0], deterministic=True)
            assert int(np.asarray(action).reshape(-1)[0]) not in forbidden_actions

        model.learn(total_timesteps=8, progress_bar=False)
        assert model.num_timesteps == 8
        assert model.rollout_buffer.guidance_targets.shape == (8, 1)
        assert np.all(
            np.isin(model.rollout_buffer.guidance_targets, np.arange(len(ACTION_NAMES)))
        )
        assert model.logger.name_to_value["train/auxiliary_loss"] >= 0.0
        assert model.logger.name_to_value[
            "train/auxiliary_loss_weight"
        ] == pytest.approx(0.05)
        checkpoint = tmp_path / "auxiliary-recurrent-policy"
        model.save(str(checkpoint))
        loaded_model = algorithm.load(str(checkpoint), env=None, device="cpu")
        assert loaded_model.num_timesteps == 8
        assert loaded_model.auxiliary_loss_weight == pytest.approx(0.1)
        assert loaded_model.auxiliary_loss_anneal_steps == 16

        # Critical regression: the serialized custom LSTM policy must still
        # assign exactly zero probability to BUY at full position.
        full_observation = np.zeros((1, len(OBSERVATION_FIELDS)), dtype=np.float32)
        full_observation[0, position_index] = 1.0
        full_observation_tensor = torch.as_tensor(full_observation, dtype=torch.float32)
        loaded_distribution, _ = loaded_model.policy.get_distribution(
            full_observation_tensor,
            (hidden.clone(), hidden.clone()),
            episode_starts,
        )
        for buy_action in BUY_ACTION_INDICES:
            assert loaded_distribution.distribution.probs[0, buy_action].item() == 0.0
        reloaded_action, _ = loaded_model.predict(
            full_observation[0], deterministic=True
        )
        assert int(np.asarray(reloaded_action).reshape(-1)[0]) not in BUY_ACTION_INDICES
    finally:
        env.close()


def test_full_position_masks_buy_probability_to_exact_zero_before_sampling() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    torch = importlib.import_module("torch")
    dummy_vec_env = getattr(
        importlib.import_module("stable_baselines3.common.vec_env"), "DummyVecEnv"
    )
    from scripts.rl_auxiliary_ppo import build_auxiliary_recurrent_ppo
    from services.rl_research.rl_trading_environment import (
        BTCUSDTTradingEnv,
        MarketObservationScaler,
    )

    timestamps = 1_700_000_000_000 + np.arange(8, dtype=np.float64) * 60_000
    closes = np.full(8, 100.0, dtype=np.float64)
    data = np.column_stack(
        (
            timestamps,
            closes,
            closes + 1.0,
            closes - 1.0,
            closes,
            np.ones(8, dtype=np.float64),
        )
    )
    scaler = MarketObservationScaler.fit(data)
    env = dummy_vec_env(
        [
            lambda: BTCUSDTTradingEnv(
                data,
                scaler=scaler,
                initial_equity=1_000.0,
                fee_bps_per_side=0.0,
                slippage_bps_per_side=0.0,
                min_remaining_steps=1,
                signal_guidance_reward_scale=0.0,
            )
        ]
    )
    algorithm = build_auxiliary_recurrent_ppo()
    model = algorithm(
        algorithm.policy_class,
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        seed=20261026,
        device="cpu",
        verbose=0,
        policy_kwargs={
            "lstm_hidden_size": 16,
            "n_lstm_layers": 1,
            "net_arch": [16],
        },
    )
    try:
        position_index = OBSERVATION_FIELDS.index("current_position_fraction")
        observation = np.zeros((1, len(OBSERVATION_FIELDS)), dtype=np.float32)
        observation[0, position_index] = 1.0
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        hidden = torch.zeros((1, 1, 16), dtype=torch.float32)
        episode_starts = torch.ones((1,), dtype=torch.float32)
        assert model.action_space.n == 7
        distribution, _ = model.policy.get_distribution(
            observation_tensor,
            (hidden.clone(), hidden.clone()),
            episode_starts,
        )
        action_distribution = distribution.distribution
        assert all(
            action_distribution.probs[0, buy_action].item() == 0.0
            for buy_action in BUY_ACTION_INDICES
        )
        action, _ = model.predict(observation[0], deterministic=True)
        assert int(np.asarray(action).reshape(-1)[0]) not in BUY_ACTION_INDICES
    finally:
        env.close()
