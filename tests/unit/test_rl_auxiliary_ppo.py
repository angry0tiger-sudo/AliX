from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

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
    assert report["target_classes"] == {"0": "HOLD", "1": "BUY", "2": "SELL"}


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
        "MlpLstmPolicy",
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
        model.learn(total_timesteps=8, progress_bar=False)
        assert model.num_timesteps == 8
        assert model.rollout_buffer.guidance_targets.shape == (8,)
        assert np.all(
            np.isin(model.rollout_buffer.guidance_targets, np.asarray([0, 1, 2]))
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
    finally:
        env.close()
