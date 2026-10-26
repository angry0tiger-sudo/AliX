"""Recurrent PPO training with an annealed guidance-classification loss.

Stable-Baselines3-Contrib is an optional, pinned research dependency. All imports
of that package and PyTorch are therefore delayed until the training workflow
constructs the research algorithm.
"""

from __future__ import annotations

import importlib
import math
from copy import deepcopy
from typing import Any, NamedTuple

import numpy as np

from services.rl_research.rl_trading_environment import (
    ACTION_NAMES,
    OBSERVATION_FIELDS,
)

DEFAULT_AUXILIARY_LOSS_WEIGHT = 0.1
DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS = 500_000
GUIDANCE_ACTION_COUNT = 3
ACTION_MASK_POSITION_EPSILON = 1e-6
ACTION_MASKED_POLICY_CLASS_NAME = "ActionMaskedRecurrentActorCriticPolicy"


class GuidanceRecurrentRolloutBufferSamples(NamedTuple):
    observations: Any
    actions: Any
    old_values: Any
    old_log_prob: Any
    advantages: Any
    returns: Any
    lstm_states: Any
    episode_starts: Any
    mask: Any
    guidance_targets: Any


def auxiliary_loss_weight(
    num_timesteps: int,
    initial_weight: float = DEFAULT_AUXILIARY_LOSS_WEIGHT,
    anneal_steps: int = DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS,
) -> float:
    """Return a linear auxiliary-loss schedule, clamped at zero after annealing."""
    if isinstance(num_timesteps, bool) or not isinstance(num_timesteps, int):
        raise ValueError("auxiliary_loss_timestep_must_be_integer")
    if num_timesteps < 0:
        raise ValueError("auxiliary_loss_timestep_must_be_non_negative")
    if isinstance(initial_weight, bool) or not math.isfinite(initial_weight):
        raise ValueError("auxiliary_loss_weight_must_be_finite")
    if initial_weight < 0.0:
        raise ValueError("auxiliary_loss_weight_must_be_non_negative")
    if isinstance(anneal_steps, bool) or not isinstance(anneal_steps, int):
        raise ValueError("auxiliary_loss_anneal_steps_must_be_integer")
    if anneal_steps < 1:
        raise ValueError("auxiliary_loss_anneal_steps_must_be_positive")
    return float(initial_weight * max(0.0, 1.0 - num_timesteps / anneal_steps))


def auxiliary_loss_curriculum_report(
    trained_timesteps: int,
    initial_weight: float = DEFAULT_AUXILIARY_LOSS_WEIGHT,
    anneal_steps: int = DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS,
) -> dict[str, Any]:
    """Describe the fixed, training-only schedule in the machine-readable report."""
    return {
        "method": "recurrent_policy_direction_cross_entropy",
        "initial_weight": float(initial_weight),
        "anneal_steps": int(anneal_steps),
        "trained_timesteps": int(trained_timesteps),
        "effective_weight_at_training_end": auxiliary_loss_weight(
            int(trained_timesteps), initial_weight, anneal_steps
        ),
        "anneal_completed": int(trained_timesteps) >= anneal_steps,
        "target_classes": {"0": "HOLD", "1": "BUY", "2": "SELL"},
        "target_rule": (
            "causal 20-completed-candle SMA slope; choose BUY or SELL only when "
            "that action is legal in the current portfolio state; otherwise HOLD"
        ),
        "validation_enabled": False,
        "final_holdout_enabled": False,
    }


def _validate_guidance_batch(targets: Any, n_envs: int) -> np.ndarray:
    values = np.asarray(targets)
    if values.shape != (n_envs,) or values.dtype.kind not in "iu":
        raise ValueError("guidance_target_batch_shape_or_dtype_invalid")
    if np.any(values < 0) or np.any(values >= GUIDANCE_ACTION_COUNT):
        raise ValueError("guidance_target_class_out_of_range")
    return values.astype(np.int64, copy=False)


def _buffer_reset(self: Any) -> None:
    super(type(self), self).reset()
    self.guidance_targets = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)


def _buffer_add(
    self: Any, *args: Any, lstm_states: Any, guidance_targets: Any, **kwargs: Any
) -> None:
    targets = _validate_guidance_batch(guidance_targets, self.n_envs)
    self.guidance_targets[self.pos] = targets
    super(type(self), self).add(*args, lstm_states=lstm_states, **kwargs)


def _buffer_get(self: Any, batch_size: int | None = None) -> Any:
    if not self.generator_ready:
        self.guidance_targets = self.swap_and_flatten(self.guidance_targets)
    return super(type(self), self).get(batch_size)


def _buffer_get_samples(
    self: Any, batch_inds: np.ndarray, env_change: np.ndarray, env: Any = None
) -> GuidanceRecurrentRolloutBufferSamples:
    base_samples = super(type(self), self)._get_samples(batch_inds, env_change, env)
    targets = self.pad_and_flatten(self.guidance_targets[batch_inds]).long()
    return GuidanceRecurrentRolloutBufferSamples(
        observations=base_samples.observations,
        actions=base_samples.actions,
        old_values=base_samples.old_values,
        old_log_prob=base_samples.old_log_prob,
        advantages=base_samples.advantages,
        returns=base_samples.returns,
        lstm_states=base_samples.lstm_states,
        episode_starts=base_samples.episode_starts,
        mask=base_samples.mask,
        guidance_targets=targets,
    )


def _bootstrap_truncated_rewards(
    model: Any,
    infos: list[dict[str, Any]],
    dones: np.ndarray,
    rewards: np.ndarray,
    lstm_states: Any,
    torch: Any,
) -> None:
    for index, done in enumerate(dones):
        if (
            not done
            or infos[index].get("terminal_observation") is None
            or not infos[index].get("TimeLimit.truncated", False)
        ):
            continue
        terminal_observation = model.policy.obs_to_tensor(
            infos[index]["terminal_observation"]
        )[0]
        critic_state = (
            lstm_states.vf[0][:, index : index + 1, :].contiguous(),
            lstm_states.vf[1][:, index : index + 1, :].contiguous(),
        )
        episode_start = torch.tensor([False], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            terminal_value = model.policy.predict_values(
                terminal_observation, critic_state, episode_start
            )[0]
        rewards[index] += model.gamma * terminal_value


def _collect_guided_step(
    model: Any,
    env: Any,
    callback: Any,
    rollout_buffer: Any,
    lstm_states: Any,
    step_index: int,
    torch: Any,
    obs_as_tensor: Any,
    spaces: Any,
) -> tuple[Any, np.ndarray, Any] | None:
    if (
        model.use_sde
        and model.sde_sample_freq > 0
        and step_index % model.sde_sample_freq == 0
    ):
        model.policy.reset_noise(env.num_envs)

    with torch.no_grad():
        observation_tensor = obs_as_tensor(model._last_obs, model.device)
        episode_starts = torch.tensor(
            model._last_episode_starts,
            dtype=torch.float32,
            device=model.device,
        )
        actions, values, log_probs, next_lstm_states = model.policy(
            observation_tensor, lstm_states, episode_starts
        )
    actions = actions.cpu().numpy()
    clipped_actions = actions
    if isinstance(model.action_space, spaces.Box):
        clipped_actions = np.clip(
            actions, model.action_space.low, model.action_space.high
        )

    new_observations, rewards, dones, infos = env.step(clipped_actions)
    model.num_timesteps += env.num_envs
    callback.update_locals(locals())
    if not callback.on_step():
        return None
    model._update_info_buffer(infos, dones)

    if isinstance(model.action_space, spaces.Discrete):
        actions = actions.reshape(-1, 1)
    _bootstrap_truncated_rewards(model, infos, dones, rewards, next_lstm_states, torch)
    guidance_targets = _validate_guidance_batch(
        [info.get("guidance_target_index") for info in infos], env.num_envs
    )
    rollout_buffer.add(
        model._last_obs,
        actions,
        rewards,
        model._last_episode_starts,
        values,
        log_probs,
        lstm_states=model._last_lstm_states,
        guidance_targets=guidance_targets,
    )
    return new_observations, dones, next_lstm_states


def _collect_rollouts_with_guidance(
    self: Any, env: Any, callback: Any, rollout_buffer: Any, n_rollout_steps: int
) -> bool:
    torch = importlib.import_module("torch")
    spaces = importlib.import_module("gymnasium.spaces")
    obs_as_tensor = getattr(
        importlib.import_module("stable_baselines3.common.utils"), "obs_as_tensor"
    )
    recurrent_buffers = importlib.import_module("sb3_contrib.common.recurrent.buffers")
    valid_buffer_types = (
        getattr(recurrent_buffers, "RecurrentRolloutBuffer"),
        getattr(recurrent_buffers, "RecurrentDictRolloutBuffer"),
    )
    if not isinstance(rollout_buffer, valid_buffer_types):
        raise TypeError("guidance_rollout_buffer_must_support_recurrent_policy")
    if self._last_obs is None:
        raise RuntimeError("recurrent_policy_initial_observation_missing")

    self.policy.set_training_mode(False)
    rollout_buffer.reset()
    if self.use_sde:
        self.policy.reset_noise(env.num_envs)
    callback.on_rollout_start()
    lstm_states = deepcopy(self._last_lstm_states)
    new_observations = self._last_obs
    dones = np.asarray(self._last_episode_starts, dtype=np.bool_)

    for step_index in range(n_rollout_steps):
        result = _collect_guided_step(
            self,
            env,
            callback,
            rollout_buffer,
            lstm_states,
            step_index,
            torch,
            obs_as_tensor,
            spaces,
        )
        if result is None:
            return False
        new_observations, dones, lstm_states = result
        self._last_obs = new_observations
        self._last_episode_starts = dones
        self._last_lstm_states = lstm_states

    with torch.no_grad():
        episode_starts = torch.tensor(dones, dtype=torch.float32, device=self.device)
        final_values = self.policy.predict_values(
            obs_as_tensor(new_observations, self.device),
            lstm_states.vf,
            episode_starts,
        )
    rollout_buffer.compute_returns_and_advantage(last_values=final_values, dones=dones)
    callback.on_rollout_end()
    return True


def _compute_minibatch_losses(
    model: Any,
    rollout_data: Any,
    clip_range: float,
    clip_range_vf: float | None,
    aux_weight: float,
    torch: Any,
    functional: Any,
) -> dict[str, Any]:
    actions = rollout_data.actions
    if isinstance(
        model.action_space, importlib.import_module("gymnasium.spaces").Discrete
    ):
        actions = actions.long().flatten()
    mask = rollout_data.mask > 1e-8
    values, log_prob, entropy = model.policy.evaluate_actions(
        rollout_data.observations,
        actions,
        rollout_data.lstm_states,
        rollout_data.episode_starts,
    )
    values = values.flatten()
    advantages = rollout_data.advantages
    if model.normalize_advantage:
        advantages = (advantages - advantages[mask].mean()) / (
            advantages[mask].std() + 1e-8
        )
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    loss_1 = advantages * ratio
    loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.mean(torch.min(loss_1, loss_2)[mask])
    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()[mask])
    values_pred = values
    if clip_range_vf is not None:
        values_pred = rollout_data.old_values + torch.clamp(
            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
        )
    value_loss = torch.mean(((rollout_data.returns - values_pred) ** 2)[mask])
    entropy_loss = (
        -torch.mean(-log_prob[mask]) if entropy is None else -torch.mean(entropy[mask])
    )
    distribution, _ = model.policy.get_distribution(
        rollout_data.observations,
        rollout_data.lstm_states.pi,
        rollout_data.episode_starts,
    )
    components = getattr(distribution, "distribution", None)
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError("auxiliary_loss_requires_two_component_action_distribution")
    direction_logits = components[0].logits
    targets = rollout_data.guidance_targets.long()
    auxiliary_loss = functional.cross_entropy(direction_logits[mask], targets[mask])
    total_loss = (
        policy_loss
        + model.ent_coef * entropy_loss
        + model.vf_coef * value_loss
        + aux_weight * auxiliary_loss
    )
    with torch.no_grad():
        log_ratio = log_prob - rollout_data.old_log_prob
        approx_kl = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)[mask])
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "auxiliary_loss": auxiliary_loss,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }


def _optimize_minibatch(model: Any, losses: dict[str, Any], torch: Any) -> None:
    model.policy.optimizer.zero_grad()
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
    model.policy.optimizer.step()


def _run_training_epoch(
    model: Any,
    clip_range: float,
    clip_range_vf: float | None,
    aux_weight: float,
    torch: Any,
    functional: Any,
    history: dict[str, list[float]],
) -> tuple[bool, list[float]]:
    epoch_kl: list[float] = []
    for rollout_data in model.rollout_buffer.get(model.batch_size):
        losses = _compute_minibatch_losses(
            model,
            rollout_data,
            clip_range,
            clip_range_vf,
            aux_weight,
            torch,
            functional,
        )
        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "auxiliary_loss",
            "clip_fraction",
        ):
            history[key].append(float(losses[key].item()))
        approx_kl = float(losses["approx_kl"].cpu().item())
        epoch_kl.append(approx_kl)
        if model.target_kl is not None and approx_kl > 1.5 * model.target_kl:
            if model.verbose >= 1:
                print(f"Early stopping due to target KL: {approx_kl:.4f}")
            return False, epoch_kl
        _optimize_minibatch(model, losses, torch)
    return True, epoch_kl


def _train_with_auxiliary_loss(self: Any) -> None:
    torch = importlib.import_module("torch")
    functional = getattr(torch.nn, "functional")
    utils = importlib.import_module("stable_baselines3.common.utils")
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)
    clip_range_vf = (
        self.clip_range_vf(self._current_progress_remaining)
        if self.clip_range_vf is not None
        else None
    )
    aux_weight = auxiliary_loss_weight(
        int(self.num_timesteps),
        float(self.auxiliary_loss_weight),
        int(self.auxiliary_loss_anneal_steps),
    )
    history: dict[str, list[float]] = {
        key: []
        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "auxiliary_loss",
            "clip_fraction",
        )
    }
    epoch_kls: list[float] = []
    for _ in range(self.n_epochs):
        keep_training, approx_kls = _run_training_epoch(
            self,
            clip_range,
            clip_range_vf,
            aux_weight,
            torch,
            functional,
            history,
        )
        epoch_kls.extend(approx_kls)
        self._n_updates += 1
        if not keep_training:
            break

    self.logger.record("train/entropy_loss", float(np.mean(history["entropy_loss"])))
    self.logger.record(
        "train/policy_gradient_loss", float(np.mean(history["policy_loss"]))
    )
    self.logger.record("train/value_loss", float(np.mean(history["value_loss"])))
    self.logger.record(
        "train/auxiliary_loss", float(np.mean(history["auxiliary_loss"]))
    )
    self.logger.record("train/auxiliary_loss_weight", aux_weight)
    self.logger.record("train/approx_kl", float(np.mean(epoch_kls)))
    self.logger.record("train/clip_fraction", float(np.mean(history["clip_fraction"])))
    self.logger.record("train/loss", history["loss"][-1])
    explained_var = utils.explained_variance(
        self.rollout_buffer.values.flatten(),
        self.rollout_buffer.returns.flatten(),
    )
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
        self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if clip_range_vf is not None:
        self.logger.record("train/clip_range_vf", clip_range_vf)


def _build_action_masked_recurrent_policy() -> Any:
    """Create a recurrent actor-critic policy that masks illegal direction logits."""
    policy_base = getattr(
        importlib.import_module("sb3_contrib.common.recurrent.policies"),
        "RecurrentActorCriticPolicy",
    )
    existing = globals().get(ACTION_MASKED_POLICY_CLASS_NAME)
    if isinstance(existing, type) and issubclass(existing, policy_base):
        return existing

    try:
        position_index = OBSERVATION_FIELDS.index("current_position_fraction")
        buy_index = ACTION_NAMES.index("BUY")
        sell_index = ACTION_NAMES.index("SELL")
    except ValueError as error:
        raise RuntimeError("action_mask_observation_contract_invalid") from error

    def call_with_observation_context(
        self: Any, method: Any, observations: Any, *args: Any, **kwargs: Any
    ) -> Any:
        had_context = hasattr(self, "_action_mask_observation")
        previous_context = getattr(self, "_action_mask_observation", None)
        self._action_mask_observation = observations
        try:
            return method(self, observations, *args, **kwargs)
        finally:
            if had_context:
                self._action_mask_observation = previous_context
            else:
                delattr(self, "_action_mask_observation")

    def forward(
        self: Any,
        obs: Any,
        lstm_states: Any,
        episode_starts: Any,
        deterministic: bool = False,
    ) -> Any:
        return call_with_observation_context(
            self, policy_base.forward, obs, lstm_states, episode_starts, deterministic
        )

    def evaluate_actions(
        self: Any, obs: Any, actions: Any, lstm_states: Any, episode_starts: Any
    ) -> Any:
        return call_with_observation_context(
            self,
            policy_base.evaluate_actions,
            obs,
            actions,
            lstm_states,
            episode_starts,
        )

    def get_distribution(
        self: Any, obs: Any, lstm_states: Any, episode_starts: Any
    ) -> Any:
        return call_with_observation_context(
            self, policy_base.get_distribution, obs, lstm_states, episode_starts
        )

    def _get_action_dist_from_latent(self: Any, latent_pi: Any) -> Any:
        torch = importlib.import_module("torch")
        observations = getattr(self, "_action_mask_observation", None)
        if observations is None:
            raise RuntimeError("action_mask_observation_context_missing")

        distribution = policy_base._get_action_dist_from_latent(self, latent_pi)
        components = getattr(distribution, "distribution", None)
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError(
                "action_mask_requires_two_component_multidiscrete_distribution"
            )
        if components[0].logits.ndim != 2 or components[0].logits.shape[1] != len(
            ACTION_NAMES
        ):
            raise ValueError("action_mask_direction_logits_shape_invalid")

        observation_shape = getattr(observations, "shape", ())
        if len(observation_shape) < 2 or observation_shape[-1] <= position_index:
            raise ValueError("action_mask_position_observation_missing")
        position_fraction = (
            observations[..., position_index]
            .reshape(-1)
            .to(
                device=components[0].logits.device,
                dtype=components[0].logits.dtype,
            )
        )
        direction_logits = components[0].logits
        if position_fraction.numel() != direction_logits.shape[0]:
            raise ValueError("action_mask_observation_batch_alignment_invalid")
        if (
            not bool(torch.isfinite(position_fraction).all().item())
            or bool(torch.any(position_fraction < -ACTION_MASK_POSITION_EPSILON).item())
            or bool(
                torch.any(position_fraction > 1.0 + ACTION_MASK_POSITION_EPSILON).item()
            )
        ):
            raise ValueError("action_mask_position_fraction_out_of_range")

        fully_allocated = position_fraction >= 1.0 - ACTION_MASK_POSITION_EPSILON
        flat = position_fraction <= ACTION_MASK_POSITION_EPSILON
        masked_logits = direction_logits.clone()
        masked_logits[fully_allocated, buy_index] = -torch.inf
        masked_logits[flat, sell_index] = -torch.inf
        components[0] = torch.distributions.Categorical(logits=masked_logits)
        distribution.distribution = components
        return distribution

    policy_class = type(
        ACTION_MASKED_POLICY_CLASS_NAME,
        (policy_base,),
        {
            "forward": forward,
            "evaluate_actions": evaluate_actions,
            "get_distribution": get_distribution,
            "_get_action_dist_from_latent": _get_action_dist_from_latent,
            "__module__": __name__,
        },
    )
    globals()[ACTION_MASKED_POLICY_CLASS_NAME] = policy_class
    return policy_class


def build_auxiliary_recurrent_ppo() -> Any:
    """Build a RecurrentPPO subclass with guidance labels in its recurrent buffer."""
    recurrent_ppo = getattr(
        importlib.import_module("sb3_contrib.ppo_recurrent.ppo_recurrent"),
        "RecurrentPPO",
    )
    buffer_base = getattr(
        importlib.import_module("sb3_contrib.common.recurrent.buffers"),
        "RecurrentRolloutBuffer",
    )

    guidance_buffer = type(
        "GuidanceRecurrentRolloutBuffer",
        (buffer_base,),
        {
            "reset": _buffer_reset,
            "add": _buffer_add,
            "get": _buffer_get,
            "_get_samples": _buffer_get_samples,
        },
    )

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        weight = float(
            kwargs.pop("auxiliary_loss_weight", DEFAULT_AUXILIARY_LOSS_WEIGHT)
        )
        anneal_steps = kwargs.pop(
            "auxiliary_loss_anneal_steps",
            DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS,
        )
        auxiliary_loss_weight(0, weight, anneal_steps)
        recurrent_ppo.__init__(self, *args, **kwargs)
        self.auxiliary_loss_weight = weight
        self.auxiliary_loss_anneal_steps = int(anneal_steps)

    def setup_model(self: Any) -> None:
        recurrent_ppo._setup_model(self)
        original_buffer = self.rollout_buffer
        self.rollout_buffer = guidance_buffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            original_buffer.hidden_state_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    policy_class = _build_action_masked_recurrent_policy()
    auxiliary_recurrent_ppo = type(
        "AuxiliaryRecurrentPPO",
        (recurrent_ppo,),
        {
            "__init__": initialize,
            "_setup_model": setup_model,
            "collect_rollouts": _collect_rollouts_with_guidance,
            "train": _train_with_auxiliary_loss,
            "policy_class": policy_class,
        },
    )
    return auxiliary_recurrent_ppo
