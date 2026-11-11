"""Train a PPO binary filter on immutable pre-hack Nobitex signal episodes.

The primary Ensemble is fitted only on the earliest 45% of the verified dataset.
Its frozen probabilities define out-of-sample signal episodes. PPO decides once
per episode whether to take or skip; a taken position is closed when the signal
episode ends. Validation and final holdout remain untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame

from services.rl_research.risk_allocation_environment import (
    ACTION_FRACTIONS,
    ACTION_NAMES,
    BAR_INTERVAL_MS,
    DRAWDOWN_EXCESS_PENALTY,
    DRAWDOWN_INCREMENT_PENALTY,
    DRAWDOWN_LIMIT,
    DRAWDOWN_SEVERE_PENALTY,
    OBSERVATION_FIELDS,
    PRIMARY_SIGNAL_OBSERVATION_INDEX,
    ROUND_TRIP_COST_BPS,
    TAKE_SIGNAL_REWARD_BONUS,
    RiskAllocationEnv,
)

SOURCE_DATA_COMMIT_SHA = "e6c86b08da98ad28e479b45827fb8de6cd3cc8dd"
EXPECTED_SOURCE_POLICY = "NOBITEX_ONLY_NO_BACKFILL"
EXPECTED_PROVIDER = "Nobitex public UDF history API"
EXPECTED_ROWS = 31_801
EXPECTED_START_TIMESTAMP = 1_635_712_200
EXPECTED_END_TIMESTAMP = 1_750_192_200
EMBARGO_BARS = 24
TEACHER_FIT_FRACTION = 0.45
PPO_TRAIN_END_FRACTION = 0.75
VALIDATION_END_FRACTION = 0.875
PRIMARY_PROBABILITY_THRESHOLD = 0.50
TRAINING_EPISODE_STEPS = 512
MINIMUM_OOS_TRADES = 51
PPO_DEFAULT_TIMESTEPS = 200_000
PPO_MINIMUM_TIMESTEPS = 200_000
CSV_NAME = "nobitex-btcusdt-1h.csv"
MANIFEST_NAME = "nobitex-btcusdt-1h.manifest.json"
REPORT_SCHEMA = "alix.ai6.ppo-risk-research.v3"


@dataclass(frozen=True, slots=True)
class MarketSegment:
    market: NDArray[np.float64]
    atr_fraction: NDArray[np.float64]
    recent_returns: NDArray[np.float64]
    primary_signal: NDArray[np.bool_]
    signal_probability: NDArray[np.float64]
    start_index: int
    end_index: int

    def __len__(self) -> int:
        return len(self.market)


def _json_default(value: object) -> object:
    """Convert NumPy scalar report values to native JSON-compatible scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported_json_value:{type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_commit_sha(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("rl_commit_sha_invalid")


def _validate_dataset_manifest(manifest: dict[str, Any], csv_digest: str) -> None:
    expected_manifest = {
        "source_policy": EXPECTED_SOURCE_POLICY,
        "data_sources": [EXPECTED_PROVIDER],
        "dataset_window": "PRE_HACK_NOBITEX_ONLY",
        "provider": EXPECTED_PROVIDER,
        "symbol": "BTCUSDT",
        "pair": "BTC/USDT",
        "interval": "1h",
        "requested_days": 1325,
        "row_count": EXPECTED_ROWS,
        "expected_candles": EXPECTED_ROWS,
        "start_timestamp": EXPECTED_START_TIMESTAMP,
        "end_timestamp": EXPECTED_END_TIMESTAMP,
        "missing_intervals": 0,
        "csv_sha256": csv_digest,
        "commit_sha": SOURCE_DATA_COMMIT_SHA,
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"rl_dataset_manifest_contract_mismatch:{field}")
    if manifest.get("source_overrides") not in (None, []):
        raise ValueError("rl_external_source_override_forbidden")


def _validate_dataset_timestamps(data: DataFrame) -> None:
    if data["date"].isna().any() or len(data) != EXPECTED_ROWS:
        raise ValueError("rl_dataset_timestamp_or_row_count_invalid")
    timestamps = data["date"].astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000
    expected_timestamps = np.arange(
        EXPECTED_START_TIMESTAMP,
        EXPECTED_END_TIMESTAMP + 1,
        3600,
        dtype=np.int64,
    )
    if not np.array_equal(timestamps, expected_timestamps):
        raise ValueError("rl_dataset_prehack_hourly_grid_mismatch")
    if np.any(timestamps > EXPECTED_END_TIMESTAMP):
        raise ValueError("rl_post_hack_candle_forbidden")


def _validate_dataset_ohlcv_values(data: DataFrame) -> None:
    numeric = data.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("rl_dataset_contains_non_finite_values")
    if np.any(numeric[:, :4] <= 0.0) or np.any(numeric[:, 4] < 0.0):
        raise ValueError("rl_dataset_ohlcv_out_of_range")


def _read_validated_dataset_csv(csv_path: Path) -> DataFrame:
    data = pd.read_csv(csv_path)
    required = ("date", "open", "high", "low", "close", "volume")
    if tuple(data.columns) != required:
        raise ValueError("rl_dataset_columns_mismatch")
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    _validate_dataset_timestamps(data)
    _validate_dataset_ohlcv_values(data)
    return data


def _load_validated_dataset(input_dir: Path) -> tuple[DataFrame, dict[str, Any], str]:
    csv_path = input_dir / CSV_NAME
    manifest_path = input_dir / MANIFEST_NAME
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError("rl_prehack_nobitex_artifact_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("rl_dataset_manifest_must_be_object")
    digest = _sha256(csv_path)
    _validate_dataset_manifest(manifest, digest)
    data = _read_validated_dataset_csv(csv_path)
    return data, manifest, digest


def _build_teacher_features(
    candles: DataFrame,
) -> tuple[
    DataFrame,
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    list[int],
]:
    from research.ai_ensemble.features import build_causal_features
    from research.ai_ensemble.labels import build_net_profit_targets

    raw_features = build_causal_features(candles)
    positions = np.flatnonzero(
        np.isfinite(raw_features.to_numpy(dtype=np.float64)).all(axis=1)
    )
    if len(positions) < 2_000:
        raise ValueError("rl_causal_feature_rows_insufficient")
    features = raw_features.iloc[positions].reset_index(drop=True)
    position_map = {int(original): index for index, original in enumerate(positions)}
    targets = build_net_profit_targets(candles)
    labels = np.zeros(len(features), dtype=np.int64)
    label_ends = np.full(len(features), -1, dtype=np.int64)
    original_label_ends = np.full(len(features), -1, dtype=np.int64)
    eligible: list[int] = []
    for model_index, original_index in enumerate(positions):
        record = targets.iloc[int(original_index)]
        raw_target = record["target"]
        raw_end = record["label_end_idx"]
        raw_return = record["net_return"]
        if pd.isna(raw_target) or pd.isna(raw_end) or pd.isna(raw_return):
            continue
        end_index = int(raw_end)
        mapped_end = position_map.get(end_index)
        if mapped_end is None or mapped_end < model_index:
            continue
        labels[model_index] = int(raw_target)
        label_ends[model_index] = mapped_end
        original_label_ends[model_index] = end_index
        eligible.append(model_index)
    return features, positions, labels, label_ends, original_label_ends, eligible


def _select_teacher_training_indices(
    positions: NDArray[np.int64],
    original_label_ends: NDArray[np.int64],
    eligible: list[int],
    teacher_cutoff: int,
) -> list[int]:
    selected = [
        index
        for index in eligible
        if positions[index] < teacher_cutoff
        and original_label_ends[index] >= 0
        and original_label_ends[index] < teacher_cutoff - EMBARGO_BARS
    ]
    if len(selected) < 500:
        raise ValueError("rl_teacher_training_samples_insufficient")
    return selected


def _validate_teacher_training_labels(
    labels: NDArray[np.int64], training_indices: list[int]
) -> NDArray[np.int64]:
    selected_labels = labels[training_indices]
    if len(np.unique(selected_labels)) != 2:
        raise ValueError("rl_teacher_training_requires_both_classes")
    return selected_labels


def _fit_stacking_teacher(
    features: DataFrame,
    labels: NDArray[np.int64],
    label_ends: NDArray[np.int64],
    training_indices: list[int],
) -> Any:
    from research.ai_ensemble.ensemble import StackingConfig, StackingEnsemble

    config = StackingConfig(
        fold_count=3,
        embargo_bars=EMBARGO_BARS,
        minimum_train_samples=120,
        n_estimators=160,
        random_state=17,
        probability_threshold=PRIMARY_PROBABILITY_THRESHOLD,
    )
    return StackingEnsemble(config).fit(
        features,
        labels.tolist(),
        label_ends.tolist(),
        training_indices,
    )


def _predict_primary_oos(
    teacher: Any,
    features: DataFrame,
    positions: NDArray[np.int64],
    teacher_cutoff: int,
) -> tuple[NDArray[np.float64], int]:
    first_oos_feature = int(np.searchsorted(positions, teacher_cutoff))
    if first_oos_feature >= len(features) - 100:
        raise ValueError("rl_teacher_oos_window_insufficient")
    probabilities = np.full(len(features), np.nan, dtype=np.float64)
    probabilities[first_oos_feature:] = teacher.predict_proba(
        features.iloc[first_oos_feature:]
    )
    if not np.isfinite(probabilities[first_oos_feature:]).all():
        raise ValueError("rl_primary_oos_probabilities_non_finite")
    return probabilities, first_oos_feature


def _build_teacher_audit(
    positions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    original_label_ends: NDArray[np.int64],
    training_indices: list[int],
    selected_labels: NDArray[np.int64],
    teacher_cutoff: int,
    first_oos_feature: int,
) -> dict[str, Any]:
    teacher_max_label_end_row = int(
        max(original_label_ends[index] for index in training_indices)
    )
    if teacher_max_label_end_row >= teacher_cutoff - EMBARGO_BARS:
        raise ValueError("rl_teacher_label_crosses_training_boundary")
    return {
        "model": "LightGBM+XGBoost+causal-HMM+LogisticRegression",
        "probability_threshold": PRIMARY_PROBABILITY_THRESHOLD,
        "teacher_fit_fraction": TEACHER_FIT_FRACTION,
        "teacher_cutoff_row": teacher_cutoff,
        "teacher_training_samples": len(training_indices),
        "teacher_training_first_row": int(positions[training_indices[0]]),
        "teacher_training_last_row": int(positions[training_indices[-1]]),
        "teacher_max_label_end_row": teacher_max_label_end_row,
        "required_embargo_bars": EMBARGO_BARS,
        "oos_start_row": int(positions[first_oos_feature]),
        "oos_rows": len(probabilities) - first_oos_feature,
        "oos_signal_count": int(
            (probabilities[first_oos_feature:] >= PRIMARY_PROBABILITY_THRESHOLD).sum()
        ),
        "oos_mean_probability": float(np.mean(probabilities[first_oos_feature:])),
        "training_label_classes": sorted(
            int(value) for value in np.unique(selected_labels)
        ),
    }


def _fit_frozen_primary_model(
    candles: DataFrame,
) -> tuple[Any, DataFrame, NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
    features, positions, labels, label_ends, original_label_ends, eligible = (
        _build_teacher_features(candles)
    )
    teacher_cutoff = int(len(candles) * TEACHER_FIT_FRACTION)
    training_indices = _select_teacher_training_indices(
        positions, original_label_ends, eligible, teacher_cutoff
    )
    selected_labels = _validate_teacher_training_labels(labels, training_indices)
    teacher = _fit_stacking_teacher(features, labels, label_ends, training_indices)
    probabilities, first_oos_feature = _predict_primary_oos(
        teacher, features, positions, teacher_cutoff
    )
    audit = _build_teacher_audit(
        positions,
        probabilities,
        original_label_ends,
        training_indices,
        selected_labels,
        teacher_cutoff,
        first_oos_feature,
    )
    return teacher, features, positions, probabilities, audit


def _make_market_segment(
    candles: DataFrame,
    features: DataFrame,
    positions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    start_index: int,
    end_index: int,
) -> MarketSegment:
    selected = np.flatnonzero((positions >= start_index) & (positions < end_index))
    if len(selected) < 3:
        raise ValueError("rl_temporal_segment_too_short")
    original_positions = positions[selected]
    bars = candles.iloc[original_positions]
    timestamps = bars["date"].astype("int64").to_numpy(dtype=np.int64) // 1_000_000
    market = np.column_stack(
        (
            timestamps,
            bars["open"].to_numpy(dtype=np.float64),
            bars["high"].to_numpy(dtype=np.float64),
            bars["low"].to_numpy(dtype=np.float64),
            bars["close"].to_numpy(dtype=np.float64),
            bars["volume"].to_numpy(dtype=np.float64),
        )
    )
    close = candles["close"].astype(float)
    returns = np.column_stack(
        (
            close.pct_change(1, fill_method=None).fillna(0.0).clip(-1.0, 1.0),
            close.pct_change(6, fill_method=None).fillna(0.0).clip(-1.0, 1.0),
            close.pct_change(24, fill_method=None).fillna(0.0).clip(-1.0, 1.0),
        )
    )[original_positions]
    atr = features["atr_fraction"].to_numpy(dtype=np.float64)[selected]
    probability = probabilities[selected]
    if not np.isfinite(probability).all():
        raise ValueError("rl_oos_probability_missing_from_temporal_segment")
    signal = probability >= PRIMARY_PROBABILITY_THRESHOLD
    return MarketSegment(
        market=market.astype(np.float64),
        atr_fraction=atr,
        recent_returns=returns.astype(np.float64),
        primary_signal=signal.astype(np.bool_),
        signal_probability=probability.astype(np.float64),
        start_index=start_index,
        end_index=end_index,
    )


def _environment(segment: MarketSegment, *, training: bool) -> RiskAllocationEnv:
    return RiskAllocationEnv(
        segment.market,
        segment.atr_fraction,
        segment.recent_returns,
        segment.primary_signal,
        segment.signal_probability,
        bar_interval_ms=BAR_INTERVAL_MS,
        random_start=training,
        episode_steps=TRAINING_EPISODE_STEPS if training else None,
    )


def _evaluate_policy(
    model: Any | None, segment: MarketSegment, *, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = _environment(segment, training=False)
    observation, _ = env.reset(seed=seed)
    terminated = False
    decisions: list[dict[str, Any]] = []
    while not terminated:
        if model is None:
            signal = bool(observation[PRIMARY_SIGNAL_OBSERVATION_INDEX] >= 0.5)
            action = 1 if signal else 0
        else:
            predicted, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(predicted).item())
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("rl_oos_evaluation_unexpected_truncation")
        if info["signal_episode_started"]:
            decisions.append(
                {
                    "fill_timestamp_ms": info["timestamp_ms"],
                    "signal_probability": info["baseline_signal_probability"],
                    "decision": ACTION_NAMES[action],
                    "accepted": info["signal_episode_taken"],
                    "reward_on_entry_step": reward,
                    "drawdown_at_decision": info["current_drawdown"],
                }
            )
    metrics = env.get_metrics()
    env.close()
    return metrics, decisions


def _train_ppo(
    segment: MarketSegment,
    *,
    timesteps: int,
    seed: int,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    if isinstance(timesteps, bool) or timesteps < PPO_MINIMUM_TIMESTEPS:
        raise ValueError("rl_training_timesteps_must_be_at_least_200000")
    sb3 = importlib.import_module("stable_baselines3")
    vec_module = importlib.import_module("stable_baselines3.common.vec_env")
    dummy_vec_env = getattr(vec_module, "DummyVecEnv")
    ppo_class = getattr(sb3, "PPO")
    vector_env = dummy_vec_env([lambda: _environment(segment, training=True)])
    model = ppo_class(
        "MlpPolicy",
        vector_env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(output_dir / "ppo-risk-manager"))
    vector_env.close()
    return model, {
        "algorithm": "PPO",
        "stable_baselines3_version": sb3.__version__,
        "policy": "MlpPolicy",
        "device": "cpu",
        "total_timesteps": timesteps,
        "seed": seed,
        "episode_steps": TRAINING_EPISODE_STEPS,
        "n_steps": 512,
        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coefficient": 0.02,
        "minimum_required_timesteps": PPO_MINIMUM_TIMESTEPS,
        "model_file": "ppo-risk-manager.zip",
    }


def _promotion_decision(
    validation_baseline: dict[str, Any],
    validation_rl: dict[str, Any],
    holdout_baseline: dict[str, Any],
    holdout_rl: dict[str, Any],
) -> tuple[str, dict[str, bool]]:
    checks = {
        "validation_minimum_trade_support": min(
            validation_baseline["trade_count"], validation_rl["trade_count"]
        )
        >= MINIMUM_OOS_TRADES,
        "holdout_minimum_trade_support": min(
            holdout_baseline["trade_count"], holdout_rl["trade_count"]
        )
        >= MINIMUM_OOS_TRADES,
        "validation_net_pf_at_least_1_1": (
            validation_rl["net_profit_factor"] is not None
            and validation_rl["net_profit_factor"] >= 1.1
        ),
        "holdout_net_pf_at_least_1_1": (
            holdout_rl["net_profit_factor"] is not None
            and holdout_rl["net_profit_factor"] >= 1.1
        ),
        "validation_net_pf_improved": (
            validation_baseline["net_profit_factor"] is not None
            and validation_rl["net_profit_factor"] is not None
            and validation_rl["net_profit_factor"]
            > validation_baseline["net_profit_factor"]
        ),
        "holdout_net_pf_improved": (
            holdout_baseline["net_profit_factor"] is not None
            and holdout_rl["net_profit_factor"] is not None
            and holdout_rl["net_profit_factor"] > holdout_baseline["net_profit_factor"]
        ),
        "validation_drawdown_below_15_percent": validation_rl["max_drawdown"] < 0.15,
        "holdout_drawdown_below_15_percent": holdout_rl["max_drawdown"] < 0.15,
        "validation_drawdown_improved": (
            validation_rl["max_drawdown"] < validation_baseline["max_drawdown"]
        ),
        "holdout_drawdown_improved": (
            holdout_rl["max_drawdown"] < holdout_baseline["max_drawdown"]
        ),
        "holdout_positive_statistical_z_score": (
            holdout_rl["statistical_z_score"] is not None
            and holdout_rl["statistical_z_score"] > 0.0
        ),
    }
    status = "GO_FOR_OWNER_REVIEW" if all(checks.values()) else "NO-GO"
    return status, checks


def run_training(
    *,
    input_dir: Path,
    output_dir: Path,
    commit_sha: str,
    timesteps: int = PPO_DEFAULT_TIMESTEPS,
    seed: int = 42,
) -> dict[str, Any]:
    _validate_commit_sha(commit_sha)
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("rl_seed_must_be_non_negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    candles, dataset_manifest, dataset_digest = _load_validated_dataset(input_dir)
    teacher, features, positions, probabilities, teacher_audit = (
        _fit_frozen_primary_model(candles)
    )

    teacher_cutoff = teacher_audit["teacher_cutoff_row"]
    ppo_train_end = int(len(candles) * PPO_TRAIN_END_FRACTION)
    validation_end = int(len(candles) * VALIDATION_END_FRACTION)
    train_segment = _make_market_segment(
        candles, features, positions, probabilities, teacher_cutoff, ppo_train_end
    )
    validation_segment = _make_market_segment(
        candles, features, positions, probabilities, ppo_train_end, validation_end
    )
    holdout_segment = _make_market_segment(
        candles, features, positions, probabilities, validation_end, len(candles)
    )
    if not (
        train_segment.end_index <= validation_segment.start_index
        and validation_segment.end_index <= holdout_segment.start_index
    ):
        raise ValueError("rl_temporal_segments_overlap")
    if int(teacher_audit["teacher_max_label_end_row"]) >= teacher_cutoff - EMBARGO_BARS:
        raise ValueError("rl_primary_model_temporal_leakage")

    ppo_model, ppo_config = _train_ppo(
        train_segment, timesteps=timesteps, seed=seed, output_dir=output_dir
    )
    validation_baseline, validation_baseline_decisions = _evaluate_policy(
        None, validation_segment, seed=seed + 1
    )
    validation_rl, validation_rl_decisions = _evaluate_policy(
        ppo_model, validation_segment, seed=seed + 1
    )
    holdout_baseline, holdout_baseline_decisions = _evaluate_policy(
        None, holdout_segment, seed=seed + 2
    )
    holdout_rl, holdout_rl_decisions = _evaluate_policy(
        ppo_model, holdout_segment, seed=seed + 2
    )
    decision, promotion_checks = _promotion_decision(
        validation_baseline, validation_rl, holdout_baseline, holdout_rl
    )

    with (output_dir / "frozen-primary-ensemble.pkl").open("wb") as stream:
        pickle.dump(teacher, stream, protocol=pickle.HIGHEST_PROTOCOL)
    oos_start = int(np.searchsorted(positions, teacher_cutoff))
    signal_timeline = DataFrame(
        {
            "source_row": positions[oos_start:],
            "timestamp": candles["date"]
            .iloc[positions[oos_start:]]
            .astype(str)
            .to_numpy(),
            "primary_signal_probability": probabilities[oos_start:],
            "primary_buy_signal": (
                probabilities[oos_start:] >= PRIMARY_PROBABILITY_THRESHOLD
            ),
        }
    )
    signal_timeline.to_csv(output_dir / "primary-oos-signals.csv", index=False)
    DataFrame(validation_baseline_decisions).to_csv(
        output_dir / "validation-baseline-decisions.csv", index=False
    )
    DataFrame(validation_rl_decisions).to_csv(
        output_dir / "validation-ppo-decisions.csv", index=False
    )
    DataFrame(holdout_baseline_decisions).to_csv(
        output_dir / "holdout-baseline-decisions.csv", index=False
    )
    DataFrame(holdout_rl_decisions).to_csv(
        output_dir / "holdout-ppo-decisions.csv", index=False
    )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETED_RESEARCH_EVALUATION",
        "commit_sha": commit_sha,
        "source_dataset": {
            "producer_commit_sha": dataset_manifest["commit_sha"],
            "source_policy": EXPECTED_SOURCE_POLICY,
            "provider": EXPECTED_PROVIDER,
            "symbol": "BTCUSDT",
            "interval": "1h",
            "row_count": len(candles),
            "start_timestamp": EXPECTED_START_TIMESTAMP,
            "end_timestamp": EXPECTED_END_TIMESTAMP,
            "csv_sha256": dataset_digest,
            "manifest_sha256": _sha256(input_dir / MANIFEST_NAME),
            "private_api_calls": 0,
            "exchange_calls": 0,
            "nobitex_api_calls_during_rl_run": 0,
            "external_backfill_used": False,
        },
        "primary_signal": {
            **teacher_audit,
            "signal_model_status": "FROZEN_BEFORE_PPO_TRAIN_VALIDATION_AND_HOLDOUT",
            "probability_artifact": "primary-oos-signals.csv",
            "model_artifact": "frozen-primary-ensemble.pkl",
        },
        "temporal_split": {
            "teacher_fit_fraction": TEACHER_FIT_FRACTION,
            "ppo_training_fraction_end": PPO_TRAIN_END_FRACTION,
            "validation_fraction_end": VALIDATION_END_FRACTION,
            "final_holdout_fraction": 1.0 - VALIDATION_END_FRACTION,
            "teacher_end_row": teacher_cutoff,
            "ppo_training_start_row": teacher_cutoff,
            "ppo_training_end_row": ppo_train_end,
            "validation_start_row": ppo_train_end,
            "validation_end_row": validation_end,
            "holdout_start_row": validation_end,
            "holdout_end_row_exclusive": len(candles),
            "purge_embargo_bars": EMBARGO_BARS,
            "holdout_used_for_training_or_tuning": False,
            "validation_used_for_training_or_tuning": False,
        },
        "minimum_oos_trades_required": MINIMUM_OOS_TRADES,
        "risk_environment": {
            "action_type": "DISCRETE_BASELINE_SIGNAL_EPISODE_FILTER",
            "action_space_size": 2,
            "actions": list(ACTION_NAMES),
            "actions_target_exposure_fraction": list(ACTION_FRACTIONS),
            "one_decision_per_contiguous_signal_episode": True,
            "decision_latched_until_signal_turns_off": True,
            "force_flat_when_primary_signal_off": True,
            "directional_actions_available": False,
            "short_or_leverage_allowed": False,
            "no_primary_signal_policy": "FORCE_FLAT_AT_NEXT_HOURLY_OPEN",
            "observation_fields": list(OBSERVATION_FIELDS),
            "execution_timing": "DECISION_AFTER_COMPLETED_HOURLY_CLOSE; FILL_AT_NEXT_HOURLY_OPEN; EXIT_ON_FIRST_SIGNAL_OFF",
            "drawdown_limit_fraction": DRAWDOWN_LIMIT,
            "drawdown_penalty_coefficients": {
                "incremental_drawdown": DRAWDOWN_INCREMENT_PENALTY,
                "linear_excess_above_15_percent": DRAWDOWN_EXCESS_PENALTY,
                "quadratic_excess_above_15_percent": DRAWDOWN_SEVERE_PENALTY,
            },
            "participation_bonus": TAKE_SIGNAL_REWARD_BONUS,
            "participation_bonus_semantics": "ONE_SHOT_PER_ACCEPTED_SIGNAL_EPISODE; REWARD_SHAPING_ONLY_EXCLUDED_FROM_EQUITY_AND_PNL",
            "participation_bonus_count_validation": "AUDITED_IN_ENVIRONMENT_METRICS",
            "inactivity_penalty": 0.0,
            "reward": "RUNNING_SORTINO_STYLE_NET_LOG_RETURN_PLUS_ONE_SHOT_TAKE_BONUS_MINUS_SCALED_DRAWDOWN_PENALTIES",
            "trade_count_definition": "COMPLETED_TAKEN_CONTIGUOUS_PRIMARY_SIGNAL_EPISODES",
        },
        "execution_costs": {
            "maker_fee_bps_per_side": 25.0,
            "slippage_bps_per_side": 0.0,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "ppo": ppo_config,
        "validation": {
            "baseline": validation_baseline,
            "ppo_risk_manager": validation_rl,
        },
        "final_holdout": {
            "baseline": holdout_baseline,
            "ppo_risk_manager": holdout_rl,
        },
        "promotion_checks": promotion_checks,
        "paper_trading_decision": decision,
        "safety": {
            "ALIX_DRY_RUN": True,
            "ALIX_ALLOW_LIVE_TRADING": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "nobitex_api_calls": 0,
            "orders_or_execution_api_used": False,
        },
    }
    report_path = output_dir / "ai6-ppo-risk-report.json"
    report_path.write_text(
        json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--timesteps", type=int, default=PPO_DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_training(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        commit_sha=args.commit_sha,
        timesteps=args.timesteps,
        seed=args.seed,
    )
    output = {
        "status": report["status"],
        "commit_sha": report["commit_sha"],
        "round_trip_cost_bps": report["execution_costs"]["round_trip_cost_bps"],
        "validation": report["validation"],
        "final_holdout": report["final_holdout"],
        "paper_trading_decision": report["paper_trading_decision"],
        "safety": report["safety"],
    }
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
