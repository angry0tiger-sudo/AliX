"""Regime-aware Freqtrade strategy with fail-closed model and selection contracts.

Entry signals are research candidates only. HMM/Meta artifacts require review, and
an upstream accepted selector result is required before any entry signal is emitted.
All real order authorization remains outside this strategy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.data_quality.gate import (
    QualityReasonCode,
    QualityResult,
    QualityStatus,
    validate_candles,
)
from services.freqai_baseline.features import FEATURE_COLUMNS, add_market_features
from services.meta_labeling.meta_model import MetaLabelModel
from services.strategy_selection.model_io import load_hmm_model_file
from services.strategy_selection.regime import (
    HMMRegimeClassifier,
    PreviousRegimePosterior,
    RegimeInferenceStatus,
    RegimeObservation,
)

try:
    from freqtrade.strategy import IStrategy  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only used by lightweight tooling.

    class IStrategy:  # type: ignore[no-redef]
        """Static-check fallback for environments without Freqtrade."""


_TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}
_HMM_MODEL_PATH = Path("user_data/models/hmm-regime-model.json")
_META_MODEL_PATH = Path("user_data/models/meta-label-model.json")
_META_MANIFEST_PATH = Path("user_data/models/meta-label-model.manifest.json")
_HMM_APPROVAL_PATH = Path("config/strategy_selection/hmm-model-approval.json")
_META_APPROVAL_PATH = Path("config/strategy_selection/meta-model-approval.json")
_META_THRESHOLD = 0.70
_BREAKEVEN_BAND = 0.0025


def _aware_timestamp(value: object) -> datetime | None:
    try:
        parsed = pd.Timestamp(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.to_pydatetime().astimezone(timezone.utc)


def _row_fingerprint(row: dict[str, object], previous_hash: Any) -> str:
    content = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    next_hash = previous_hash.copy()
    next_hash.update(content.encode("utf-8"))
    next_hash.update(b"\n")
    return cast(str, next_hash.hexdigest())


def _add_technical_indicators(dataframe: DataFrame) -> DataFrame:
    result = dataframe.copy()
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    close = pd.to_numeric(result["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    result["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    relative_strength = gains / losses.where(losses > 0)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    result["rsi_14"] = rsi.where(losses > 0, 100.0).where(gains.notna())
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    result["bb_middle"] = middle
    result["bb_upper"] = middle + (2.0 * deviation)
    result["bb_lower"] = middle - (2.0 * deviation)
    return result


def _read_meta_documents() -> (
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None
):
    try:
        raw = json.loads(_META_MODEL_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(_META_MANIFEST_PATH.read_text(encoding="utf-8"))
        approval = json.loads(_META_APPROVAL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(manifest, dict):
        return None
    if not isinstance(approval, dict):
        return None
    return raw, manifest, approval


def _parse_meta_approval_timestamp(value: object) -> pd.Timestamp | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return None
    return timestamp


def _meta_manifest_matches_model(
    manifest: dict[str, Any], digest: str
) -> bool:
    dataset_hash = manifest.get("dataset_sha256")
    if manifest.get("model_sha256") != digest or manifest.get("dataset_rows") != 8760:
        return False
    return isinstance(dataset_hash, str) and len(dataset_hash) == 64


def _meta_manifest_release_is_closed(manifest: dict[str, Any]) -> bool:
    return (
        manifest.get("economic_acceptance") == "NO-GO"
        and manifest.get("trading_authorization") == "NO-GO"
    )


def _meta_approval_identity_matches(approval: dict[str, Any]) -> bool:
    approval_id = approval.get("approval_id")
    return (
        approval.get("schema_version") == "alix.model-technical-approval.v1"
        and approval.get("model_type") == "meta_label_logistic"
        and approval.get("status") == "OWNER_APPROVED_TECHNICAL_ONLY"
        and approval.get("owner") == "angry0tiger-sudo"
        and isinstance(approval_id, str)
        and bool(approval_id.strip())
    )


def _meta_approval_hashes_match(
    approval: dict[str, Any], digest: str, dataset_hash: object
) -> bool:
    return (
        approval.get("model_sha256") == digest
        and approval.get("dataset_sha256") == dataset_hash
    )


def _meta_approval_release_is_closed(approval: dict[str, Any]) -> bool:
    return (
        approval.get("economic_acceptance") == "NO-GO"
        and approval.get("order_permission") is False
        and approval.get("trading_authorization") == "NO-GO"
    )


def _meta_approval_is_valid(
    approval: dict[str, Any], digest: str, dataset_hash: object
) -> bool:
    approved_at = _parse_meta_approval_timestamp(approval.get("approved_at"))
    return (
        _meta_approval_identity_matches(approval)
        and _meta_approval_hashes_match(approval, digest, dataset_hash)
        and approved_at is not None
        and _meta_approval_release_is_closed(approval)
    )


def _load_meta_model() -> tuple[MetaLabelModel | None, bool]:
    documents = _read_meta_documents()
    if documents is None:
        return None, False
    raw, manifest, approval = documents
    try:
        model = MetaLabelModel.from_dict(raw)
        digest = hashlib.sha256(model.to_json().encode("utf-8")).hexdigest()
    except (ValueError, TypeError, KeyError):
        return None, False
    dataset_hash = manifest.get("dataset_sha256")
    if not _meta_manifest_matches_model(manifest, digest):
        return None, False
    if not _meta_manifest_release_is_closed(manifest):
        return None, False
    return model, _meta_approval_is_valid(approval, digest, dataset_hash)


def _empty_regime_columns(dataframe: DataFrame) -> DataFrame:
    dataframe["alix_regime"] = "NO_TRADE"
    dataframe["alix_regime_confidence"] = np.nan
    dataframe["alix_hmm_revision"] = None
    dataframe["alix_hmm_valid"] = False
    dataframe["alix_meta_probability"] = np.nan
    dataframe["alix_meta_model_approved"] = False
    dataframe["alix_signal_bundle_approved"] = False
    return dataframe


def _infer_regimes(dataframe: DataFrame) -> DataFrame:
    """Apply the pinned model sequentially without using later candles as features."""
    try:
        model = load_hmm_model_file(_HMM_MODEL_PATH, approval_path=_HMM_APPROVAL_PATH)
    except (ValueError, OSError):
        return dataframe
    classifier = HMMRegimeClassifier()
    regime_features = dataframe.loc[
        :, ["directional_efficiency", "realized_volatility"]
    ]
    required = ("date", "open", "high", "low", "close", "volume")
    rolling_hash = hashlib.sha256()
    previous: PreviousRegimePosterior | None = None
    regimes: list[str] = []
    confidence_values: list[float] = []
    revisions: list[str | None] = []
    valid_values: list[bool] = []
    for position, (_, row) in enumerate(dataframe.iterrows()):
        candle = {name: row[name] for name in required}
        prefix_hash = _row_fingerprint(candle, rolling_hash)
        rolling_hash.update(
            json.dumps(
                candle, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
        rolling_hash.update(b"\n")
        close_time = _aware_timestamp(row["date"])
        if close_time is None:
            regimes.append("NO_TRADE")
            confidence_values.append(float("nan"))
            revisions.append(None)
            valid_values.append(False)
            previous = None
            continue
        close_time += timedelta(seconds=_TIMEFRAME_SECONDS["1h"])
        if position >= len(regime_features):
            regimes.append("NO_TRADE")
            confidence_values.append(float("nan"))
            revisions.append(None)
            valid_values.append(False)
            continue
        values = regime_features.iloc[position]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            regimes.append("NO_TRADE")
            confidence_values.append(float("nan"))
            revisions.append(None)
            valid_values.append(False)
            continue
        quality = QualityResult(
            passed=True,
            reasons=(),
            status=QualityStatus.VALID,
            reason_codes=(),
            fingerprint=prefix_hash,
        )
        observation = RegimeObservation(
            observed_at=close_time,
            evaluated_at=close_time,
            symbol="BTC/USDT",
            timeframe="1h",
            feature_schema_version=model.feature_schema_version,
            feature_values=(
                (
                    "directional_efficiency",
                    Decimal(str(float(values["directional_efficiency"]))),
                ),
                (
                    "realized_volatility",
                    Decimal(str(float(values["realized_volatility"]))),
                ),
            ),
            data_quality=quality,
            data_fingerprint=prefix_hash,
        )
        result = classifier.infer(
            model=model, observation=observation, previous=previous
        )
        if result.status is RegimeInferenceStatus.CLASSIFIED:
            regimes.append(str(result.selected_regime))
            confidence_values.append(float(result.confidence or Decimal("0")))
            revisions.append(result.model_revision)
            valid_values.append(True)
            previous = PreviousRegimePosterior(
                model_revision=model.revision,
                as_of=close_time,
                probabilities=tuple(
                    Decimal(str(dict(result.posterior)[name])) for name in model.regimes
                ),
                fingerprint=result.fingerprint,
            )
        else:
            regimes.append("NO_TRADE")
            confidence_values.append(float("nan"))
            revisions.append(result.model_revision)
            valid_values.append(False)
            previous = None
    dataframe["alix_regime"] = regimes
    dataframe["alix_regime_confidence"] = confidence_values
    dataframe["alix_hmm_revision"] = revisions
    dataframe["alix_hmm_valid"] = valid_values
    return dataframe


def _strategy_data_quality(dataframe: DataFrame, timeframe: str) -> QualityResult:
    interval_seconds = _TIMEFRAME_SECONDS.get(timeframe)
    if interval_seconds is None:
        return QualityResult(
            False,
            ("unsupported_timeframe",),
            QualityStatus.REJECTED,
            (QualityReasonCode.UNSUPPORTED_TIMEFRAME,),
        )
    rows = [
        {str(key): value for key, value in row.items()}
        for row in dataframe.to_dict(orient="records")
    ]
    return validate_candles(
        rows,
        expected_interval_seconds=interval_seconds,
        gap_policy="reject",
    )


def _fail_closed(dataframe: DataFrame) -> DataFrame:
    dataframe["alix_data_quality_valid"] = False
    _empty_regime_columns(dataframe)
    dataframe["alix_selection_status"] = "NO_TRADE"
    return dataframe


def _apply_model_features(dataframe: DataFrame) -> bool:
    try:
        engineered = add_market_features(dataframe)
        for name in FEATURE_COLUMNS:
            dataframe[name] = engineered[name]
        from services.strategy_selection.hmm_training import calculate_regime_features

        regime_features = calculate_regime_features(dataframe)
        dataframe["directional_efficiency"] = regime_features["directional_efficiency"]
        dataframe["realized_volatility"] = regime_features["realized_volatility"]
    except (ValueError, KeyError, TypeError):
        return False
    return True


def _score_meta_candidates(dataframe: DataFrame) -> DataFrame:
    model, model_approved = _load_meta_model()
    if model is None:
        return dataframe
    feature_frame = dataframe.loc[:, list(FEATURE_COLUMNS)]
    try:
        dataframe["alix_meta_probability"] = model.predict_probability(feature_frame)
        dataframe["alix_meta_model_approved"] = model_approved
    except (ValueError, TypeError):
        dataframe["alix_meta_probability"] = np.nan
        dataframe["alix_meta_model_approved"] = False
    return dataframe


def _attach_selector_gate(dataframe: DataFrame) -> DataFrame:
    if "alix_selected_strategy_id" not in dataframe:
        dataframe["alix_selected_strategy_id"] = None
    if "alix_selection_status" not in dataframe:
        dataframe["alix_selection_status"] = "NO_TRADE"
    dataframe["alix_signal_bundle_approved"] = (
        dataframe["alix_hmm_valid"].astype(bool)
        & dataframe["alix_meta_model_approved"].astype(bool)
        & dataframe["alix_selection_status"].eq("SELECTED")
        & dataframe["alix_selected_strategy_id"].isin(
            ["trend_following", "mean_reversion"]
        )
        & dataframe["alix_data_quality_valid"].astype(bool)
    )
    return dataframe


def _read_atr_stop_inputs(
    strategy: object, pair: str, trade: Any
) -> tuple[float, float, str] | None:
    data_provider = getattr(strategy, "dp", None)
    getter = getattr(data_provider, "get_analyzed_dataframe", None)
    if not callable(getter):
        return None
    try:
        analyzed, _ = getter(pair, getattr(strategy, "timeframe", "1h"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(analyzed, DataFrame) or analyzed.empty:
        return None
    row = analyzed.iloc[-1]
    try:
        atr = float(row["atr_14"])
        entry_rate = float(trade.open_rate)
        regime = str(row.get("alix_regime", "NO_TRADE"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in (atr, entry_rate)):
        return None
    return atr, entry_rate, regime


def _calculate_atr_stop(
    current_rate: float, atr: float, entry_rate: float, regime: str
) -> float:
    multiplier = {"trend": 2.0, "range": 1.0, "high_volatility": 0.75}.get(regime, 0.75)
    stop_rate = entry_rate - multiplier * atr
    if not math.isfinite(stop_rate) or stop_rate <= 0:
        return -0.01
    relative_stop = (stop_rate / current_rate) - 1.0
    if not math.isfinite(relative_stop) or relative_stop >= 0:
        return -0.001
    return max(relative_stop, -0.99)


class AliXStrategy(IStrategy):  # type: ignore[misc]
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi: dict[int, float] = {}
    # Catastrophic fallback only; the operative stop is returned by custom_stoploss.
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 100

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        """Validate OHLCV, derive causal features and load reviewed AI artifacts."""
        result = _strategy_data_quality(dataframe, self.timeframe)
        dataframe.attrs["alix_data_quality"] = {
            "status": result.status.value if result.status is not None else "REJECTED",
            "reason_codes": tuple(code.value for code in result.reason_codes),
            "fingerprint": result.fingerprint,
        }
        if not result.passed:
            return _fail_closed(dataframe)
        dataframe["alix_data_quality_valid"] = True
        dataframe = _add_technical_indicators(dataframe)
        if not _apply_model_features(dataframe):
            return _fail_closed(dataframe)
        dataframe = _empty_regime_columns(dataframe)
        dataframe = _infer_regimes(dataframe)
        dataframe = _score_meta_candidates(dataframe)
        return _attach_selector_gate(dataframe)

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        """Emit a candidate only after HMM, Meta-Model and selector agree."""
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        required = {
            "alix_data_quality_valid",
            "alix_signal_bundle_approved",
            "alix_regime",
            "alix_selected_strategy_id",
            "alix_meta_probability",
            "ema_20",
            "ema_50",
            "rsi_14",
            "bb_lower",
            "close",
        }
        if not required.issubset(dataframe.columns):
            return dataframe
        safe = (
            dataframe["alix_data_quality_valid"].astype(bool)
            & dataframe["alix_signal_bundle_approved"].astype(bool)
            & (dataframe["alix_meta_probability"] >= _META_THRESHOLD)
        )
        trend = (
            safe
            & dataframe["alix_regime"].eq("trend")
            & dataframe["alix_selected_strategy_id"].eq("trend_following")
            & (dataframe["close"] > dataframe["ema_20"])
            & (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["close"].shift(1) <= dataframe["ema_20"].shift(1))
        )
        mean_reversion = (
            safe
            & dataframe["alix_regime"].eq("range")
            & dataframe["alix_selected_strategy_id"].eq("mean_reversion")
            & (dataframe["rsi_14"] < 30.0)
            & (dataframe["close"] <= dataframe["bb_lower"])
        )
        dataframe.loc[trend | mean_reversion, "enter_long"] = 1
        dataframe.loc[trend, "enter_tag"] = "hmm_trend_following_meta70"
        dataframe.loc[mean_reversion, "enter_tag"] = "hmm_mean_reversion_meta70"
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        """Keep a model-independent technical exit path for open long positions."""
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = None
        required = {"close", "ema_20", "rsi_14"}
        if not required.issubset(dataframe.columns):
            return dataframe
        close_below_trend = dataframe["close"] < dataframe["ema_20"]
        momentum_reversal = dataframe["rsi_14"] >= 60.0
        dataframe.loc[close_below_trend | momentum_reversal, "exit_long"] = 1
        dataframe.loc[close_below_trend, "exit_tag"] = "technical_trend_exit"
        dataframe.loc[momentum_reversal & ~close_below_trend, "exit_tag"] = (
            "momentum_exit"
        )
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
    ) -> float | None:
        """Set an ATR stop at 2x in trend, 1x in range, tighter in high volatility."""
        if not math.isfinite(current_rate) or current_rate <= 0:
            return -0.01
        stop_inputs = _read_atr_stop_inputs(self, pair, trade)
        if stop_inputs is None:
            return -0.01
        atr, entry_rate, regime = stop_inputs
        return _calculate_atr_stop(current_rate, atr, entry_rate, regime)

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        """Release a near-breakeven position after 12 full timeframe candles."""
        opened_at = getattr(trade, "open_date_utc", None)
        if not isinstance(opened_at, datetime):
            return None
        if opened_at.tzinfo is None or opened_at.utcoffset() is None:
            return None
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            return None
        elapsed = current_time.astimezone(timezone.utc) - opened_at.astimezone(
            timezone.utc
        )
        if elapsed < timedelta(seconds=12 * _TIMEFRAME_SECONDS.get(self.timeframe, 0)):
            return None
        if not math.isfinite(current_profit) or abs(current_profit) > _BREAKEVEN_BAND:
            return None
        return "time_decay_breakeven_12_candles"
