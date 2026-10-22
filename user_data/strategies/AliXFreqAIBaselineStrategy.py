"""FreqAI baseline strategy scaffold with a fail-closed quality boundary.

This strategy runs prediction plumbing for offline research only. It deliberately
emits no trade-entry or exit signals in any environment.
"""

from __future__ import annotations

from typing import Any

from pandas import DataFrame

from services.data_quality.gate import (
    QualityResult,
    QualityStatus,
    validate_candles,
)
from services.freqai_baseline.features import (
    TARGET_COLUMN,
    add_market_features,
    forward_return_target,
)

try:
    from freqtrade.strategy import IStrategy  # type: ignore[import-not-found]
except ImportError:  # Quality CI intentionally has no Freqtrade runtime dependency.

    class IStrategy:  # type: ignore[no-redef]
        pass


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


class AliXFreqAIBaselineStrategy(IStrategy):  # type: ignore[misc]
    """Versioned feature/target wiring; all trading signals remain disabled."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.0}
    stoploss = -0.10
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 100
    use_exit_signal = False

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs: Any
    ) -> DataFrame:
        """Produce % features using only current and historical candles."""
        return add_market_features(dataframe)

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs: Any
    ) -> DataFrame:
        """Build the & target separately; its future horizon is label-only."""
        freqai_info = getattr(self, "freqai_info", {})
        feature_parameters = freqai_info.get("feature_parameters", {})
        horizon = feature_parameters.get("label_period_candles", 12)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("FreqAI label_period_candles must be a positive integer")
        dataframe[TARGET_COLUMN] = forward_return_target(dataframe, horizon)
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        """Reject invalid data before FreqAI features or prediction are requested."""
        interval_seconds = _TIMEFRAME_SECONDS.get(self.timeframe)
        if interval_seconds is None:
            result = QualityResult(
                False,
                ("unsupported_timeframe",),
                QualityStatus.REJECTED,
            )
        else:
            rows = [
                {str(key): value for key, value in row.items()}
                for row in dataframe.to_dict(orient="records")
            ]
            result = validate_candles(
                rows,
                expected_interval_seconds=interval_seconds,
                gap_policy="reject",
            )
        dataframe["alix_data_quality_valid"] = result.passed
        dataframe.attrs["alix_data_quality"] = {
            "status": result.status.value if result.status is not None else "REJECTED",
            "reason_codes": tuple(code.value for code in result.reason_codes),
            "fingerprint": result.fingerprint,
        }
        if not result.passed:
            dataframe["do_predict"] = 0
            return dataframe
        freqai = getattr(self, "freqai", None)
        if freqai is None or not callable(getattr(freqai, "start", None)):
            dataframe["do_predict"] = 0
            dataframe.attrs["freqai_status"] = "unavailable_fail_closed"
            return dataframe
        result_frame = freqai.start(dataframe, metadata, self)
        result_frame["alix_data_quality_valid"] = True
        return result_frame

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = None
        return dataframe
