from __future__ import annotations

from datetime import timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pytest

from scripts import run_ml_core_v4aq_ai6 as ai6
from scripts import run_ml_core_v4ap_diagnostic_sweep as v4ap
from services.meta_labeling import v4x
from services.meta_labeling.v4aq import V4AQ_META_FEATURE_COLUMNS

UTC = timezone.utc
START = pd.Timestamp("2021-01-01T00:00:00Z")
END = START + pd.Timedelta(days=1825) - pd.Timedelta(minutes=1)
COMMIT = "a" * 40


def _funding_artifact(
    directory: Path, symbol: str, *, ready: bool = True
) -> dict[str, Any]:
    stem = f"bybit-{symbol.lower()}"
    csv_path = directory / f"{stem}-funding.csv"
    timestamps = pd.date_range(START, END, periods=20)
    rows = DataFrame(
        {
            "symbol": symbol,
            "funding_time": timestamps.astype(str),
            "funding_rate": np.linspace(-0.001, 0.001, len(timestamps)),
            "mark_price": [None] * len(timestamps),
            "available_at": (timestamps + pd.Timedelta(minutes=5)).astype(str),
            "content_hash": [f"{index:064x}" for index in range(len(timestamps))],
        },
        columns=ai6.FUNDING_CSV_COLUMNS,
    )
    rows.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "alix.bybit.linear-funding.v1",
        "commit_sha": COMMIT,
        "provider": "Bybit V5 public Linear Funding Rate History",
        "host": "api.bybit.com",
        "endpoint": "/v5/market/funding/history",
        "category": "linear",
        "page_limit": 200,
        "symbol": symbol,
        "requested_days": 1825,
        "requested_start_utc": START.isoformat(),
        "requested_end_utc": END.isoformat(),
        "public_api_only": True,
        "credentials_used": False,
        "api_key_header_sent": False,
        "private_api_calls": 0,
        "exchange_order_calls": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "funding": {
            "row_count": len(rows),
            "observed_span_days": 1825.0,
            "coverage_status": (
                "FULL_REQUESTED_HISTORY" if ready else "INSUFFICIENT_HISTORY"
            ),
            "full_requested_history_available": ready,
        },
        "funding_csv": csv_path.name,
        "funding_csv_sha256": digest,
        "research_status": (
            "READY_FOR_FUNDING_FEATURE_JOIN"
            if ready
            else "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
        ),
    }
    (directory / f"{stem}-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _bars(rows: int = 120, timeframe: str = "4h") -> DataFrame:
    frequency = {"15m": "15min", "1h": "1h", "4h": "4h"}[timeframe]
    close = 100.0 + np.arange(rows) * 0.05 + np.sin(np.arange(rows) / 3.0)
    return DataFrame(
        {
            "date": pd.date_range(START, periods=rows, freq=frequency),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )


def test_forward_labeling_applies_30bps_cost_and_rejects_gap_and_tail() -> None:
    bars = _bars(8, "4h")
    bars.loc[1, "open"] = 100.0
    bars.loc[1, "close"] = 101.0
    bars.loc[2, "open"] = 100.0
    bars.loc[2, "close"] = 99.0
    events = DataFrame({"event_index": [0, 1, 7]})
    labelled = ai6._label_primary_events(bars, events, timeframe="4h", horizon_bars=1)
    assert labelled["event_index"].tolist() == [0, 1]
    assert labelled["meta_label"].tolist() == [1, 0]
    net_returns = labelled["net_return_fraction"].to_numpy(dtype=float)
    assert net_returns[0] > 0.0
    assert net_returns[1] < 0.0

    discontinuous = bars.copy()
    discontinuous_dates = pd.to_datetime(discontinuous["date"], utc=True)
    discontinuous_dates.iloc[2] = discontinuous_dates.iloc[2] + pd.Timedelta(minutes=1)
    discontinuous["date"] = discontinuous_dates
    gap_labels = ai6._label_primary_events(
        discontinuous,
        DataFrame({"event_index": [0, 1]}),
        timeframe="4h",
        horizon_bars=1,
    )
    assert gap_labels["event_index"].tolist() == [0]


def test_candidate_acceptance_requires_all_economic_and_statistical_gates() -> None:
    candidate: dict[str, Any] = {
        "trade_count": 100,
        "win_rate": 0.55,
        "net_profit_factor": 1.25,
        "dsr_z_score": 0.1,
    }
    accepted = ai6._candidate_acceptance(candidate)
    assert accepted["statistical_edge_accepted"] is True

    candidate["dsr_z_score"] = None
    rejected = ai6._candidate_acceptance(candidate)
    assert rejected["net_profit_factor_passed"] is True
    assert rejected["dsr_passed"] is False
    assert rejected["statistical_edge_accepted"] is False

    candidate["dsr_z_score"] = 0.1
    candidate["trade_count"] = 49
    assert ai6._candidate_acceptance(candidate)["statistical_edge_accepted"] is False


def test_read_funding_artifact_verifies_provenance_hash_and_oi_absence(
    tmp_path: Path,
) -> None:
    manifest = _funding_artifact(tmp_path, "BTCUSDT")
    funding, received, ready = ai6._read_funding_artifact(
        tmp_path, "BTCUSDT", COMMIT, START, END
    )
    assert ready is True
    assert len(funding) == 20
    assert received == manifest

    manifest_path = tmp_path / "bybit-btcusdt-manifest.json"
    manifest["open_interest"] = {"row_count": 1}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="oi_contract_must_be_absent"):
        ai6._read_funding_artifact(tmp_path, "BTCUSDT", COMMIT, START, END)


def test_read_funding_artifact_rejects_hash_safety_and_window_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _funding_artifact(tmp_path, "BTCUSDT")
    csv_path = tmp_path / manifest["funding_csv"]
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="csv_hash_mismatch"):
        ai6._read_funding_artifact(tmp_path, "BTCUSDT", COMMIT, START, END)

    _funding_artifact(tmp_path, "BTCUSDT")
    manifest_path = tmp_path / "bybit-btcusdt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["private_api_calls"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_contract_invalid"):
        ai6._read_funding_artifact(tmp_path, "BTCUSDT", COMMIT, START, END)

    _funding_artifact(tmp_path, "BTCUSDT")
    with pytest.raises(ValueError, match="window_mismatch"):
        ai6._read_funding_artifact(
            tmp_path,
            "BTCUSDT",
            COMMIT,
            START + pd.Timedelta(minutes=1),
            END,
        )


def test_evaluate_primary_group_uses_purged_oof_fit_and_defers_auc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(200, "4h")
    primary = Series(np.arange(len(bars)) % 3 == 0)
    features = DataFrame(
        {name: np.linspace(0.1, 1.0, len(bars)) for name in V4AQ_META_FEATURE_COLUMNS}
    )

    def fake_oof(
        signal: Series[bool],
        *,
        start_index: int,
        stop_index: int,
        fold_count: int,
    ) -> DataFrame:
        indexes = np.arange(start_index + 2, stop_index - 2, 2)
        return DataFrame(
            {
                "event_index": indexes,
                "primary_signal": "BUY",
                "primary_oof": True,
                "fold_id": np.arange(len(indexes)) % fold_count,
            }
        )

    def fake_labels(
        frame: DataFrame,
        events: DataFrame,
        *,
        timeframe: str,
        horizon_bars: int,
    ) -> DataFrame:
        result = events.copy().reset_index(drop=True)
        result["label_end_idx"] = result["event_index"] + horizon_bars
        result["meta_label"] = np.arange(len(result)) % 2
        result["net_return_fraction"] = np.where(
            result["meta_label"].eq(1), 0.01, -0.01
        )
        return result

    def fake_split(
        events: DataFrame, *, outer_train_end_index: int
    ) -> tuple[DataFrame, DataFrame]:
        return events.iloc[:-20].copy(), events.iloc[-20:].copy()

    def fake_fit(
        fit_features: DataFrame,
        fit_labels: Series[int],
        calibration_features: DataFrame,
        **kwargs: Any,
    ) -> tuple[object, dict[str, Any]]:
        assert len(fit_features) >= 40
        assert fit_labels.nunique() == 2
        assert len(calibration_features) == 20
        return object(), {
            "fit_rows": len(fit_features),
            "calibration_rows": len(calibration_features),
            "selected_threshold": 0.5,
        }

    monkeypatch.setattr(v4x, "generate_oof_primary_events", fake_oof)
    monkeypatch.setattr(v4x, "split_meta_fit_calibration", fake_split)
    monkeypatch.setattr(ai6, "_label_primary_events", fake_labels)
    monkeypatch.setattr(ai6, "fit_v4aq_lightgbm_meta_filter", fake_fit)
    monkeypatch.setattr(
        ai6,
        "predict_v4aq_meta_probabilities",
        lambda model, frame: np.linspace(0.1, 0.9, len(frame)),
    )

    filtered, diagnostics, decisions = ai6._evaluate_primary_group(
        bars,
        features,
        primary,
        strategy=v4ap.STRATEGIES[1],
        timeframe="4h",
        train_end=160,
        forward_end=190,
    )
    assert diagnostics["fit_performed"] is True
    assert diagnostics["fit_status"] == "FITTED"
    assert diagnostics["roc_auc_status"] == (
        "NOT_EVALUATED_ROC_AUC_DEFERRED_FOR_ECONOMIC_SWEEP"
    )
    assert diagnostics["roc_auc"] is None
    assert diagnostics["threshold"] == 0.5
    assert len(decisions) == diagnostics["validation_labeled_event_count"]
    assert filtered.any()


def test_evaluate_primary_group_fails_closed_when_training_support_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(40, "4h")
    primary = Series(np.zeros(len(bars), dtype=bool))
    features = DataFrame(
        {name: np.ones(len(bars)) for name in V4AQ_META_FEATURE_COLUMNS}
    )
    monkeypatch.setattr(
        v4x,
        "generate_oof_primary_events",
        lambda *args, **kwargs: DataFrame(),
    )
    filtered, diagnostics, decisions = ai6._evaluate_primary_group(
        bars,
        features,
        primary,
        strategy=v4ap.STRATEGIES[0],
        timeframe="4h",
        train_end=25,
        forward_end=35,
    )
    assert not filtered.any()
    assert diagnostics["fit_performed"] is False
    assert diagnostics["fit_status"].startswith("BLOCKED_TRAINING_SUPPORT")
    assert decisions == []


def test_blocked_run_builds_and_validates_all_36_no_fit_cells(
    tmp_path: Path,
) -> None:
    candidates = [
        ai6._placeholder_candidate(
            symbol,
            timeframe,
            strategy,
            geometry,
            status="NOT_RUN_INSUFFICIENT_FUNDING_HISTORY",
        )
        for symbol in v4ap.ASSETS
        for timeframe in v4ap.TIMEFRAMES
        for strategy in v4ap.STRATEGIES
        for geometry in v4ap.EXIT_GEOMETRIES
    ]
    dsr = v4ap._attach_dsr_to_cohort(candidates)
    for candidate in candidates:
        candidate["qualifies"] = v4ap._candidate_qualifies(candidate)
    funding_manifest = {
        "research_status": "BLOCKED_INSUFFICIENT_FUNDING_HISTORY",
        "funding": {"full_requested_history_available": False},
    }
    report = ai6._build_report(
        commit_sha=COMMIT,
        start_utc=START,
        end_utc=END,
        market_provenance={},
        funding_provenance={
            "BTCUSDT": funding_manifest,
            "ETHUSDT": funding_manifest,
        },
        candidates=candidates,
        dsr_cohort=dsr,
        model_groups=[],
        status="BLOCKED_INSUFFICIENT_FUNDING_HISTORY",
    )
    ai6._validate_report(report, COMMIT)
    assert len(report["candidates"]) == 36
    assert report["economic_results"] is None
    assert report["model"]["fit_performed"] is False
    assert report["conclusion"] == (
        "FUNDING_COVERAGE_INSUFFICIENT_NO_ECONOMIC_CONCLUSION"
    )
    assert report["Champion"] is None
    summary = ai6._render_summary(report)
    assert "all 36 combinations" in summary.lower()
    assert "ROC-AUC" in summary
    assert "Net PF" in summary
    assert "DSR Z" in summary
    assert "OI: DISABLED" in summary
    assert "NOT evidence for or against statistical edge" in summary
    assert "CLOSE after this completed negative diagnostic" not in summary


def test_invalid_ai6_report_is_rejected() -> None:
    with pytest.raises(ValueError, match="report_schema_invalid"):
        ai6._validate_report({"schema_version": "wrong"}, COMMIT)
    with pytest.raises(ValueError, match="commit_sha_invalid"):
        ai6._validate_commit_sha("bad")


def test_v4aq_ai6_sweep_runs_exact_grid_and_keeps_final_holdout_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    _funding_artifact(external_dir, "BTCUSDT")
    _funding_artifact(external_dir, "ETHUSDT")
    start_utc = START.isoformat()
    end_utc = END.isoformat()
    calls: dict[str, int] = {"models": 0, "candidates": 0}

    def fake_load_market(
        input_dir: Path,
        *,
        symbol: str,
        commit_sha: str,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
    ) -> tuple[DataFrame, dict[str, Any]]:
        return DataFrame(), {
            "symbol": symbol,
            "row_count": 1000,
            "missing_gap_ranges": [],
            "public_api_only": True,
        }

    def fake_resample(minute_data: DataFrame, timeframe: str) -> DataFrame:
        return _bars(800, timeframe)

    def fake_external(
        bars: DataFrame,
        funding: DataFrame,
        *,
        symbol: str,
        timeframe: str,
    ) -> tuple[DataFrame, dict[str, int]]:
        return (
            DataFrame(
                {
                    "funding_rate_zscore": np.zeros(len(bars)),
                    "funding_rate_ma": np.zeros(len(bars)),
                },
                index=bars.index,
            ),
            {"complete_funding_feature_rows": len(bars)},
        )

    def fake_primary_context(
        bars: DataFrame, external: DataFrame, *, strategy: str
    ) -> DataFrame:
        return DataFrame(
            {name: np.ones(len(bars)) for name in V4AQ_META_FEATURE_COLUMNS},
            index=bars.index,
        )

    def fake_signal(bars: DataFrame) -> tuple[Series[bool], dict[str, int]]:
        signal = Series(np.arange(len(bars)) % 5 == 0, index=bars.index)
        return signal, {"primary_signal_count": int(signal.sum())}

    def fake_evaluate(
        bars: DataFrame,
        features: DataFrame,
        signal: Series[bool],
        *,
        strategy: str,
        timeframe: str,
        train_end: int,
        forward_end: int,
    ) -> tuple[Series[bool], dict[str, Any], list[dict[str, Any]]]:
        calls["models"] += 1
        filtered = Series(False, index=bars.index)
        filtered.iloc[train_end:forward_end] = signal.iloc[train_end:forward_end]
        return (
            filtered,
            {
                "fit_performed": True,
                "fit_status": "FITTED",
                "fit_rows": 100,
                "calibration_rows": 20,
                "validation_event_count": int(signal.iloc[train_end:forward_end].sum()),
                "validation_labeled_event_count": int(
                    signal.iloc[train_end:forward_end].sum()
                ),
                "roc_auc": 0.61,
                "roc_auc_status": "COMPUTED",
                "threshold": 0.5,
                "accepted_event_count": int(filtered.sum()),
                "acceptance_rate": 0.5,
            },
            [],
        )

    def fake_run_candidate(
        bars: DataFrame,
        *,
        symbol: str,
        timeframe: str,
        strategy: str,
        geometry: dict[str, Any],
        signal: Series[bool],
        strategy_diagnostics: dict[str, Any],
        execution: Any,
        documented_gap_ranges: tuple[tuple[str, str], ...] = (),
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        calls["candidates"] += 1
        candidate = {
            "candidate_id": v4ap._candidate_id(
                symbol, timeframe, strategy, geometry["name"]
            ),
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "geometry": geometry["name"],
            "bars_evaluated": len(bars),
            "raw_signal_count": strategy_diagnostics["primary_signal_count"],
            "eligible_entry_signal_count": int(signal.sum()),
            "gap_flatten_count": 0,
            "trade_count": 60,
            "win_rate": 0.5,
            "net_profit_factor": 1.2,
            "net_profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": 5.0,
            "net_pnl_usd": 100.0,
            "net_expectancy_usd_per_trade": 1.0,
            "total_fees_usd": 10.0,
            "total_slippage_cost_usd": 5.0,
            "round_trip_cost_bps": 30.0,
            "fee_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "stop_loss_atr": geometry["stop_loss_atr"],
            "take_profit_atr": geometry["take_profit_atr"],
            "trailing_enabled": geometry["trailing_enabled"],
            "trailing_activation_atr": geometry["trailing_activation_atr"],
            "trailing_step_atr": geometry["trailing_step_atr"],
            "horizon_hours_nominal": 24,
            "horizon_bars": 6,
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "execution_policy": "CLOSED_BAR_NEXT_BAR_OPEN",
            "return_observation_count": 60,
            "sharpe_ratio_per_trade": 2.0,
            "return_skewness": 0.0,
            "return_kurtosis_non_excess": 3.0,
            "sharpe_status": "COMPUTED",
        }
        return candidate, []

    def fake_dsr(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        for candidate in candidates:
            candidate["dsr_z_score"] = 0.0
            candidate["dsr_trial_count"] = 36
        return {
            "trial_count": 36,
            "valid_sharpe_estimate_count": 36,
            "trial_scope": "ALL_36_PREREGISTERED_COMBINATIONS",
        }

    monkeypatch.setattr(v4ap, "_load_market_data", fake_load_market)
    monkeypatch.setattr(ai6, "resample_ohlcv", fake_resample)
    monkeypatch.setattr(v4ap, "generate_mean_reversion_signal", fake_signal)
    monkeypatch.setattr(v4ap, "generate_donchian_breakout_signal", fake_signal)
    monkeypatch.setattr(v4ap, "_execution_config", lambda: object())
    monkeypatch.setattr(v4ap, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(v4ap, "_attach_dsr_to_cohort", fake_dsr)
    monkeypatch.setattr(v4ap, "_candidate_qualifies", lambda candidate: False)
    monkeypatch.setattr(ai6, "build_external_market_features", fake_external)
    monkeypatch.setattr(ai6, "build_v4aq_meta_features", fake_primary_context)
    monkeypatch.setattr(ai6, "_evaluate_primary_group", fake_evaluate)

    report_path = tmp_path / "report.json"
    trades_path = tmp_path / "trades.csv"
    decisions_path = tmp_path / "decisions.csv"
    summary_path = tmp_path / "summary.md"
    report = ai6.run_v4aq_ai6(
        tmp_path / "spot",
        external_dir,
        report_path,
        trades_path,
        decisions_path,
        summary_path,
        commit_sha=COMMIT,
        start_utc=start_utc,
        end_utc=end_utc,
    )

    assert report["status"] == "COMPLETED_RESEARCH_EVALUATION"
    assert len(report["candidates"]) == 36
    assert calls == {"models": 12, "candidates": 36}
    assert report["model"]["fit_group_count"] == 12
    assert report["dataset"]["final_holdout_fraction"] == pytest.approx(0.15)
    assert report["dataset"]["final_holdout_status"].startswith("UNTOUCHED")
    assert report["funding"]["open_interest_used"] is False
    assert report["Champion"] is None
    assert report["trading_authorization"] == "NO-GO"
    assert report_path.is_file() and trades_path.is_file()
    assert decisions_path.is_file() and summary_path.is_file()
