from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core import hierarchical_target_audit as audit
from services.ml_core.adaptive_targets import apply_adaptive_hierarchical_targets
from services.ml_core.hierarchical_target_audit import (
    audit_hierarchical_targets,
    load_manifested_dataset,
    resample_ohlcv,
    run_hierarchical_target_audit,
)

COMMIT_SHA = "a" * 40


def minute_frame(rows: int = 1440) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="min")
    opened = np.full(rows, 100.0)
    close = opened.copy()
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": opened + 0.5,
            "low": opened - 0.5,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )


def write_manifested_dataset(
    input_dir: Path,
    *,
    symbol: str = "BTCUSDT",
    days: int = 1,
    commit_sha: str = COMMIT_SHA,
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    frame = minute_frame(days * 1440)
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    frame.to_csv(csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "alix.binance.public-ohlcv.v1",
        "provider": "Binance Spot public klines API",
        "host": "data-api.binance.vision",
        "endpoint": "/api/v3/klines",
        "symbol": symbol,
        "pair": f"{symbol[:-4]}/USDT",
        "interval": "1m",
        "start_utc": frame["date"].iloc[0].isoformat().replace("+00:00", "Z"),
        "end_utc": frame["date"].iloc[-1].isoformat().replace("+00:00", "Z"),
        "row_count": days * 1440,
        "requested_days": days,
        "retrieved_at": "2026-01-02T00:00:00+00:00",
        "missing_intervals": 0,
        "missing_minute_intervals": 0,
        "gap_policy": "reject; no interpolation, resampling, or synthetic fill",
        "fetch_page_size_candles": 1000,
        "inter_page_delay_seconds": 0.25,
        "csv_sha256": digest,
        "commit_sha": commit_sha,
        "data_authority": "PUBLIC_MARKET_DATA_ONLY_NOT_ACCOUNT_ORDER_OR_FILL_TRUTH",
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def target_frame(rows: int = 80) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="15min")
    opened = np.full(rows, 100.0)
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": opened + 0.5,
            "low": opened - 0.5,
            "close": opened,
            "volume": np.full(rows, 10.0),
        }
    )


def test_load_manifested_dataset_verifies_hash_provenance_and_rows(
    tmp_path: Path,
) -> None:
    write_manifested_dataset(tmp_path)
    candles, provenance = load_manifested_dataset(
        tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
    )
    assert len(candles) == 1440
    assert provenance["public_api_only"] is True
    assert provenance["private_api_calls"] == 0
    assert len(provenance["csv_sha256"]) == 64


def test_load_manifested_dataset_rejects_manifest_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    write_manifested_dataset(tmp_path)
    manifest_path = tmp_path / "binance-btcusdt-1m.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["private_api_calls"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError, match="public_data_manifest_validation_failed:private_api_calls"
    ):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
        )

    manifest["private_api_calls"] = 0
    manifest["host"] = "untrusted.example"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="public_data_manifest_validation_failed:host"):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
        )

    manifest["host"] = "data-api.binance.vision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    csv_path = tmp_path / "binance-btcusdt-1m.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="public_data_manifest_validation_failed:csv_sha256"
    ):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
        )


def test_load_manifested_dataset_rejects_invalid_identity_and_missing_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="public_data_symbol_out_of_policy"):
        load_manifested_dataset(
            tmp_path, symbol="DOGEUSDT", days=1, commit_sha=COMMIT_SHA
        )
    with pytest.raises(ValueError, match="public_data_days_out_of_policy"):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=True, commit_sha=COMMIT_SHA
        )
    with pytest.raises(ValueError, match="commit_sha_invalid"):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha="not-a-sha"
        )
    with pytest.raises(ValueError, match="public_data_artifact_missing"):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
        )


def test_load_manifested_dataset_rejects_invalid_manifest_and_timestamp(
    tmp_path: Path,
) -> None:
    write_manifested_dataset(tmp_path)
    manifest_path = tmp_path / "binance-btcusdt-1m.manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="public_data_manifest_invalid"):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
        )

    write_manifested_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["start_utc"] = "not-a-timestamp"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_timestamp_invalid:start_utc"):
        load_manifested_dataset(
            tmp_path, symbol="BTCUSDT", days=1, commit_sha=COMMIT_SHA
        )


def test_minute_frame_rejects_schema_count_gap_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="public_data_ohlcv_schema_invalid"):
        audit._validate_minute_frame(DataFrame({"date": []}), expected_rows=0)
    with pytest.raises(ValueError, match="public_data_row_count_mismatch"):
        audit._validate_minute_frame(minute_frame(20), expected_rows=21)

    frame = minute_frame(20)
    frame.loc[10, "date"] = pd.Timestamp(frame["date"].iloc[10]) + pd.Timedelta(
        seconds=30
    )
    with pytest.raises(ValueError, match="public_data_minute_gap_detected"):
        audit._validate_minute_frame(frame, expected_rows=20)

    invalid = minute_frame(20)
    invalid.loc[5, "close"] = np.inf
    with pytest.raises(ValueError, match="public_data_non_finite:close"):
        audit._validate_minute_frame(invalid, expected_rows=20)

    invalid = minute_frame(20)
    invalid.loc[5, "low"] = 110.0
    with pytest.raises(ValueError, match="public_data_ohlc_geometry_invalid"):
        audit._validate_minute_frame(invalid, expected_rows=20)


def test_minute_frame_rejects_price_volume_and_timestamp_contracts() -> None:
    invalid = minute_frame(20)
    invalid.loc[5, "volume"] = -1.0
    with pytest.raises(ValueError, match="public_data_price_or_volume_out_of_range"):
        audit._validate_minute_frame(invalid, expected_rows=20)

    invalid = minute_frame(20)
    invalid.loc[5, "date"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="public_data_timestamp_invalid"):
        audit._validate_minute_frame(invalid, expected_rows=20)

    invalid = minute_frame(20)
    invalid.loc[5, "date"] = invalid.loc[4, "date"]
    with pytest.raises(
        ValueError, match="public_data_timestamps_not_unique_and_increasing"
    ):
        audit._validate_minute_frame(invalid, expected_rows=20)


def test_resample_keeps_only_complete_allowed_entry_timeframes() -> None:
    frame = minute_frame(16)
    bars = resample_ohlcv(frame, "15m")
    assert len(bars) == 1
    assert bars["date"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="entry_timeframe_out_of_policy"):
        resample_ohlcv(frame, "1d")
    with pytest.raises(ValueError, match="no_complete_resampled_bars"):
        resample_ohlcv(minute_frame(10), "15m")


def test_resample_fails_closed_on_source_gaps_and_missing_bar_intervals() -> None:
    gapped = minute_frame(45).drop(index=20).reset_index(drop=True)
    with pytest.raises(ValueError, match="public_data_minute_gap_detected"):
        resample_ohlcv(gapped, "15m")


def test_hierarchical_audit_confirms_late_hits_remain_no_hit() -> None:
    candles = target_frame()
    candles.loc[33, "high"] = 103.0
    result = audit_hierarchical_targets(candles, pair="BTC/USDT", timeframe="15m")
    assert result["target_contract_status"] == "PASS"
    assert result["late_hit_transition_count_13_to_24"] > 0
    assert result["stage1_distribution"]["NO_HIT"] > 0
    assert result["stage2_distribution"]["UNLABELED"] > 0


def test_hierarchical_audit_fails_closed_on_stage1_and_stage2_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = apply_adaptive_hierarchical_targets

    def broken_stage1(candles: DataFrame) -> DataFrame:
        result = original(candles)
        result.loc[20, "stage1_hit_target"] = 1.0
        return result

    monkeypatch.setattr(audit, "apply_adaptive_hierarchical_targets", broken_stage1)
    with pytest.raises(ValueError, match="hierarchical_stage1_mapping_invalid"):
        audit_hierarchical_targets(target_frame(), pair="BTC/USDT", timeframe="15m")

    def broken_stage2(candles: DataFrame) -> DataFrame:
        result = original(candles)
        result.loc[20, "stage2_direction_target"] = 0.0
        return result

    monkeypatch.setattr(audit, "apply_adaptive_hierarchical_targets", broken_stage2)
    with pytest.raises(ValueError, match="hierarchical_stage2_mapping_invalid"):
        audit_hierarchical_targets(target_frame(), pair="BTC/USDT", timeframe="15m")


def test_hierarchical_audit_rejects_invalid_scope_and_pair() -> None:
    with pytest.raises(ValueError, match="entry_timeframe_out_of_policy"):
        audit_hierarchical_targets(target_frame(), pair="BTC/USDT", timeframe="1d")
    with pytest.raises(ValueError, match="pair_invalid"):
        audit_hierarchical_targets(target_frame(), pair="", timeframe="15m")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "label": None,
                "label_end_idx": 1,
                "unlabeled_reason": "incomplete_horizon",
            },
            "hierarchical_unlabeled_row_contract_invalid",
        ),
        (
            {
                "label": "HOLD",
                "label_end_idx": 12.5,
                "exit_reason": "HORIZON_EXPIRY",
                "unlabeled_reason": None,
            },
            "hierarchical_label_end_index_invalid",
        ),
        (
            {
                "label": "HOLD",
                "label_end_idx": 11,
                "exit_reason": "HORIZON_EXPIRY",
                "unlabeled_reason": None,
            },
            "hierarchical_no_hit_end_index_invalid",
        ),
        (
            {
                "label": "BUY",
                "label_end_idx": None,
                "exit_reason": "TP_HIT",
                "unlabeled_reason": None,
            },
            "hierarchical_label_end_index_missing",
        ),
        (
            {
                "label": "SELL",
                "label_end_idx": 2,
                "exit_reason": "unexpected",
                "unlabeled_reason": None,
            },
            "hierarchical_hit_exit_reason_invalid",
        ),
    ],
)
def test_label_end_index_validator_rejects_invalid_rows(
    row: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        audit._validate_label_end_indices(DataFrame([row]))


def test_late_hit_validator_rejects_a_short_horizon_direction() -> None:
    short = DataFrame(
        [
            {
                "label": "BUY",
                "exit_reason": "TP_HIT",
                "label_end_idx": 12,
            }
        ]
    )
    long = DataFrame([{"label": "BUY", "label_end_idx": 13}])
    with pytest.raises(ValueError, match="hierarchical_late_hit_must_remain_no_hit"):
        audit._count_late_hits(short, long)


def test_late_hit_validator_rejects_missing_or_out_of_range_end_indices() -> None:
    short = DataFrame(
        [{"label": "HOLD", "exit_reason": "HORIZON_EXPIRY", "label_end_idx": 12}]
    )
    with pytest.raises(ValueError, match="legacy_hit_end_index_missing"):
        audit._count_late_hits(
            short, DataFrame([{"label": "BUY", "label_end_idx": None}])
        )
    with pytest.raises(ValueError, match="legacy_hit_end_index_out_of_horizon"):
        audit._count_late_hits(
            short, DataFrame([{"label": "BUY", "label_end_idx": 25}])
        )


def test_run_audit_emits_eight_experiments_and_preserves_safety(
    tmp_path: Path,
) -> None:
    write_manifested_dataset(tmp_path, symbol="BTCUSDT")
    write_manifested_dataset(tmp_path, symbol="ETHUSDT")
    report = run_hierarchical_target_audit(tmp_path, days=1, commit_sha=COMMIT_SHA)
    assert report["experiment_count"] == 8
    assert {item["timeframe"] for item in report["experiments"]} == {
        "15m",
        "30m",
        "1h",
        "4h",
    }
    assert report["model_training_performed"] is False
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["private_api_calls"] == 0
    assert report["safety"]["real_order_attempts"] == 0
    assert report["safety"]["trading_authorization"] == "NO-GO"


def test_run_audit_rejects_unknown_horizon_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="public_data_days_out_of_policy"):
        run_hierarchical_target_audit(tmp_path, days=0, commit_sha=COMMIT_SHA)
