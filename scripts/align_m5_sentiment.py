"""Point-in-time align daily sentiment to supplied BTCUSDT decision timestamps.

This utility accepts timestamp-only decision grids. It deliberately does not
read or merge OHLCV, create features for model training, or perform a backtest.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence, cast

ALLOWED_TIMEFRAMES = frozenset({"1h", "4h"})
ALIGNMENT_POLICY = "latest_available_at_lte_decision_time_forward_fill"
TARGET_MARKET = "Nobitex Spot/USDT BTCUSDT"


def parse_aware_timestamp(value: object, field_name: str) -> datetime:
    """Parse an ISO-8601 timestamp with an explicit timezone."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}_must_be_iso8601_string")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError(f"{field_name}_invalid_iso8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name}_must_include_timezone")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    """Render a timezone-aware datetime in canonical UTC form."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_source_record(
    record: Mapping[str, object], seen_observations: set[str]
) -> tuple[datetime, Mapping[str, object]]:
    """Validate one sentiment row and return its availability key."""
    available_at = parse_aware_timestamp(record.get("available_at"), "available_at")
    observation_time = record.get("observation_time_utc")
    if not isinstance(observation_time, str):
        raise ValueError("sentiment_observation_time_missing")
    parse_aware_timestamp(observation_time, "observation_time_utc")
    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError("sentiment_value_invalid")
    classification = record.get("value_classification")
    if not isinstance(classification, str) or not classification.strip():
        raise ValueError("sentiment_classification_missing")
    if observation_time in seen_observations:
        raise ValueError("sentiment_duplicate_observation_time")
    seen_observations.add(observation_time)
    return available_at, record


def _validate_source_records(
    sentiment_records: Sequence[Mapping[str, object]],
) -> list[tuple[datetime, Mapping[str, object]]]:
    """Validate and sort records by their known availability timestamps."""
    seen_observations: set[str] = set()
    validated = [
        _validate_source_record(record, seen_observations)
        for record in sentiment_records
    ]
    validated.sort(key=lambda item: item[0])
    return validated


def _parse_decision_time(value: str | datetime) -> datetime:
    """Normalize one decision timestamp to UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_time_must_include_timezone")
        return value.astimezone(timezone.utc)
    return parse_aware_timestamp(value, "decision_time")


def _validate_decision_times(
    decision_times: Sequence[str | datetime],
) -> list[datetime]:
    """Normalize the decision grid and enforce strict chronological order."""
    decisions = [_parse_decision_time(value) for value in decision_times]
    for previous, current in zip(decisions, decisions[1:]):
        if current <= previous:
            raise ValueError("decision_times_must_be_strictly_increasing")
    return decisions


def _missing_alignment(decision: datetime, timeframe: str) -> dict[str, object]:
    """Represent a decision with no prior published sentiment value."""
    return {
        "target_market": TARGET_MARKET,
        "timeframe": timeframe,
        "decision_time": _iso_utc(decision),
        "feature_value": None,
        "feature_classification": None,
        "source_observation_time_utc": None,
        "source_available_at": None,
        "source_age_seconds": None,
        "alignment_policy": ALIGNMENT_POLICY,
        "status": "no_prior_available_record",
    }


def _matched_alignment(
    decision: datetime,
    available_at: datetime,
    source: Mapping[str, object],
    timeframe: str,
) -> dict[str, object]:
    """Render the latest record that was already available at decision time."""
    if available_at > decision:
        raise RuntimeError("future_sentiment_record_selected")
    observation_time = cast(str, source["observation_time_utc"])
    return {
        "target_market": TARGET_MARKET,
        "timeframe": timeframe,
        "decision_time": _iso_utc(decision),
        "feature_value": cast(int, source["value"]),
        "feature_classification": cast(str, source["value_classification"]),
        "source_observation_time_utc": observation_time,
        "source_available_at": _iso_utc(available_at),
        "source_age_seconds": int((decision - available_at).total_seconds()),
        "alignment_policy": ALIGNMENT_POLICY,
        "status": "asof_match_or_forward_fill",
    }


def _align_one_decision(
    decision: datetime,
    source_times: Sequence[datetime],
    sources: Sequence[tuple[datetime, Mapping[str, object]]],
    timeframe: str,
) -> dict[str, object]:
    """Select the latest source row available at or before one decision."""
    source_index = bisect_right(source_times, decision) - 1
    if source_index < 0:
        return _missing_alignment(decision, timeframe)
    available_at, source = sources[source_index]
    return _matched_alignment(decision, available_at, source, timeframe)


def align_records(
    sentiment_records: Sequence[Mapping[str, object]],
    decision_times: Sequence[str | datetime],
    *,
    timeframe: str,
) -> list[dict[str, object]]:
    """Use only the latest sentiment record available at each decision time."""
    if timeframe not in ALLOWED_TIMEFRAMES:
        raise ValueError("m5_sentiment_timeframe_not_allowed")
    sources = _validate_source_records(sentiment_records)
    source_times = [item[0] for item in sources]
    decisions = _validate_decision_times(decision_times)
    return [
        _align_one_decision(decision, source_times, sources, timeframe)
        for decision in decisions
    ]


def _load_json(path: Path) -> object:
    """Read one UTF-8 JSON input file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"input_json_unreadable:{path}") from None


def _write_new_json(path: Path, payload: object) -> None:
    """Write alignment output exclusively; existing files are never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    with path.open("x", encoding="utf-8", newline="\n") as file_handle:
        file_handle.write(rendered + "\n")
        file_handle.flush()


def main() -> int:
    """Align sentiment to a timestamp-only decision grid without loading OHLCV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentiment-json", required=True, type=Path)
    parser.add_argument(
        "--decision-times-json",
        required=True,
        type=Path,
        help="JSON array of timezone-aware ISO timestamps, strictly increasing.",
    )
    parser.add_argument(
        "--timeframe", required=True, choices=sorted(ALLOWED_TIMEFRAMES)
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    sentiment_payload = _load_json(arguments.sentiment_json)
    if not isinstance(sentiment_payload, dict):
        raise ValueError("sentiment_json_not_object")
    raw_records = sentiment_payload.get("records")
    if not isinstance(raw_records, list) or any(
        not isinstance(record, dict) for record in raw_records
    ):
        raise ValueError("sentiment_records_missing_or_invalid")
    records = cast(list[Mapping[str, object]], raw_records)

    decision_payload = _load_json(arguments.decision_times_json)
    if isinstance(decision_payload, dict):
        raw_decisions = decision_payload.get("decision_times")
    else:
        raw_decisions = decision_payload
    if not isinstance(raw_decisions, list):
        raise ValueError("decision_times_json_must_be_array")
    if any(not isinstance(item, (str, datetime)) for item in raw_decisions):
        raise ValueError("decision_time_entry_invalid")

    aligned = align_records(
        records,
        cast(list[str | datetime], raw_decisions),
        timeframe=arguments.timeframe,
    )
    output_payload = {
        "schema_version": "alix.m5.sentiment-alignment.v1",
        "target_market": TARGET_MARKET,
        "timeframe": arguments.timeframe,
        "alignment_policy": ALIGNMENT_POLICY,
        "ohlcv_loaded_or_merged": False,
        "record_count": len(aligned),
        "records": aligned,
    }
    _write_new_json(arguments.output, output_payload)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "record_count": len(aligned),
                "target_market": TARGET_MARKET,
                "timeframe": arguments.timeframe,
                "ohlcv_loaded_or_merged": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
