from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "traffic_history.sqlite3"
EXPECTED_INTERVAL_MINUTES = 5.0
CADENCE_TOLERANCE_MINUTES = 1.0
REQUIRED_HISTORY_LENGTH = 12


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _read_rows(database_path: str | Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(
                "SELECT * FROM traffic_observations ORDER BY location_id, timestamp, id"
            ).fetchall()]
    except sqlite3.OperationalError:
        return []


def _value_valid(row: dict[str, Any]) -> bool:
    try:
        speed = float(row["current_speed_mph"])
        free_flow = float(row["free_flow_speed_mph"])
        confidence = float(row["confidence"])
    except (TypeError, ValueError, KeyError):
        return False
    road_closure = row.get("road_closure")
    return (
        math.isfinite(speed) and speed >= 0
        and math.isfinite(free_flow) and free_flow > 0
        and math.isfinite(confidence) and 0 <= confidence <= 1
        and road_closure in (0, 1, False, True)
    )


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_timestamp: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        timestamp = row.get("timestamp")
        if timestamp in by_timestamp:
            duplicate_count += 1
        by_timestamp[timestamp] = row
    return list(sorted(by_timestamp.values(), key=lambda row: (row.get("timestamp", ""), row.get("id", 0)))), duplicate_count


def _location_quality(location_id: str, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows, duplicate_count = _deduplicate(raw_rows)
    parsed = [_parse_timestamp(row.get("timestamp")) for row in rows]
    intervals = [
        (current - previous).total_seconds() / 60
        for previous, current in zip(parsed, parsed[1:])
        if previous is not None and current is not None
    ]
    gap_count = sum(interval > EXPECTED_INTERVAL_MINUTES + CADENCE_TOLERANCE_MINUTES for interval in intervals)
    largest_gap = max(intervals, default=0.0)
    valid_flags = [_value_valid(row) and parsed[index] is not None for index, row in enumerate(rows)]
    valid_count = sum(valid_flags)

    run_lengths: list[int] = []
    current_run = 0
    trailing_run = 0
    for index, is_valid in enumerate(valid_flags):
        cadence_ok = index == 0 or (
            parsed[index] is not None and parsed[index - 1] is not None
            and abs((parsed[index] - parsed[index - 1]).total_seconds() / 60 - EXPECTED_INTERVAL_MINUTES) <= CADENCE_TOLERANCE_MINUTES
        )
        if is_valid and cadence_ok:
            current_run += 1
        else:
            if current_run:
                run_lengths.append(current_run)
            current_run = 1 if is_valid else 0
    if current_run:
        run_lengths.append(current_run)
    usable_windows = sum(max(run_length - REQUIRED_HISTORY_LENGTH + 1, 0) for run_length in run_lengths)
    trailing_run = current_run

    timestamps = [timestamp for timestamp in parsed if timestamp is not None]
    duration = (max(timestamps) - min(timestamps)).total_seconds() / 60 if len(timestamps) >= 2 else 0.0
    expected_count = math.floor(duration / EXPECTED_INTERVAL_MINUTES + 0.5) + 1 if timestamps else 0
    coverage = valid_count / expected_count * 100 if expected_count else 0.0
    return {
        "location_id": location_id,
        "observation_count": len(raw_rows),
        "expected_observation_count": expected_count,
        "coverage_percentage": round(coverage, 2),
        "gap_count": gap_count,
        "largest_gap_minutes": round(largest_gap, 2),
        "duplicate_count": duplicate_count,
        "missing_timestamp_count": sum(not row.get("timestamp") for row in raw_rows),
        "invalid_timestamp_count": sum(timestamp is None for timestamp in parsed),
        "timestamp_ordered": all(
            current >= previous for previous, current in zip(timestamps, timestamps[1:])
        ),
        "cadence_tolerance_minutes": CADENCE_TOLERANCE_MINUTES,
        "valid_observation_count": valid_count,
        "usable_training_windows": usable_windows,
        "ML_ready": usable_windows > 0,
        "missing_speed_values": sum(row.get("current_speed_mph") is None for row in raw_rows),
        "invalid_speed_values": sum(not _valid_speed_value(row.get("current_speed_mph")) for row in raw_rows),
        "invalid_free_flow_values": sum(not _valid_free_flow_value(row.get("free_flow_speed_mph")) for row in raw_rows),
        "invalid_confidence_values": sum(not _valid_confidence_value(row.get("confidence")) for row in raw_rows),
        "invalid_road_closure_values": sum(row.get("road_closure") not in (0, 1, False, True) for row in raw_rows),
        "earliest_timestamp": min((row.get("timestamp") for row in raw_rows), default=None),
        "latest_timestamp": max((row.get("timestamp") for row in raw_rows), default=None),
        "collection_duration_minutes": round(duration, 2),
        "additional_observations_needed": max(REQUIRED_HISTORY_LENGTH - trailing_run, 0),
    }


def _valid_speed_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


def _valid_free_flow_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _valid_confidence_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and 0 <= float(value) <= 1
    except (TypeError, ValueError):
        return False


def assess_quality(database_path: str | Path = DEFAULT_DATABASE, location_id: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_rows(database_path):
        if location_id is None or row.get("location_id") == location_id:
            grouped.setdefault(row["location_id"], []).append(row)
    return [_location_quality(current_location, rows) for current_location, rows in sorted(grouped.items())]


def quality_status(database_path: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    rows = _read_rows(database_path)
    reports = assess_quality(database_path)
    gaps = sum(report["gap_count"] for report in reports)
    largest_gap = max((report["largest_gap_minutes"] for report in reports), default=0.0)
    timestamps = [row["timestamp"] for row in rows]
    durations = [report["collection_duration_minutes"] for report in reports]
    ready = sum(report["ML_ready"] for report in reports)
    return {
        "total_locations": len(reports),
        "total_observations": len(rows),
        "ml_ready_locations": ready,
        "non_ready_locations": len(reports) - ready,
        "total_usable_windows": sum(report["usable_training_windows"] for report in reports),
        "total_gaps": gaps,
        "largest_gap_minutes": largest_gap,
        "collection_duration_minutes": max(durations, default=0.0),
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "recommended_additional_observations": sum(report["additional_observations_needed"] for report in reports),
        "locations": [report["location_id"] for report in reports],
    }
