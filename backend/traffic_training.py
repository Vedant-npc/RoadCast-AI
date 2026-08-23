from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.nowcasting import CongestionThresholds, calculate_congestion_score
from traffic_features import build_traffic_features
from traffic_quality import (
    CADENCE_TOLERANCE_MINUTES,
    DEFAULT_DATABASE,
    EXPECTED_INTERVAL_MINUTES,
    REQUIRED_HISTORY_LENGTH,
    _deduplicate,
    _parse_timestamp,
    _value_valid,
)

DEFAULT_EXPORT = Path(__file__).resolve().parent / "data" / "validated_training_windows.parquet"


def _read_rows(database_path: str | Path) -> list[dict[str, Any]]:
    import sqlite3

    try:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM traffic_observations ORDER BY location_id, timestamp, id"
            ).fetchall()]
    except sqlite3.OperationalError:
        return []
    return rows


def _valid_window(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != REQUIRED_HISTORY_LENGTH:
        return False
    parsed = [_parse_timestamp(row.get("timestamp")) for row in rows]
    if any(timestamp is None for timestamp in parsed) or any(not _value_valid(row) for row in rows):
        return False
    return all(
        abs((current - previous).total_seconds() / 60 - EXPECTED_INTERVAL_MINUTES) <= CADENCE_TOLERANCE_MINUTES
        for previous, current in zip(parsed, parsed[1:])
    )


def _reject_reason(rows: list[dict[str, Any]]) -> str:
    if len(rows) < REQUIRED_HISTORY_LENGTH:
        return "insufficient_history"
    parsed = [_parse_timestamp(row.get("timestamp")) for row in rows]
    if any(timestamp is None for timestamp in parsed):
        return "missing_or_invalid_timestamp"
    if any(not _value_valid(row) for row in rows):
        return "invalid_observation"
    if any(
        abs((current - previous).total_seconds() / 60 - EXPECTED_INTERVAL_MINUTES) > CADENCE_TOLERANCE_MINUTES
        for previous, current in zip(parsed, parsed[1:])
    ):
        return "timestamp_gap_or_bad_cadence"
    return "unknown"


def build_training_windows(
    database_path: str | Path = DEFAULT_DATABASE,
    location_id: str | None = None,
    thresholds: CongestionThresholds | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_rows = _read_rows(database_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        if location_id is None or row["location_id"] == location_id:
            grouped.setdefault(row["location_id"], []).append(row)

    engineered = build_traffic_features(database_path, location_id=location_id, thresholds=thresholds)
    feature_by_key = {(row["location_id"], row["timestamp"]): row for row in engineered}
    windows: list[dict[str, Any]] = []
    rejected = Counter()
    for current_location, source_rows in sorted(grouped.items()):
        rows, duplicate_count = _deduplicate(source_rows)
        if len(rows) < REQUIRED_HISTORY_LENGTH:
            rejected["insufficient_history"] += 1
            continue
        for start in range(len(rows) - REQUIRED_HISTORY_LENGTH + 1):
            candidate = rows[start : start + REQUIRED_HISTORY_LENGTH]
            if not _valid_window(candidate):
                rejected[_reject_reason(candidate)] += 1
                continue
            parsed = [_parse_timestamp(row["timestamp"]) for row in candidate]
            end_timestamp = parsed[-1]
            end_features = feature_by_key.get((current_location, candidate[-1]["timestamp"]), {})
            speed_history = [float(row["current_speed_mph"]) for row in candidate]
            free_flow_history = [float(row["free_flow_speed_mph"]) for row in candidate]
            congestion_history = [
                calculate_congestion_score(speed, reference)
                for speed, reference in zip(speed_history, free_flow_history)
            ]
            window = {
                "location_id": current_location,
                "window_start": candidate[0]["timestamp"],
                "window_end": candidate[-1]["timestamp"],
                "observation_count": REQUIRED_HISTORY_LENGTH,
                "speed_history": json.dumps(speed_history),
                "free_flow_history": json.dumps(free_flow_history),
                "congestion_history": json.dumps(congestion_history),
                "engineered_features": json.dumps({
                    key: end_features.get(key)
                    for key in (
                        "speed_ratio", "congestion_score", "congestion_severity",
                        "rolling_mean_speed_mph", "rolling_min_speed_mph",
                        "rolling_max_speed_mph", "speed_change_mph", "speed_change_pct",
                        "rolling_congestion_score", "hour", "day_of_week", "weekend_flag",
                    )
                }),
                "target_timestamp": (end_timestamp + timedelta(minutes=EXPECTED_INTERVAL_MINUTES)).isoformat(),
                "target_available": False,
                "source_duplicate_count": duplicate_count,
            }
            windows.append(window)
    return windows, dict(rejected)


def export_training_windows(
    database_path: str | Path = DEFAULT_DATABASE,
    export_path: str | Path = DEFAULT_EXPORT,
    location_id: str | None = None,
) -> Path | None:
    windows, _ = build_training_windows(database_path, location_id=location_id)
    if not windows:
        return None
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(windows).to_parquet(path, index=False)
    return path


def training_status(
    database_path: str | Path = DEFAULT_DATABASE,
    export_path: str | Path = DEFAULT_EXPORT,
) -> dict[str, Any]:
    raw_rows = _read_rows(database_path)
    windows, rejection_reasons = build_training_windows(database_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(row["location_id"], []).append(row)
    valid_observations = sum(1 for row in raw_rows if _value_valid(row) and _parse_timestamp(row.get("timestamp")) is not None)
    timestamps = [row["timestamp"] for row in raw_rows if _parse_timestamp(row.get("timestamp")) is not None]
    return {
        "total_observations": len(raw_rows),
        "valid_observations": valid_observations,
        "invalid_observations": len(raw_rows) - valid_observations,
        "total_locations": len(grouped),
        "ml_ready_locations": len({window["location_id"] for window in windows}),
        "valid_training_windows": len(windows),
        "rejected_windows": sum(rejection_reasons.values()),
        "rejection_reasons": rejection_reasons,
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "minimum_required_history": REQUIRED_HISTORY_LENGTH,
        "training_possible": bool(windows),
        "export_created": Path(export_path).exists() if windows else False,
    }
