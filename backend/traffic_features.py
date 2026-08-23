from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.nowcasting import CongestionThresholds, calculate_congestion_score, classify_congestion

DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "traffic_history.sqlite3"
REQUIRED_HISTORY_LENGTH = 12
ROLLING_WINDOWS = (3, 5)
FEATURE_NAMES = (
    "current_speed_mph",
    "free_flow_speed_mph",
    "speed_ratio",
    "congestion_score",
    "congestion_severity",
    "rolling_mean_speed_mph",
    "rolling_min_speed_mph",
    "rolling_max_speed_mph",
    "speed_change_mph",
    "speed_change_pct",
    "rolling_congestion_score",
    "hour",
    "day_of_week",
    "weekend_flag",
)


def _read_rows(database_path: str | Path, location_id: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM traffic_observations"
    parameters: list[Any] = []
    if location_id:
        query += " WHERE location_id = ?"
        parameters.append(location_id)
    query += " ORDER BY location_id, timestamp, id"
    try:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
    except sqlite3.OperationalError:
        return []

    # Keep the newest row if a database was imported with duplicate timestamps.
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        deduplicated[(payload["location_id"], payload["timestamp"])] = payload
    return sorted(deduplicated.values(), key=lambda row: (row["location_id"], row["timestamp"], row["id"]))


def _valid_speed(speed: Any) -> bool:
    return speed is not None and float(speed) >= 0


def _valid_reference(reference: Any) -> bool:
    return reference is not None and float(reference) > 0


def build_traffic_features(
    database_path: str | Path = DEFAULT_DATABASE,
    location_id: str | None = None,
    limit: int | None = None,
    thresholds: CongestionThresholds | None = None,
) -> list[dict[str, Any]]:
    rows = _read_rows(database_path, location_id)
    threshold_config = thresholds or CongestionThresholds()
    output: list[dict[str, Any]] = []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["location_id"], []).append(row)

    for current_location, location_rows in grouped.items():
        speeds: list[float | None] = []
        scores: list[float | None] = []
        parsed_timestamps: list[datetime | None] = []
        for row in location_rows:
            speed = float(row["current_speed_mph"]) if _valid_speed(row["current_speed_mph"]) else None
            reference = float(row["free_flow_speed_mph"]) if _valid_reference(row["free_flow_speed_mph"]) else None
            score = calculate_congestion_score(speed, reference) if speed is not None and reference is not None else None
            speeds.append(speed)
            scores.append(score)
            try:
                parsed_timestamps.append(datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")))
            except (AttributeError, ValueError):
                parsed_timestamps.append(None)

        for index, row in enumerate(location_rows):
            speed = speeds[index]
            reference = float(row["free_flow_speed_mph"]) if _valid_reference(row["free_flow_speed_mph"]) else None
            score = scores[index]
            prior_speed = speeds[index - 1] if index else None
            history_slice = speeds[max(0, index - REQUIRED_HISTORY_LENGTH + 1) : index + 1]
            valid_history = [value for value in history_slice if value is not None]
            valid_rolling_speeds = [value for value in speeds[max(0, index - 4) : index + 1] if value is not None]
            rolling_scores = [value for value in scores[max(0, index - 4) : index + 1] if value is not None]
            has_required_history = len(history_slice) == REQUIRED_HISTORY_LENGTH and len(valid_history) == REQUIRED_HISTORY_LENGTH
            timestamp = parsed_timestamps[index]
            speed_change = speed - prior_speed if speed is not None and prior_speed is not None else None
            speed_change_pct = speed_change / prior_speed * 100 if speed_change is not None and prior_speed else None

            output.append({
                "id": row["id"],
                "location_id": current_location,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "timestamp": row["timestamp"],
                "current_speed_mph": speed,
                "free_flow_speed_mph": reference,
                "speed_ratio": speed / reference if speed is not None and reference is not None else None,
                "congestion_score": score,
                "congestion_severity": classify_congestion(score, threshold_config) if score is not None else "UNKNOWN",
                "rolling_mean_speed_mph": sum(valid_rolling_speeds) / len(valid_rolling_speeds) if valid_rolling_speeds else None,
                "rolling_min_speed_mph": min(valid_rolling_speeds) if valid_rolling_speeds else None,
                "rolling_max_speed_mph": max(valid_rolling_speeds) if valid_rolling_speeds else None,
                "speed_change_mph": speed_change,
                "speed_change_pct": speed_change_pct,
                "rolling_congestion_score": sum(rolling_scores) / len(rolling_scores) if rolling_scores else None,
                "hour": timestamp.hour if timestamp else None,
                "day_of_week": timestamp.weekday() if timestamp else None,
                "weekend_flag": int(timestamp.weekday() >= 5) if timestamp else None,
                "history_count": len(valid_history),
                "history_usable": has_required_history,
            })

    if limit is not None:
        bounded_limit = max(1, min(int(limit), 1000))
        output = output[-bounded_limit:]
    return output


def get_feature_status(database_path: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    features = build_traffic_features(database_path)
    raw_rows = _read_raw_count(database_path)
    locations = sorted({row["location_id"] for row in features})
    per_location = {location: sum(1 for row in features if row["location_id"] == location) for location in locations}
    timestamps = [row["timestamp"] for row in features]
    usable = sum(1 for row in features if row["history_usable"])
    return {
        "total_observations": raw_rows,
        "usable_ml_rows": usable,
        "insufficient_history_rows": len(features) - usable,
        "locations": locations,
        "observations_per_location": per_location,
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "feature_names": list(FEATURE_NAMES),
        "required_history_length": REQUIRED_HISTORY_LENGTH,
    }


def _read_raw_count(database_path: str | Path) -> int:
    try:
        with sqlite3.connect(database_path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM traffic_observations").fetchone()[0])
    except sqlite3.OperationalError:
        return 0
