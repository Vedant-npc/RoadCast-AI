from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.models.nowcasting import CongestionThresholds, calculate_congestion_score, classify_congestion

DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "traffic_history.sqlite3"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


def _connect(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _valid_observation(row: sqlite3.Row) -> bool:
    speed = row["current_speed_mph"]
    reference = row["free_flow_speed_mph"]
    return (
        speed is not None
        and reference is not None
        and float(speed) >= 0
        and float(reference) > 0
    )


def _congestion(row: sqlite3.Row) -> float | None:
    if not _valid_observation(row):
        return None
    return calculate_congestion_score(float(row["current_speed_mph"]), float(row["free_flow_speed_mph"]))


def _trend(rows: list[sqlite3.Row]) -> str:
    if len(rows) < 2:
        return "INSUFFICIENT_DATA"
    latest = rows[0]["current_speed_mph"]
    previous = rows[1]["current_speed_mph"]
    if latest is None or previous is None:
        return "INSUFFICIENT_DATA"
    delta = float(latest) - float(previous)
    if abs(delta) < 0.01:
        return "STABLE"
    return "INCREASING" if delta > 0 else "DECREASING"


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "location_id": row["location_id"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "timestamp": row["timestamp"],
        "current_speed_mph": row["current_speed_mph"],
        "free_flow_speed_mph": row["free_flow_speed_mph"],
        "confidence": row["confidence"],
        "road_closure": bool(row["road_closure"]) if row["road_closure"] is not None else None,
        "congestion_score": _congestion(row),
    }


def get_history(
    database_path: str | Path = DEFAULT_DATABASE,
    location_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(int(limit), MAX_LIMIT))
    query = "SELECT * FROM traffic_observations"
    parameters: list[Any] = []
    if location_id:
        query += " WHERE location_id = ?"
        parameters.append(location_id)
    query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
    parameters.append(bounded_limit)
    with _connect(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_row_payload(row) for row in rows]


def analyze_traffic(
    database_path: str | Path = DEFAULT_DATABASE,
    location_id: str | None = None,
    thresholds: CongestionThresholds | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM traffic_observations"
    parameters: list[Any] = []
    if location_id:
        query += " WHERE location_id = ?"
        parameters.append(location_id)
    query += " ORDER BY timestamp DESC, id DESC"
    with _connect(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["location_id"], []).append(row)

    threshold_config = thresholds or CongestionThresholds()
    results = []
    for current_location, location_rows in grouped.items():
        valid_rows = [row for row in location_rows if _valid_observation(row)]
        speeds = [float(row["current_speed_mph"]) for row in valid_rows]
        scores = [float(_congestion(row)) for row in valid_rows]
        latest = valid_rows[0] if valid_rows else None
        latest_score = _congestion(latest) if latest else None
        results.append({
            "location_id": current_location,
            "observation_count": len(location_rows),
            "valid_observation_count": len(valid_rows),
            "average_speed_mph": sum(speeds) / len(speeds) if speeds else None,
            "minimum_speed_mph": min(speeds) if speeds else None,
            "maximum_speed_mph": max(speeds) if speeds else None,
            "average_congestion": sum(scores) / len(scores) if scores else None,
            "latest_speed_mph": float(latest["current_speed_mph"]) if latest else None,
            "latest_congestion": latest_score,
            "congestion_severity": classify_congestion(latest_score, threshold_config) if latest_score is not None else "UNKNOWN",
            "traffic_trend": _trend(valid_rows),
            "latest_timestamp": latest["timestamp"] if latest else None,
        })
    return results
