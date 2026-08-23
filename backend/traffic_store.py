from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    timestamp TEXT NOT NULL,
    current_speed_mph REAL,
    free_flow_speed_mph REAL,
    confidence REAL,
    road_closure INTEGER,
    UNIQUE(location_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_traffic_location ON traffic_observations(location_id);
CREATE INDEX IF NOT EXISTS idx_traffic_timestamp ON traffic_observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_location_timestamp ON traffic_observations(location_id, timestamp);
"""


class TrafficStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(SCHEMA)

    def insert_observation(self, observation: dict[str, Any]) -> bool:
        self.initialize()
        values = (
            observation["location_id"], observation["latitude"], observation["longitude"],
            observation["timestamp"], observation.get("current_speed_mph"),
            observation.get("free_flow_speed_mph"), observation.get("confidence"),
            observation.get("road_closure"),
        )
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO traffic_observations
                (location_id, latitude, longitude, timestamp, current_speed_mph,
                 free_flow_speed_mph, confidence, road_closure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
            return cursor.rowcount == 1

    def status(self) -> dict[str, Any]:
        self.initialize()
        with sqlite3.connect(self.database_path) as connection:
            total = connection.execute("SELECT COUNT(*) FROM traffic_observations").fetchone()[0]
            earliest, latest = connection.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM traffic_observations"
            ).fetchone()
            per_location = dict(connection.execute(
                "SELECT location_id, COUNT(*) FROM traffic_observations GROUP BY location_id"
            ).fetchall())
            missing = connection.execute(
                """SELECT COUNT(*) FROM traffic_observations
                WHERE current_speed_mph IS NULL OR free_flow_speed_mph IS NULL
                   OR confidence IS NULL OR road_closure IS NULL"""
            ).fetchone()[0]
            invalid = connection.execute(
                """SELECT COUNT(*) FROM traffic_observations
                WHERE current_speed_mph < 0 OR free_flow_speed_mph < 0
                   OR confidence < 0 OR confidence > 1"""
            ).fetchone()[0]
        return {
            "database": str(self.database_path),
            "total_observations": total,
            "earliest_timestamp": earliest,
            "latest_timestamp": latest,
            "observations_per_location": per_location,
            "missing_records": missing,
            "invalid_records": invalid,
        }
