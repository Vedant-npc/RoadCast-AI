import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from traffic_store import TrafficStore
from traffic_training import build_training_windows, export_training_windows, training_status

client = TestClient(app)


def row(location: str, index: int, speed: float = 20.0, timestamp: str | None = None) -> dict:
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    return {
        "location_id": location,
        "latitude": 18.5,
        "longitude": 73.8,
        "timestamp": timestamp if timestamp is not None else (start + timedelta(minutes=index * 5)).isoformat(),
        "current_speed_mph": speed,
        "free_flow_speed_mph": 40.0,
        "confidence": 0.9,
        "road_closure": False,
    }


def seed(path: Path, location: str = "a", count: int = 12, gap_index: int | None = None) -> None:
    store = TrafficStore(path)
    for index in range(count):
        timestamp = None
        if gap_index is not None and index >= gap_index:
            timestamp = (datetime(2026, 8, 23, tzinfo=timezone.utc) + timedelta(minutes=index * 5 + 10)).isoformat()
        assert store.insert_observation(row(location, index, 20.0 + index, timestamp))


def test_valid_12_observation_window(tmp_path: Path):
    database = tmp_path / "valid.sqlite3"
    seed(database)
    windows, rejected = build_training_windows(database)
    assert len(windows) == 1
    assert rejected == {}
    assert windows[0]["observation_count"] == 12
    assert len(json.loads(windows[0]["speed_history"])) == 12
    assert windows[0]["target_available"] is False


def test_insufficient_history_is_rejected(tmp_path: Path):
    database = tmp_path / "short.sqlite3"
    seed(database, count=11)
    windows, rejected = build_training_windows(database)
    assert windows == []
    assert rejected == {"insufficient_history": 1}
    assert training_status(database)["training_possible"] is False


def test_missing_timestamp_rejects_window(tmp_path: Path):
    database = tmp_path / "missing.sqlite3"
    store = TrafficStore(database)
    for index in range(12):
        payload = row("a", index, timestamp="" if index == 5 else None)
        assert store.insert_observation(payload)
    _, rejected = build_training_windows(database)
    assert rejected == {"missing_or_invalid_timestamp": 1}


def test_duplicate_timestamp_is_not_used_twice(tmp_path: Path):
    database = tmp_path / "duplicate.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE traffic_observations (id INTEGER PRIMARY KEY, location_id TEXT, latitude REAL, longitude REAL, timestamp TEXT, current_speed_mph REAL, free_flow_speed_mph REAL, confidence REAL, road_closure INTEGER)")
        timestamp = "2026-08-23T00:00:00+00:00"
        values = ("a", 18.5, 73.8, timestamp, 20, 40, 0.9, 0)
        connection.execute("INSERT INTO traffic_observations VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        connection.execute("INSERT INTO traffic_observations VALUES (2, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        for index in range(1, 11):
            values = ("a", 18.5, 73.8, f"2026-08-23T00:{index * 5:02d}:00+00:00", 20, 40, 0.9, 0)
            connection.execute("INSERT INTO traffic_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (index + 2, *values))
    windows, rejected = build_training_windows(database)
    assert windows == []
    assert rejected == {"insufficient_history": 1}


def test_invalid_observation_rejects_window(tmp_path: Path):
    database = tmp_path / "invalid.sqlite3"
    store = TrafficStore(database)
    for index in range(12):
        payload = row("a", index)
        if index == 4:
            payload["current_speed_mph"] = -1
        assert store.insert_observation(payload)
    _, rejected = build_training_windows(database)
    assert rejected == {"invalid_observation": 1}
    assert training_status(database)["invalid_observations"] == 1


def test_multiple_locations_and_chronological_windows(tmp_path: Path):
    database = tmp_path / "multiple.sqlite3"
    seed(database, "a", 13)
    seed(database, "b", 12)
    windows, rejected = build_training_windows(database)
    assert len(windows) == 3
    assert rejected == {}
    assert [window["location_id"] for window in windows] == ["a", "a", "b"]
    assert all(window["window_start"] < window["window_end"] for window in windows)


def test_no_future_data_leakage(tmp_path: Path):
    database = tmp_path / "leakage.sqlite3"
    seed(database)
    windows, _ = build_training_windows(database)
    window = windows[0]
    speeds = json.loads(window["speed_history"])
    assert speeds == [20.0 + index for index in range(12)]
    assert window["target_timestamp"] > window["window_end"]
    assert "target_speed" not in window


def test_gap_rejects_window(tmp_path: Path):
    database = tmp_path / "gap.sqlite3"
    seed(database, count=12, gap_index=6)
    windows, rejected = build_training_windows(database)
    assert windows == []
    assert rejected == {"timestamp_gap_or_bad_cadence": 1}


def test_empty_database_and_export_behavior(tmp_path: Path):
    database = tmp_path / "empty.sqlite3"
    TrafficStore(database).initialize()
    export_path = tmp_path / "validated_training_windows.parquet"
    assert export_training_windows(database, export_path) is None
    assert not export_path.exists()
    status = training_status(database, export_path)
    assert status["total_observations"] == 0
    assert status["rejected_windows"] == 0


def test_export_creates_only_valid_windows_and_api(tmp_path: Path, monkeypatch):
    database = tmp_path / "export.sqlite3"
    seed(database)
    export_path = tmp_path / "validated_training_windows.parquet"
    assert export_training_windows(database, export_path) == export_path
    assert export_path.exists()
    monkeypatch.setattr("app.main.DEFAULT_DATABASE", database)
    assert client.get("/api/traffic/training/status").status_code == 200
    response = client.get("/api/traffic/training/windows?limit=1")
    assert response.status_code == 200
    assert len(response.json()) == 1
