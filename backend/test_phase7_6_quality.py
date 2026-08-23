import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from traffic_quality import assess_quality, quality_status
from traffic_store import TrafficStore

client = TestClient(app)


def row(location: str, timestamp: datetime, speed: float = 20.0, free_flow: float = 40.0) -> dict:
    return {
        "location_id": location,
        "latitude": 18.5,
        "longitude": 73.8,
        "timestamp": timestamp.isoformat(),
        "current_speed_mph": speed,
        "free_flow_speed_mph": free_flow,
        "confidence": 0.9,
        "road_closure": False,
    }


def seed(path: Path, location: str = "a", count: int = 12, gap_after: int | None = None) -> None:
    store = TrafficStore(path)
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    for index in range(count):
        offset = 15 if gap_after is not None and index > gap_after else 0
        assert store.insert_observation(row(location, start + timedelta(minutes=index * 5 + offset), speed=20 + index))


def test_perfect_cadence_and_exact_window(tmp_path: Path):
    database = tmp_path / "quality.sqlite3"
    seed(database, count=12)
    report = assess_quality(database)[0]
    assert report["timestamp_ordered"] is True
    assert report["gap_count"] == 0
    assert report["expected_observation_count"] == 12
    assert report["coverage_percentage"] == 100.0
    assert report["usable_training_windows"] == 1
    assert report["ML_ready"] is True


def test_multiple_valid_windows(tmp_path: Path):
    database = tmp_path / "quality.sqlite3"
    seed(database, count=14)
    assert assess_quality(database)[0]["usable_training_windows"] == 3


def test_missing_timestamp_and_large_gap(tmp_path: Path):
    database = tmp_path / "quality.sqlite3"
    store = TrafficStore(database)
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    for index in range(12):
        timestamp = "" if index == 2 else (start + timedelta(minutes=index * 5 + (15 if index > 6 else 0))).isoformat()
        payload = row("a", start, speed=20)
        payload["timestamp"] = timestamp
        assert store.insert_observation(payload)
    report = assess_quality(database)[0]
    assert report["missing_timestamp_count"] == 1
    assert report["invalid_timestamp_count"] == 1
    assert report["gap_count"] >= 1
    assert report["largest_gap_minutes"] >= 15


def test_duplicate_timestamp_is_counted_and_deduplicated(tmp_path: Path):
    database = tmp_path / "duplicates.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE traffic_observations (id INTEGER PRIMARY KEY, location_id TEXT, latitude REAL, longitude REAL, timestamp TEXT, current_speed_mph REAL, free_flow_speed_mph REAL, confidence REAL, road_closure INTEGER)")
        values = ("a", 18.5, 73.8, "2026-08-23T00:00:00+00:00", 20, 40, 0.9, 0)
        connection.execute("INSERT INTO traffic_observations VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        connection.execute("INSERT INTO traffic_observations VALUES (2, ?, ?, ?, ?, ?, ?, ?, ?)", values)
    report = assess_quality(database)[0]
    assert report["observation_count"] == 2
    assert report["duplicate_count"] == 1


def test_insufficient_history_and_multiple_locations(tmp_path: Path):
    database = tmp_path / "quality.sqlite3"
    seed(database, "a", 11)
    seed(database, "b", 12)
    reports = assess_quality(database)
    assert reports[0]["ML_ready"] is False
    assert reports[1]["ML_ready"] is True
    status = quality_status(database)
    assert status["total_locations"] == 2
    assert status["ml_ready_locations"] == 1
    assert status["non_ready_locations"] == 1


def test_invalid_values_are_reported(tmp_path: Path):
    database = tmp_path / "invalid.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE traffic_observations (id INTEGER PRIMARY KEY, location_id TEXT, latitude REAL, longitude REAL, timestamp TEXT, current_speed_mph REAL, free_flow_speed_mph REAL, confidence REAL, road_closure TEXT)")
        connection.execute("INSERT INTO traffic_observations VALUES (1, 'a', 18.5, 73.8, '2026-08-23T00:00:00+00:00', -1, 0, 1.2, 'unknown')")
    report = assess_quality(database)[0]
    assert report["valid_observation_count"] == 0
    assert report["invalid_speed_values"] == 1
    assert report["invalid_free_flow_values"] == 1
    assert report["invalid_confidence_values"] == 1
    assert report["invalid_road_closure_values"] == 1


def test_empty_database(tmp_path: Path):
    database = tmp_path / "empty.sqlite3"
    TrafficStore(database).initialize()
    status = quality_status(database)
    assert assess_quality(database) == []
    assert status["total_observations"] == 0
    assert status["total_usable_windows"] == 0
    assert status["largest_gap_minutes"] == 0.0


def test_quality_api_filters_and_status(tmp_path: Path, monkeypatch):
    database = tmp_path / "api.sqlite3"
    seed(database, "a", 12)
    monkeypatch.setattr("app.main.DEFAULT_DATABASE", database)
    response = client.get("/api/traffic/quality?location_id=a")
    status = client.get("/api/traffic/quality/status")
    assert response.status_code == 200
    assert response.json()[0]["usable_training_windows"] == 1
    assert status.status_code == 200
    assert status.json()["ml_ready_locations"] == 1
