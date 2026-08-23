import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from traffic_features import build_traffic_features, get_feature_status
from traffic_store import TrafficStore

client = TestClient(app)


def observation(location_id: str, index: int, speed: float = 20.0) -> dict:
    return {
        "location_id": location_id,
        "latitude": 18.5,
        "longitude": 73.8,
        "timestamp": f"2026-08-23T10:{index:02d}:00+00:00",
        "current_speed_mph": speed,
        "free_flow_speed_mph": 40.0,
        "confidence": 0.9,
        "road_closure": False,
    }


def seed(path: Path, count: int = 12) -> None:
    store = TrafficStore(path)
    for index in range(count):
        assert store.insert_observation(observation("a", index, 10.0 + index * 2))


def test_chronological_order_and_rolling_values(tmp_path: Path):
    database = tmp_path / "traffic.sqlite3"
    seed(database, 12)
    features = build_traffic_features(database, location_id="a")
    assert [row["timestamp"] for row in features] == sorted(row["timestamp"] for row in features)
    assert features[0]["rolling_mean_speed_mph"] == 10.0
    assert features[2]["rolling_mean_speed_mph"] == 12.0
    assert features[2]["rolling_min_speed_mph"] == 10.0
    assert features[2]["rolling_max_speed_mph"] == 14.0
    assert features[2]["history_usable"] is False
    assert features[11]["history_usable"] is True


def test_congestion_and_no_future_data_leakage(tmp_path: Path):
    database = tmp_path / "traffic.sqlite3"
    seed(database, 12)
    features = build_traffic_features(database)
    assert features[0]["congestion_score"] == 0.75
    assert features[0]["congestion_severity"] == "SEVERE"
    assert features[1]["rolling_mean_speed_mph"] == 11.0
    assert features[1]["speed_change_mph"] == 2.0
    assert features[1]["speed_change_pct"] == 20.0


def test_multiple_locations_and_status(tmp_path: Path):
    database = tmp_path / "traffic.sqlite3"
    seed(database, 12)
    store = TrafficStore(database)
    for index in range(3):
        assert store.insert_observation(observation("b", index, 30.0))
    status = get_feature_status(database)
    assert status["total_observations"] == 15
    assert status["usable_ml_rows"] == 1
    assert status["insufficient_history_rows"] == 14
    assert status["locations"] == ["a", "b"]
    assert status["required_history_length"] == 12


def test_empty_database_returns_empty_features(tmp_path: Path):
    database = tmp_path / "empty.sqlite3"
    TrafficStore(database).initialize()
    assert build_traffic_features(database) == []
    status = get_feature_status(database)
    assert status["total_observations"] == 0
    assert status["usable_ml_rows"] == 0
    assert status["earliest_timestamp"] is None


def test_duplicate_timestamps_keep_latest_imported_row(tmp_path: Path):
    database = tmp_path / "duplicates.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE traffic_observations (id INTEGER PRIMARY KEY, location_id TEXT, latitude REAL, longitude REAL, timestamp TEXT, current_speed_mph REAL, free_flow_speed_mph REAL, confidence REAL, road_closure INTEGER)")
        connection.execute("INSERT INTO traffic_observations VALUES (1, 'a', 18.5, 73.8, '2026-08-23T10:00:00+00:00', 10, 40, 0.9, 0)")
        connection.execute("INSERT INTO traffic_observations VALUES (2, 'a', 18.5, 73.8, '2026-08-23T10:00:00+00:00', 30, 40, 0.9, 0)")
    features = build_traffic_features(database)
    assert len(features) == 1
    assert features[0]["current_speed_mph"] == 30.0


def test_feature_endpoints_use_configured_database(tmp_path: Path, monkeypatch):
    database = tmp_path / "api.sqlite3"
    seed(database, 12)
    monkeypatch.setattr("app.main.DEFAULT_DATABASE", database)
    response = client.get("/api/traffic/features?location_id=a&limit=2")
    status = client.get("/api/traffic/features/status")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert status.status_code == 200
    assert status.json()["usable_ml_rows"] == 1
