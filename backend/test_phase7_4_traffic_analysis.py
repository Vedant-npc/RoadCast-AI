from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from traffic_analysis import analyze_traffic, get_history
from traffic_store import TrafficStore

client = TestClient(app)


def seed_database(path: Path) -> None:
    store = TrafficStore(path)
    rows = [
        {"location_id": "a", "latitude": 18.5, "longitude": 73.8, "timestamp": "2026-08-23T10:00:00+00:00", "current_speed_mph": 20.0, "free_flow_speed_mph": 40.0, "confidence": 0.9, "road_closure": False},
        {"location_id": "a", "latitude": 18.5, "longitude": 73.8, "timestamp": "2026-08-23T09:55:00+00:00", "current_speed_mph": 30.0, "free_flow_speed_mph": 40.0, "confidence": 0.9, "road_closure": False},
        {"location_id": "a", "latitude": 18.5, "longitude": 73.8, "timestamp": "2026-08-23T09:50:00+00:00", "current_speed_mph": None, "free_flow_speed_mph": None, "confidence": None, "road_closure": None},
        {"location_id": "b", "latitude": 18.6, "longitude": 73.9, "timestamp": "2026-08-23T10:00:00+00:00", "current_speed_mph": 40.0, "free_flow_speed_mph": 40.0, "confidence": 1.0, "road_closure": False},
    ]
    for row in rows:
        assert store.insert_observation(row)


def test_history_filters_and_includes_phase4_score(tmp_path: Path):
    database = tmp_path / "history.sqlite3"
    seed_database(database)
    history = get_history(database, location_id="a", limit=2)
    assert len(history) == 2
    assert history[0]["location_id"] == "a"
    assert history[0]["congestion_score"] == 0.5
    assert history[0]["road_closure"] is False


def test_analysis_uses_exact_phase4_formula_and_trend(tmp_path: Path):
    database = tmp_path / "history.sqlite3"
    seed_database(database)
    result = analyze_traffic(database, location_id="a")[0]
    assert result["observation_count"] == 3
    assert result["valid_observation_count"] == 2
    assert result["average_speed_mph"] == 25.0
    assert result["minimum_speed_mph"] == 20.0
    assert result["maximum_speed_mph"] == 30.0
    assert result["average_congestion"] == 0.375
    assert result["latest_congestion"] == 0.5
    assert result["congestion_severity"] == "HIGH"
    assert result["traffic_trend"] == "DECREASING"


def test_api_history_and_analysis_use_configured_database(tmp_path: Path, monkeypatch):
    database = tmp_path / "history.sqlite3"
    seed_database(database)
    monkeypatch.setattr("app.main.DEFAULT_DATABASE", database)
    history = client.get("/api/traffic/history?location_id=b&limit=1")
    analysis = client.get("/api/traffic/analysis?location_id=b")
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert analysis.status_code == 200
    assert analysis.json()[0]["latest_speed_mph"] == 40.0
    assert analysis.json()[0]["congestion_severity"] == "LOW"
