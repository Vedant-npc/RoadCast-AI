from datetime import datetime, timedelta, timezone
from pathlib import Path

from traffic_readiness import assess_readiness
from traffic_store import TrafficStore


def insert_row(store: TrafficStore, location: str, timestamp: datetime, speed: float = 20.0, free_flow: float = 40.0) -> None:
    assert store.insert_observation({
        "location_id": location,
        "latitude": 18.5,
        "longitude": 73.8,
        "timestamp": timestamp.isoformat(),
        "current_speed_mph": speed,
        "free_flow_speed_mph": free_flow,
        "confidence": 0.9,
        "road_closure": False,
    })


def seed(path: Path, count: int = 12, location: str = "a", gap_index: int | None = None) -> None:
    store = TrafficStore(path)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for index in range(count):
        offset = 10 if gap_index is not None and index >= gap_index else 0
        insert_row(store, location, start + timedelta(minutes=index * 5 + offset), speed=20 + index)


def test_empty_dataset(tmp_path: Path):
    database = tmp_path / "empty.sqlite3"
    TrafficStore(database).initialize()
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["readiness_level"] == "NOT_READY"
    assert report["total_observations"] == 0
    assert report["valid_training_windows"] == 0
    assert report["drift"]["statistics"]["live_current_speed_mph"]["mean"] is None


def test_insufficient_history_is_not_ready(tmp_path: Path):
    database = tmp_path / "short.sqlite3"
    seed(database, count=11)
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["readiness_level"] == "NOT_READY"
    assert report["valid_training_windows"] == 0
    assert report["locations"][0]["rejection_reasons"] == {"insufficient_history": 1}


def test_sufficient_history_but_insufficient_future_target_is_pilot(tmp_path: Path):
    database = tmp_path / "history.sqlite3"
    seed(database, count=12)
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["valid_training_windows"] == 1
    assert report["target_available_60_min_windows"] == 0
    assert report["readiness_level"] == "PILOT_READY"
    assert report["training_possible"] is False


def test_future_target_requires_twelve_following_observations(tmp_path: Path):
    database = tmp_path / "target.sqlite3"
    seed(database, count=24)
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["valid_training_windows"] == 13
    assert report["target_available_60_min_windows"] == 1


def test_cadence_gap_prevents_windows(tmp_path: Path):
    database = tmp_path / "gap.sqlite3"
    seed(database, count=24, gap_index=12)
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["valid_training_windows"] == 2
    assert report["target_available_60_min_windows"] == 0
    assert report["missing_intervals"] == 1


def test_invalid_observation_is_excluded(tmp_path: Path):
    database = tmp_path / "invalid.sqlite3"
    store = TrafficStore(database)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for index in range(12):
        insert_row(store, "a", start + timedelta(minutes=index * 5), speed=-1 if index == 4 else 20)
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["valid_observations"] == 11
    assert report["invalid_observations"] == 1
    assert report["valid_training_windows"] == 0


def test_multiple_locations_remain_independent(tmp_path: Path):
    database = tmp_path / "multiple.sqlite3"
    seed(database, count=12, location="a")
    seed(database, count=12, location="b")
    report = assess_readiness(database, reference_values=(10.0, 20.0))
    assert report["total_locations"] == 2
    assert report["valid_training_windows"] == 2
    assert all(item["training_independent_location"] for item in report["locations"])


def test_drift_statistics_use_only_comparable_speed_field(tmp_path: Path):
    database = tmp_path / "drift.sqlite3"
    seed(database, count=12)
    report = assess_readiness(database, reference_values=tuple(float(value) for value in range(10, 30)))
    drift = report["drift"]
    assert drift["comparable_fields"] == ["speed"]
    assert drift["statistics"]["live_current_speed_mph"]["mean"] == 25.5
    assert drift["indicators"]["diagnostic_only"] is True
    assert "weather" in drift["excluded_fields"]


def test_readiness_levels_can_advance_only_with_explicit_requirements(tmp_path: Path, monkeypatch):
    database = tmp_path / "levels.sqlite3"
    seed(database, count=24)
    monkeypatch.setattr("traffic_readiness.PILOT_MIN_DAYS", 0)
    monkeypatch.setattr("traffic_readiness.TRAINING_MIN_DAYS", 0)
    monkeypatch.setattr("traffic_readiness.PRODUCTION_MIN_DAYS", 999)
    assert assess_readiness(database, reference_values=(10.0, 20.0))["readiness_level"] == "TRAINING_READY"
    monkeypatch.setattr("traffic_readiness.PRODUCTION_MIN_DAYS", 0)
    assert assess_readiness(database, reference_values=(10.0, 20.0))["readiness_level"] == "PRODUCTION_READY"
