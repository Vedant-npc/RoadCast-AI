from pathlib import Path
from unittest.mock import Mock

import pytest

from app.services.tomtom import LiveTrafficError
from traffic_collector import TrafficCollector, TrafficLocation
from traffic_store import TrafficStore


LOCATIONS = (
    TrafficLocation("test_a", "Test A", 18.5, 73.8),
    TrafficLocation("test_b", "Test B", 18.6, 73.9),
)


def traffic(speed=22.0):
    return {
        "current_speed_mph": speed,
        "free_flow_speed_mph": 40.0,
        "confidence": 0.9,
        "road_closure": False,
        "fetched_at": "2026-08-23T10:00:00+00:00",
    }


def factory_for(payloads):
    def factory(location):
        client = Mock()
        value = payloads[location.location_id]
        if isinstance(value, Exception):
            client.fetch.side_effect = value
        else:
            client.fetch.return_value = value
        return client
    return factory


def test_store_initializes_indexes_and_deduplicates(tmp_path: Path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    observation = {"location_id": "test_a", "latitude": 18.5, "longitude": 73.8, "timestamp": "2026-08-23T10:00:00+00:00", **traffic()}
    assert store.insert_observation(observation) is True
    assert store.insert_observation(observation) is False
    status = store.status()
    assert status["total_observations"] == 1
    assert status["observations_per_location"] == {"test_a": 1}


def test_mocked_collection_inserts_and_handles_provider_failure(tmp_path: Path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    collector = TrafficCollector(
        store,
        locations=LOCATIONS,
        client_factory=factory_for({"test_a": traffic(), "test_b": LiveTrafficError("rate limited")}),
    )
    result = collector.collect_once()
    assert result["inserted"] == 1
    assert result["failures"] == 1
    assert store.status()["total_observations"] == 1


def test_dry_run_does_not_write(tmp_path: Path):
    database = tmp_path / "traffic.sqlite3"
    collector = TrafficCollector(
        TrafficStore(database),
        locations=LOCATIONS,
        client_factory=factory_for({"test_a": traffic(), "test_b": traffic(18)}),
        dry_run=True,
    )
    result = collector.collect_once()
    assert result["inserted"] == 2
    assert not database.exists()


def test_missing_values_are_reported(tmp_path: Path):
    store = TrafficStore(tmp_path / "traffic.sqlite3")
    observation = {"location_id": "test_a", "latitude": 18.5, "longitude": 73.8, "timestamp": "2026-08-23T10:00:00+00:00", "current_speed_mph": 22.0, "free_flow_speed_mph": None, "confidence": None, "road_closure": None}
    store.insert_observation(observation)
    status = store.status()
    assert status["missing_records"] == 1
    assert status["invalid_records"] == 0


def test_invalid_location_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("TRAFFIC_LOCATIONS_JSON", "not-json")
    with pytest.raises(ValueError, match="TRAFFIC_LOCATIONS_JSON"):
        TrafficCollector(TrafficStore(":memory:"))
