from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app
from traffic_live import HORIZONS, build_live_prediction_status, build_live_traffic_snapshot

client = TestClient(app)


def test_valid_live_data_uses_phase4_congestion_logic():
    snapshot = build_live_traffic_snapshot({
        "current_speed_mph": 20.0,
        "free_flow_speed_mph": 40.0,
        "confidence": 0.9,
    })
    assert snapshot["congestion_score"] == 0.5
    assert snapshot["congestion_severity"] == "HIGH"
    assert snapshot["traffic_trend"] == "CURRENT_OBSERVATION_ONLY"


def test_missing_live_history_is_not_claimed_as_prediction():
    status = build_live_prediction_status()
    assert status["prediction_available"] is False
    assert set(status["horizons"]) == set(HORIZONS)
    assert all(value["available"] is False for value in status["horizons"].values())


def test_missing_or_invalid_live_data_is_safe():
    missing = build_live_traffic_snapshot({"current_speed_mph": 20.0, "free_flow_speed_mph": None})
    assert missing["congestion_score"] is None
    assert missing["congestion_severity"] == "UNKNOWN"
    invalid = build_live_traffic_snapshot({"current_speed_mph": 20.0, "free_flow_speed_mph": 0})
    assert invalid["congestion_score"] is None


def test_live_dashboard_response_structure(monkeypatch):
    traffic = {"current_speed_mph": 20.0, "free_flow_speed_mph": 40.0, "confidence": 0.9, "road_closure": False, "fetched_at": "2026-08-23T10:00:00+00:00"}
    weather = {"temperature_c": 25.0, "time": "2026-08-23T10:00"}
    traffic_client = Mock()
    traffic_client.fetch.return_value = traffic
    weather_client = Mock()
    weather_client.fetch.return_value = weather
    monkeypatch.setattr("app.main._traffic_client", lambda: traffic_client)
    monkeypatch.setattr("app.main._weather_client", lambda: weather_client)
    response = client.get("/api/live/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["traffic"]["congestion_score"] == 0.5
    assert payload["weather"] == weather
    assert payload["prediction"]["prediction_available"] is False
    assert len(payload["locations"]) == 5
    assert payload["data_quality"]["total_observations"] > 0


def test_live_dashboard_handles_provider_failure(monkeypatch):
    from app.services.tomtom import LiveTrafficError
    traffic_client = Mock()
    traffic_client.fetch.side_effect = LiveTrafficError("temporarily unavailable")
    weather_client = Mock()
    weather_client.fetch.return_value = {"temperature_c": 25.0}
    monkeypatch.setattr("app.main._traffic_client", lambda: traffic_client)
    monkeypatch.setattr("app.main._weather_client", lambda: weather_client)
    response = client.get("/api/live/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["traffic"] is None
    assert payload["status"] == "partial"
    assert payload["prediction"]["prediction_available"] is False
