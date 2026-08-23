from unittest.mock import Mock, patch

import pytest
import requests

from app.main import app
from app.services.open_meteo import LiveWeatherError, OpenMeteoClient
from app.services.tomtom import LiveTrafficError, TomTomTrafficClient
from fastapi.testclient import TestClient

client = TestClient(app)


def response(payload, status_code=200):
    result = Mock()
    result.raise_for_status.return_value = None
    result.json.return_value = payload
    result.status_code = status_code
    return result


def test_missing_tomtom_key_is_clear(monkeypatch):
    monkeypatch.setattr("app.main.settings.tomtom_api_key", None)
    with pytest.raises(LiveTrafficError, match="TOMTOM_API_KEY"):
        TomTomTrafficClient(None, 18.5, 73.8).fetch()


def test_tomtom_success_and_cache():
    payload = {"flowSegmentData": {"currentSpeed": 21, "freeFlowSpeed": 42, "confidence": 0.9}}
    with patch("app.services.tomtom.requests.get", return_value=response(payload)) as get:
        service = TomTomTrafficClient("secret", 18.5, 73.8)
        first = service.fetch()
        second = service.fetch()
    assert first["current_speed_mph"] == 21
    assert first["free_flow_speed_mph"] == 42
    assert second == first
    get.assert_called_once()


def test_tomtom_failure_and_malformed_response():
    with patch("app.services.tomtom.requests.get", side_effect=requests.Timeout()):
        with pytest.raises(LiveTrafficError, match="unavailable"):
            TomTomTrafficClient("secret", 18.5, 73.8).fetch()
    with patch("app.services.tomtom.requests.get", return_value=response({"bad": True})):
        with pytest.raises(LiveTrafficError, match="unexpected"):
            TomTomTrafficClient("secret", 18.5, 73.8).fetch()


def test_weather_parsing_and_failure():
    payload = {"current": {"temperature_2m": 28, "precipitation": 1.2, "rain": 0.7, "wind_speed_10m": 11, "weather_code": 61, "time": "2026-08-23T10:00"}}
    with patch("app.services.open_meteo.requests.get", return_value=response(payload)):
        weather = OpenMeteoClient(18.5, 73.8).fetch()
    assert weather["temperature_c"] == 28
    assert weather["rain_mm"] == 0.7
    with patch("app.services.open_meteo.requests.get", side_effect=requests.ConnectionError()):
        with pytest.raises(LiveWeatherError, match="unavailable"):
            OpenMeteoClient(18.5, 73.8).fetch()


def test_live_status_does_not_claim_ml_prediction(monkeypatch):
    monkeypatch.setattr("app.main.settings.tomtom_api_key", None)
    response = client.get("/api/live/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["traffic"] is None
    assert payload["nowcast"]["available"] is False
    assert "TOMTOM_API_KEY" in payload["status"]
