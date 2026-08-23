from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_available_sensors_endpoint_returns_list():
    response = client.get("/api/sensors")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_invalid_sensor_id_returns_404():
    response = client.get("/api/sensors/999999")
    assert response.status_code == 404


def test_missing_prediction_data_is_handled():
    response = client.get("/api/sensors/0/forecast")
    assert response.status_code in {200, 404}


def test_summary_endpoint_returns_network_counts():
    response = client.get("/api/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_sensors"] > 0
    assert sum(payload[key] for key in ("low", "moderate", "high", "severe")) == payload["total_sensors"]
