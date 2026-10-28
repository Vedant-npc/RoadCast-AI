from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import LiveDataResponse, LiveNowcastStatus, SensorSummary, SensorTrafficState
from app.services.live_adapter import adapt_live_traffic
from app.services.open_meteo import LiveWeatherError, OpenMeteoClient
from app.services.tomtom import LiveTrafficError, TomTomTrafficClient
from traffic_store import TrafficStore
from src.models.nowcasting import (
    CongestionThresholds,
    calculate_congestion_score,
    classify_congestion,
    derive_reference_speed_lookup,
)
from src.models.sensor_map import get_sensor_locations
from src.models.training import build_feature_frame, get_model
from traffic_analysis import DEFAULT_DATABASE, analyze_traffic, get_history
from traffic_features import build_traffic_features, get_feature_status
from traffic_quality import assess_quality, quality_status
from traffic_training import DEFAULT_EXPORT, build_training_windows, training_status
from traffic_readiness import assess_readiness
from traffic_live import build_live_prediction_status, build_live_traffic_snapshot

settings = get_settings()

DATA_DIR = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la")
MODEL_DIR = Path(r"d:\My_Projects\RoadCast AI\backend\models")
THRESHOLDS = CongestionThresholds()
HORIZON_MAP = {"15_min": 3, "30_min": 6, "45_min": 9, "60_min": 12}
MODEL_HORIZON_MAP = {"15_min": "15min", "30_min": "30min", "45_min": "45min", "60_min": "60min"}

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="AI-Powered Urban Traffic Congestion Nowcasting System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TrafficStore(DEFAULT_DATABASE).initialize()


@lru_cache(maxsize=2)
def _load_split(file_name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / file_name)
    if "t0_timestamp" in df.columns:
        df["t0_timestamp"] = pd.to_datetime(df["t0_timestamp"], errors="coerce")
    return df.copy()


@lru_cache(maxsize=1)
def _reference_speed_lookup() -> Dict[int, float]:
    return derive_reference_speed_lookup(_load_split("train.parquet"))


@lru_cache(maxsize=1)
def _traffic_client() -> TomTomTrafficClient:
    return TomTomTrafficClient(
        api_key=settings.tomtom_api_key,
        latitude=settings.live_latitude,
        longitude=settings.live_longitude,
        timeout_seconds=settings.live_request_timeout_seconds,
        cache_ttl_seconds=settings.live_traffic_cache_seconds,
    )


@lru_cache(maxsize=1)
def _weather_client() -> OpenMeteoClient:
    return OpenMeteoClient(
        latitude=settings.live_latitude,
        longitude=settings.live_longitude,
        timeout_seconds=settings.live_request_timeout_seconds,
        cache_ttl_seconds=settings.live_weather_cache_seconds,
    )


def _unavailable_nowcast(reason: str) -> LiveNowcastStatus:
    return LiveNowcastStatus(available=False, reason=reason)


def _load_sensor_row(node_id: int) -> pd.Series:
    test_df = _load_split("test.parquet")
    rows = test_df[test_df["node_id"] == int(node_id)]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Sensor {node_id} not found")
    return rows.iloc[0]


def _predict_horizon(model_name: str, horizon_key: str, row: pd.Series) -> float:
    model_path = MODEL_DIR / f"{model_name}_{MODEL_HORIZON_MAP[horizon_key]}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for horizon {horizon_key}: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_frame = build_feature_frame(pd.DataFrame([row]))
    feature_cols = [
        col
        for col in feature_frame.columns
        if col not in {"node_id", "t0_timestamp"}
    ]
    X = feature_frame[feature_cols].astype(float)
    return float(model.predict(X)[0])


def _build_sensor_payload(node_id: int) -> Dict[str, Any]:
    row = _load_sensor_row(node_id)
    reference_lookup = _reference_speed_lookup()
    reference_speed = float(reference_lookup.get(int(node_id), float(max(reference_lookup.values(), default=70.0))))

    current_speed = float(row["x_t+0_d0"])
    current_congestion = calculate_congestion_score(current_speed, reference_speed)
    current_severity = classify_congestion(current_congestion, THRESHOLDS)

    forecast: Dict[str, Dict[str, Any]] = {}
    for horizon_key, steps in HORIZON_MAP.items():
        predicted_speed = _predict_horizon("xgboost", horizon_key, row)
        predicted_score = calculate_congestion_score(predicted_speed, reference_speed)
        forecast[horizon_key] = {
            "speed": float(predicted_speed),
            "congestion_score": float(predicted_score),
            "severity": classify_congestion(predicted_score, THRESHOLDS),
        }

    location = next((item for item in get_sensor_locations() if int(item.get("node_id", -1)) == node_id), {})
    return {
        "node_id": int(node_id),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "road_name": location.get("road_name"),
        "current_speed": current_speed,
        "current_congestion": float(current_congestion),
        "current_severity": current_severity,
        "forecast": {
            horizon_key: {
                "speed": payload["speed"],
                "congestion_score": payload["congestion_score"],
                "severity": payload["severity"],
            }
            for horizon_key, payload in forecast.items()
        },
    }


@app.get("/")
def read_root() -> dict:
    return {
        "project": settings.app_name,
        "status": "backend is running",
        "phase": 5,
        "message": "Interactive traffic map backend is ready.",
    }


@app.get(f"{settings.api_prefix}/health")
def health_check() -> dict:
    return {"status": "ok", "service": settings.app_name}


@app.get(f"{settings.api_prefix}/sensors", response_model=List[dict])
def list_sensors() -> List[dict]:
    sensor_locations = get_sensor_locations()
    if sensor_locations:
        return sensor_locations

    train_df = _load_split("train.parquet")
    node_ids = sorted(train_df["node_id"].dropna().unique().astype(int).tolist())
    return [
        {
            "node_id": int(node_id),
            "latitude": None,
            "longitude": None,
            "road_name": None,
        }
        for node_id in node_ids
    ]


@app.get(f"{settings.api_prefix}/sensors/{{node_id}}", response_model=SensorTrafficState)
def get_sensor_state(node_id: int) -> SensorTrafficState:
    payload = _build_sensor_payload(node_id)
    return SensorTrafficState(**payload)


@app.get(f"{settings.api_prefix}/sensors/{{node_id}}/forecast", response_model=SensorTrafficState)
def get_sensor_forecast(node_id: int) -> SensorTrafficState:
    payload = _build_sensor_payload(node_id)
    return SensorTrafficState(**payload)


@app.get(f"{settings.api_prefix}/summary", response_model=SensorSummary)
def get_dashboard_summary() -> SensorSummary:
    sensors = list_sensors()
    test_df = _load_split("test.parquet")
    reference_lookup = _reference_speed_lookup()
    low = moderate = high = severe = 0
    total = 0
    for sensor in sensors:
        node_id = sensor.get("node_id")
        if node_id is None:
            continue
        total += 1
        rows = test_df[test_df["node_id"] == int(node_id)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        reference_speed = float(reference_lookup.get(int(node_id), max(reference_lookup.values(), default=70.0)))
        current_speed = float(row["x_t+0_d0"])
        severity = classify_congestion(
            calculate_congestion_score(current_speed, reference_speed),
            THRESHOLDS,
        )
        if severity == "LOW":
            low += 1
        elif severity == "MODERATE":
            moderate += 1
        elif severity == "HIGH":
            high += 1
        elif severity == "SEVERE":
            severe += 1

    if severe >= high and severe >= moderate:
        outlook = "Severe congestion risk is elevated"
    elif high >= moderate and high >= low:
        outlook = "High congestion is concentrated in key corridors"
    elif moderate >= low:
        outlook = "Moderate congestion is present across the network"
    else:
        outlook = "Current conditions stable"

    return SensorSummary(
        total_sensors=total,
        low=low,
        moderate=moderate,
        high=high,
        severe=severe,
        outlook=outlook,
    )


@app.get(f"{settings.api_prefix}/traffic/history", response_model=List[dict])
def get_traffic_history(location_id: str | None = None, limit: int = 100) -> List[dict]:
    return get_history(DEFAULT_DATABASE, location_id=location_id, limit=limit)


@app.get(f"{settings.api_prefix}/traffic/analysis", response_model=List[dict])
def get_traffic_analysis(location_id: str | None = None) -> List[dict]:
    return analyze_traffic(DEFAULT_DATABASE, location_id=location_id)


@app.get(f"{settings.api_prefix}/traffic/features", response_model=List[dict])
def get_traffic_features(location_id: str | None = None, limit: int | None = None) -> List[dict]:
    return build_traffic_features(DEFAULT_DATABASE, location_id=location_id, limit=limit)


@app.get(f"{settings.api_prefix}/traffic/features/status", response_model=dict)
def get_traffic_features_status() -> dict:
    return get_feature_status(DEFAULT_DATABASE)


@app.get(f"{settings.api_prefix}/traffic/quality", response_model=List[dict])
def get_traffic_quality(location_id: str | None = None) -> List[dict]:
    return assess_quality(DEFAULT_DATABASE, location_id=location_id)


@app.get(f"{settings.api_prefix}/traffic/quality/status", response_model=dict)
def get_traffic_quality_status() -> dict:
    return quality_status(DEFAULT_DATABASE)


@app.get(f"{settings.api_prefix}/traffic/training/status", response_model=dict)
def get_traffic_training_status() -> dict:
    return training_status(DEFAULT_DATABASE, DEFAULT_EXPORT)


@app.get(f"{settings.api_prefix}/traffic/training/windows", response_model=List[dict])
def get_traffic_training_windows(location_id: str | None = None, limit: int = 100) -> List[dict]:
    windows, _ = build_training_windows(DEFAULT_DATABASE, location_id=location_id)
    return windows[: max(1, min(int(limit), 1000))]


@app.get(f"{settings.api_prefix}/traffic/readiness", response_model=dict)
def get_traffic_readiness() -> dict:
    return assess_readiness(DEFAULT_DATABASE)


@app.get(f"{settings.api_prefix}/traffic/readiness/status", response_model=dict)
def get_traffic_readiness_status() -> dict:
    report = assess_readiness(DEFAULT_DATABASE)
    return {
        key: report[key]
        for key in (
            "readiness_level", "training_possible", "total_observations",
            "valid_observations", "invalid_observations", "total_locations",
            "observations_per_location", "collection_duration_days",
            "valid_training_windows", "target_available_60_min_windows",
            "usable_feature_rows", "coverage_percentage", "missing_intervals",
            "duplicate_timestamps", "earliest_timestamp", "latest_timestamp",
            "recommended_additional_observations",
        )
    }


@app.get(f"{settings.api_prefix}/live/traffic", response_model=LiveDataResponse)
def get_live_traffic() -> LiveDataResponse:
    try:
        traffic = adapt_live_traffic(_traffic_client().fetch())
    except LiveTrafficError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return LiveDataResponse(
        location=settings.live_location_name,
        latitude=settings.live_latitude,
        longitude=settings.live_longitude,
        traffic=traffic,
        nowcast=_unavailable_nowcast(traffic["reason"]),
        status="available",
        last_updated=traffic.get("fetched_at"),
    )


@app.get(f"{settings.api_prefix}/live/weather", response_model=LiveDataResponse)
def get_live_weather() -> LiveDataResponse:
    try:
        weather = _weather_client().fetch()
    except LiveWeatherError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return LiveDataResponse(
        location=settings.live_location_name,
        latitude=settings.live_latitude,
        longitude=settings.live_longitude,
        weather=weather,
        nowcast=_unavailable_nowcast("Weather is contextual and is not used by the current METR-LA model."),
        status="available",
        last_updated=weather.get("time"),
    )


@app.get(f"{settings.api_prefix}/live/status", response_model=LiveDataResponse)
def get_live_status() -> LiveDataResponse:
    traffic = weather = None
    errors: list[str] = []
    try:
        traffic = adapt_live_traffic(_traffic_client().fetch())
    except LiveTrafficError as exc:
        errors.append(str(exc))
    try:
        weather = _weather_client().fetch()
    except LiveWeatherError as exc:
        errors.append(str(exc))

    status = "available" if traffic is not None and weather is not None else "partial"
    return LiveDataResponse(
        location=settings.live_location_name,
        latitude=settings.live_latitude,
        longitude=settings.live_longitude,
        traffic=traffic,
        weather=weather,
        nowcast=_unavailable_nowcast(
            "Live TomTom features do not match the Phase 3 XGBoost schema; live ML nowcasting is disabled."
        ),
        status=status if not errors else f"{status}: {' '.join(errors)}",
        last_updated=(traffic or {}).get("fetched_at") or (weather or {}).get("time"),
    )


@app.get(f"{settings.api_prefix}/live/prediction", response_model=dict)
def get_live_prediction() -> dict:
    return build_live_prediction_status()


@app.get(f"{settings.api_prefix}/live/dashboard", response_model=dict)
def get_live_dashboard() -> dict:
    traffic = weather = None
    errors: list[str] = []
    try:
        traffic = build_live_traffic_snapshot(adapt_live_traffic(_traffic_client().fetch()))
    except LiveTrafficError as exc:
        errors.append(str(exc))
    try:
        weather = _weather_client().fetch()
    except LiveWeatherError as exc:
        errors.append(str(exc))

    return {
        "location": settings.live_location_name,
        "traffic": traffic,
        "weather": weather,
        "locations": analyze_traffic(DEFAULT_DATABASE),
        "data_quality": quality_status(DEFAULT_DATABASE),
        "readiness": assess_readiness(DEFAULT_DATABASE),
        "prediction": build_live_prediction_status(),
        "status": "available" if not errors else "partial",
        "errors": errors,
    }
