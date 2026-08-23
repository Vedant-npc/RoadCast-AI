from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["LOW", "MODERATE", "HIGH", "SEVERE"]


class TrafficSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speed: float | None = Field(default=None, description="Observed or predicted speed in mph")
    congestion_score: float | None = Field(default=None)
    severity: Severity | None = Field(default=None)


class SensorTrafficState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: int
    latitude: float | None = None
    longitude: float | None = None
    road_name: str | None = None
    current_speed: float | None = None
    current_congestion: float | None = None
    current_severity: Severity | None = None
    forecast: dict[str, TrafficSnapshot] = Field(default_factory=dict)


class SensorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_sensors: int
    low: int
    moderate: int
    high: int
    severe: int
    outlook: str


class LiveNowcastStatus(BaseModel):
    available: bool
    reason: str
    horizons: dict[str, TrafficSnapshot] = Field(default_factory=dict)


class LiveDataResponse(BaseModel):
    location: str
    latitude: float
    longitude: float
    traffic: dict | None = None
    weather: dict | None = None
    nowcast: LiveNowcastStatus
    status: str
    last_updated: str | None = None
