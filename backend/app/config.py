from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ROADCAST AI"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api"
    tomtom_api_key: str | None = None
    live_location_name: str = "Pune"
    live_latitude: float = 18.5204
    live_longitude: float = 73.8567
    live_traffic_cache_seconds: int = 60
    live_weather_cache_seconds: int = 300
    live_request_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parent.parent / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
