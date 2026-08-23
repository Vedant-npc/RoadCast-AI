from __future__ import annotations

import logging
from typing import Any

import requests

from app.services.live_cache import TTLCache

logger = logging.getLogger(__name__)


class LiveWeatherError(RuntimeError):
    pass


class OpenMeteoClient:
    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.timeout_seconds = timeout_seconds
        self.cache: TTLCache[dict[str, Any]] = TTLCache(cache_ttl_seconds)

    def fetch(self, force_refresh: bool = False) -> dict[str, Any]:
        if not force_refresh:
            cached = self.cache.get()
            if cached is not None:
                return cached

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,precipitation,rain,wind_speed_10m,weather_code",
            "timezone": "auto",
        }
        try:
            response = requests.get(self.endpoint, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.warning("Open-Meteo request failed: %s", exc)
            raise LiveWeatherError("Open-Meteo weather service is unavailable.") from exc
        except ValueError as exc:
            logger.warning("Open-Meteo returned malformed JSON: %s", exc)
            raise LiveWeatherError("Open-Meteo returned malformed weather data.") from exc

        current = payload.get("current") if isinstance(payload, dict) else None
        if not isinstance(current, dict) or not isinstance(current.get("temperature_2m"), (int, float)):
            raise LiveWeatherError("Open-Meteo returned an unexpected weather response.")

        result = {
            "source": "Open-Meteo",
            "temperature_c": float(current["temperature_2m"]),
            "precipitation_mm": float(current.get("precipitation", 0.0)),
            "rain_mm": float(current.get("rain", 0.0)),
            "wind_speed_kmh": float(current.get("wind_speed_10m", 0.0)),
            "weather_code": current.get("weather_code"),
            "time": current.get("time"),
        }
        return self.cache.set(result)
