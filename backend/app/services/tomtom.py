from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from app.services.live_cache import TTLCache

logger = logging.getLogger(__name__)


class LiveTrafficError(RuntimeError):
    pass


class TomTomTrafficClient:
    endpoint = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    def __init__(
        self,
        api_key: str | None,
        latitude: float,
        longitude: float,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 60,
    ) -> None:
        self.api_key = api_key
        self.latitude = latitude
        self.longitude = longitude
        self.timeout_seconds = timeout_seconds
        self.cache: TTLCache[dict[str, Any]] = TTLCache(cache_ttl_seconds)

    def fetch(self, force_refresh: bool = False) -> dict[str, Any]:
        if not self.api_key:
            raise LiveTrafficError("TOMTOM_API_KEY is not configured.")
        if not force_refresh:
            cached = self.cache.get()
            if cached is not None:
                return cached

        params = {
            "key": self.api_key,
            "point": f"{self.latitude},{self.longitude}",
            "unit": "MPH",
        }
        try:
            response = requests.get(self.endpoint, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.warning("TomTom traffic request failed: %s", exc)
            raise LiveTrafficError("TomTom traffic service is unavailable.") from exc
        except ValueError as exc:
            logger.warning("TomTom returned malformed JSON: %s", exc)
            raise LiveTrafficError("TomTom returned malformed traffic data.") from exc

        flow = payload.get("flowSegmentData") if isinstance(payload, dict) else None
        if not isinstance(flow, dict) or not isinstance(flow.get("currentSpeed"), (int, float)):
            logger.warning("TomTom response is missing flowSegmentData.currentSpeed")
            raise LiveTrafficError("TomTom returned an unexpected traffic response.")

        result = {
            "source": "TomTom",
            "current_speed_mph": float(flow["currentSpeed"]),
            "free_flow_speed_mph": float(flow["freeFlowSpeed"]) if isinstance(flow.get("freeFlowSpeed"), (int, float)) else None,
            "confidence": float(flow["confidence"]) if isinstance(flow.get("confidence"), (int, float)) else None,
            "road_closure": bool(flow.get("roadClosure", False)),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.cache.set(result)
