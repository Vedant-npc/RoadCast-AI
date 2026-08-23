from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import get_settings
from app.services.tomtom import LiveTrafficError, TomTomTrafficClient
from traffic_store import TrafficStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrafficLocation:
    location_id: str
    name: str
    latitude: float
    longitude: float


# Public OSM geocoder results; these are query points, not METR-LA sensor mappings.
DEFAULT_LOCATIONS = (
    TrafficLocation("pune_shivaji_nagar", "Shivaji Nagar", 18.532172, 73.8496602),
    TrafficLocation("pune_wakad_flyover", "Wakad Flyover", 18.5920497, 73.7579029),
    TrafficLocation("pune_koregaon_park", "Koregaon Park", 18.5366225, 73.8932738),
    TrafficLocation("pune_katraj", "Katraj", 18.4536792, 73.8563196),
    TrafficLocation("pune_viman_nagar", "Viman Nagar", 18.570387, 73.9133336),
)


def load_locations() -> tuple[TrafficLocation, ...]:
    configured = os.getenv("TRAFFIC_LOCATIONS_JSON")
    if not configured:
        return DEFAULT_LOCATIONS
    try:
        items = json.loads(configured)
        return tuple(TrafficLocation(**item) for item in items)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("TRAFFIC_LOCATIONS_JSON must be a JSON list of location objects.") from exc


class TrafficCollector:
    def __init__(
        self,
        store: TrafficStore,
        locations: tuple[TrafficLocation, ...] | None = None,
        client_factory: Callable[[TrafficLocation], TomTomTrafficClient] | None = None,
        dry_run: bool = False,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.locations = locations or load_locations()
        self.interval_seconds = int(os.getenv("TRAFFIC_POLL_INTERVAL_SECONDS", "300"))
        self.dry_run = dry_run
        self.client_factory = client_factory or (lambda location: TomTomTrafficClient(
            api_key=settings.tomtom_api_key,
            latitude=location.latitude,
            longitude=location.longitude,
            timeout_seconds=settings.live_request_timeout_seconds,
            cache_ttl_seconds=0,
        ))

    def collect_once(self) -> dict[str, Any]:
        if not self.dry_run:
            self.store.initialize()
        attempted = inserted = duplicates = failures = 0
        for location in self.locations:
            attempted += 1
            try:
                traffic = self.client_factory(location).fetch(force_refresh=True)
                observation = {
                    "location_id": location.location_id,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "timestamp": traffic.get("fetched_at") or datetime.now(timezone.utc).isoformat(),
                    "current_speed_mph": traffic.get("current_speed_mph"),
                    "free_flow_speed_mph": traffic.get("free_flow_speed_mph"),
                    "confidence": traffic.get("confidence"),
                    "road_closure": traffic.get("road_closure"),
                }
                if self.dry_run:
                    inserted += 1
                elif self.store.insert_observation(observation):
                    inserted += 1
                else:
                    duplicates += 1
            except LiveTrafficError as exc:
                failures += 1
                logger.warning("Collection failed for %s: %s", location.location_id, exc)
        return {
            "locations": len(self.locations),
            "attempted": attempted,
            "inserted": inserted,
            "duplicates": duplicates,
            "failures": failures,
            "dry_run": self.dry_run,
        }

    def run_forever(self) -> None:
        while True:
            started = time.monotonic()
            self.collect_once()
            time.sleep(max(0, self.interval_seconds - (time.monotonic() - started)))
