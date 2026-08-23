from __future__ import annotations

from typing import Any


def adapt_live_traffic(traffic: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider traffic while refusing unsafe METR-LA feature fabrication."""
    return {
        "current_speed_mph": traffic.get("current_speed_mph"),
        "free_flow_speed_mph": traffic.get("free_flow_speed_mph"),
        "confidence": traffic.get("confidence"),
        "road_closure": traffic.get("road_closure", False),
        "fetched_at": traffic.get("fetched_at"),
        "model_compatible": False,
        "unavailable_model_features": [
            "METR-LA sensor node identity",
            "historical lag features",
            "METR-LA graph-neighbor features",
            "trained feature distribution and timestamps",
        ],
        "reason": "TomTom traffic schema does not match the Phase 3 XGBoost feature schema.",
    }
