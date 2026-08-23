from __future__ import annotations

from typing import Any

from src.models.nowcasting import CongestionThresholds, calculate_congestion_score, classify_congestion

HORIZONS = ("15_min", "30_min", "45_min", "60_min")
INCOMPATIBILITY_REASON = (
    "Live TomTom data does not provide the 34-feature METR-LA vector required by the trained XGBoost models."
)


def build_live_prediction_status() -> dict[str, Any]:
    return {
        "prediction_available": False,
        "reason": INCOMPATIBILITY_REASON,
        "horizons": {
            horizon: {"available": False, "reason": INCOMPATIBILITY_REASON}
            for horizon in HORIZONS
        },
    }


def build_live_traffic_snapshot(traffic: dict[str, Any]) -> dict[str, Any]:
    current_speed = traffic.get("current_speed_mph")
    free_flow_speed = traffic.get("free_flow_speed_mph")
    if current_speed is None or free_flow_speed is None or float(free_flow_speed) <= 0:
        return {
            **traffic,
            "congestion_score": None,
            "congestion_severity": "UNKNOWN",
            "traffic_trend": "INSUFFICIENT_DATA",
        }

    score = calculate_congestion_score(float(current_speed), float(free_flow_speed))
    return {
        **traffic,
        "congestion_score": score,
        "congestion_severity": classify_congestion(score, CongestionThresholds()),
        "traffic_trend": "CURRENT_OBSERVATION_ONLY",
    }
