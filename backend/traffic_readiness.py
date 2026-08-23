from __future__ import annotations

import math
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from traffic_training import _read_rows, _valid_window
from traffic_quality import _deduplicate, _parse_timestamp, _value_valid
from traffic_quality import assess_quality, quality_status

DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "traffic_history.sqlite3"
METRLA_TRAINING_FILE = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la\train.parquet")
MIN_HISTORY = 12
TARGET_STEPS_60_MIN = 12
PILOT_MIN_DAYS = 1
TRAINING_MIN_DAYS = 14
PRODUCTION_MIN_DAYS = 30


def _statistics(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("mean", "std", "min", "max", "p05", "p50", "p95")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


@lru_cache(maxsize=1)
def get_metrla_speed_values() -> tuple[float, ...]:
    if not METRLA_TRAINING_FILE.exists():
        return ()
    frame = pd.read_parquet(METRLA_TRAINING_FILE, columns=["x_t+0_d0"])
    values = pd.to_numeric(frame["x_t+0_d0"], errors="coerce")
    return tuple(float(value) for value in values.dropna() if math.isfinite(float(value)) and float(value) >= 0)


def _live_speed_values(rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        if _value_valid(row):
            values.append(float(row["current_speed_mph"]))
    return values


def _drift_report(live_values: list[float], reference_values: tuple[float, ...]) -> dict[str, Any]:
    comparable = {
        "live_current_speed_mph": _statistics(live_values),
        "metrla_x_t+0_d0": _statistics(reference_values),
    }
    live = comparable["live_current_speed_mph"]
    reference = comparable["metrla_x_t+0_d0"]
    if live["mean"] is None or reference["mean"] is None:
        indicators = None
    else:
        reference_std = float(reference["std"] or 0.0)
        indicators = {
            "mean_difference_mph": float(live["mean"] - reference["mean"]),
            "p50_difference_mph": float(live["p50"] - reference["p50"]),
            "p95_difference_mph": float(live["p95"] - reference["p95"]),
            "std_ratio_live_to_metrla": float(live["std"] / reference_std) if reference_std else None,
            "standardized_mean_difference": float((live["mean"] - reference["mean"]) / reference_std) if reference_std else None,
            "diagnostic_only": True,
        }
    return {
        "comparable_fields": ["speed"],
        "statistics": comparable,
        "indicators": indicators,
        "note": "Diagnostic distribution indicators only; not statistical proof of production drift.",
        "excluded_fields": ["free_flow_speed", "confidence", "road_closure", "weather", "location_id"],
    }


def _target_available(rows: list[dict[str, Any]], window_end_index: int) -> bool:
    future = rows[window_end_index + 1 : window_end_index + 1 + TARGET_STEPS_60_MIN]
    if len(future) != TARGET_STEPS_60_MIN:
        return False
    if any(not _value_valid(row) or _parse_timestamp(row.get("timestamp")) is None for row in future):
        return False
    timestamps = [_parse_timestamp(row["timestamp"]) for row in future]
    previous = _parse_timestamp(rows[window_end_index]["timestamp"])
    if previous is None:
        return False
    for current in timestamps:
        if current is None or abs((current - previous).total_seconds() / 60 - 5.0) > 1.0:
            return False
        previous = current
    return True


def _location_readiness(location_id: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows, duplicate_count = _deduplicate(source_rows)
    valid_observations = sum(_value_valid(row) and _parse_timestamp(row.get("timestamp")) is not None for row in rows)
    valid_windows = target_windows = 0
    rejected = 0
    reasons: dict[str, int] = {}
    if len(rows) < MIN_HISTORY:
        rejected = 1
        reasons["insufficient_history"] = 1
    else:
        for start in range(len(rows) - MIN_HISTORY + 1):
            candidate = rows[start : start + MIN_HISTORY]
            if not _valid_window(candidate):
                rejected += 1
                reasons["invalid_or_nonconsecutive_window"] = reasons.get("invalid_or_nonconsecutive_window", 0) + 1
                continue
            valid_windows += 1
            if _target_available(rows, start + MIN_HISTORY - 1):
                target_windows += 1
    timestamps = [_parse_timestamp(row.get("timestamp")) for row in rows]
    parsed = [timestamp for timestamp in timestamps if timestamp is not None]
    duration_days = (max(parsed) - min(parsed)).total_seconds() / 86400 if len(parsed) >= 2 else 0.0
    return {
        "location_id": location_id,
        "observation_count": len(source_rows),
        "valid_observations": valid_observations,
        "valid_training_windows": valid_windows,
        "target_available_60_min_windows": target_windows,
        "rejected_windows": rejected,
        "rejection_reasons": reasons,
        "duplicate_count": duplicate_count,
        "collection_duration_days": round(duration_days, 4),
        "training_independent_location": True,
        "ML_ready": valid_windows > 0 and target_windows > 0,
    }


def assess_readiness(
    database_path: str | Path = DEFAULT_DATABASE,
    reference_values: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    rows = _read_rows(database_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["location_id"], []).append(row)
    location_reports = [_location_readiness(location_id, source_rows) for location_id, source_rows in sorted(grouped.items())]
    quality_reports = {report["location_id"]: report for report in assess_quality(database_path)}
    for report in location_reports:
        quality = quality_reports.get(report["location_id"], {})
        report["coverage_percentage"] = quality.get("coverage_percentage", 0.0)
        report["gap_count"] = quality.get("gap_count", 0)
        report["largest_gap_minutes"] = quality.get("largest_gap_minutes", 0.0)
        report["additional_observations_needed"] = quality.get("additional_observations_needed", MIN_HISTORY)
    valid_observations = sum(report["valid_observations"] for report in location_reports)
    valid_windows = sum(report["valid_training_windows"] for report in location_reports)
    target_windows = sum(report["target_available_60_min_windows"] for report in location_reports)
    durations = [report["collection_duration_days"] for report in location_reports]
    max_duration = max(durations, default=0.0)
    locations = len(location_reports)
    all_have_history = locations > 0 and all(report["valid_training_windows"] > 0 for report in location_reports)
    if not all_have_history:
        level = "NOT_READY"
    elif max_duration < PILOT_MIN_DAYS or target_windows == 0:
        level = "PILOT_READY"
    elif max_duration < TRAINING_MIN_DAYS:
        level = "PILOT_READY"
    elif max_duration < PRODUCTION_MIN_DAYS:
        level = "TRAINING_READY"
    else:
        level = "PRODUCTION_READY"
    timestamps = [row["timestamp"] for row in rows if _parse_timestamp(row.get("timestamp")) is not None]
    live_values = _live_speed_values(rows)
    return {
        "readiness_level": level,
        "training_possible": level in {"TRAINING_READY", "PRODUCTION_READY"},
        "total_observations": len(rows),
        "valid_observations": valid_observations,
        "invalid_observations": len(rows) - valid_observations,
        "total_locations": locations,
        "observations_per_location": {report["location_id"]: report["observation_count"] for report in location_reports},
        "collection_duration_days": max_duration,
        "minimum_history_observations": MIN_HISTORY,
        "minimum_forecast_horizon_minutes": 60,
        "valid_training_windows": valid_windows,
        "target_available_60_min_windows": target_windows,
        "usable_feature_rows": valid_windows,
        "coverage_percentage": round(
            sum(report["coverage_percentage"] for report in location_reports) / locations, 2
        ) if locations else 0.0,
        "timestamp_cadence": "5 minutes with +/-1 minute collection jitter tolerance",
        "missing_intervals": sum(report["gap_count"] for report in location_reports),
        "duplicate_timestamps": sum(report["duplicate_count"] for report in location_reports),
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "recommended_additional_observations": quality_status(database_path).get("recommended_additional_observations", 0),
        "locations": location_reports,
        "location_identity_note": "Pune TomTom query locations remain independent external locations; no METR-LA node mapping is claimed.",
        "drift": _drift_report(live_values, reference_values if reference_values is not None else get_metrla_speed_values()),
        "requirements": {
            "pilot_ready": "One valid history window per location; not sufficient for model training.",
            "training_ready": "At least 14 days of history and target-backed valid windows with stable cadence and quality.",
            "production_ready": "At least 30 days, broad target-backed coverage, validated drift, and operational data-quality checks.",
        },
    }
