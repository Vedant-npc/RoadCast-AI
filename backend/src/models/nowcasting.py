from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.training import HORIZONS, build_feature_frame, compute_metrics

BASE_DIR = Path(r"d:\My_Projects\RoadCast AI")
MODEL_DIR = BASE_DIR / "backend" / "models"
ARTIFACT_DIR = BASE_DIR / "backend" / "artifacts" / "phase4"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class CongestionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: float = Field(default=0.25, ge=0.0, le=1.0)
    moderate: float = Field(default=0.50, ge=0.0, le=1.0)
    high: float = Field(default=0.75, ge=0.0, le=1.0)
    severe: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "CongestionThresholds":
        if not (0.0 <= self.low < self.moderate < self.high < self.severe <= 1.0):
            raise ValueError("Thresholds must satisfy 0 <= low < moderate < high < severe <= 1.")
        return self


def _coerce_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} cannot be null.")
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value.") from exc
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite.")
    return output


def clamp_speed_to_reference(speed: float, reference_speed: float) -> float:
    speed_value = _coerce_float(speed, "speed")
    reference_value = _coerce_float(reference_speed, "reference_speed")

    if reference_value <= 0:
        raise ValueError("reference_speed must be greater than zero.")

    if speed_value < 0:
        return 0.0
    return float(min(max(speed_value, 0.0), reference_value))


def calculate_congestion_score(speed: float, reference_speed: float) -> float:
    speed_value = _coerce_float(speed, "speed")
    reference_value = _coerce_float(reference_speed, "reference_speed")

    if reference_value <= 0:
        raise ValueError("reference_speed must be greater than zero.")
    if speed_value < 0:
        raise ValueError("speed cannot be negative.")

    if speed_value >= reference_value:
        return 0.0
    if speed_value <= 0:
        return 1.0

    score = (reference_value - speed_value) / reference_value
    return float(max(0.0, min(1.0, score)))


def classify_congestion(score: float, thresholds: CongestionThresholds | None = None) -> str:
    score_value = _coerce_float(score, "congestion_score")
    config = thresholds or CongestionThresholds()

    if not 0.0 <= score_value <= 1.0:
        raise ValueError("congestion_score must be normalized to the 0-1 range.")

    if score_value < config.low:
        return "LOW"
    if score_value < config.moderate:
        return "MODERATE"
    if score_value < config.high:
        return "HIGH"
    return "SEVERE"


def build_nowcast_output(
    current_speed: float,
    predicted_speeds: Mapping[str, float],
    reference_speed: float,
    thresholds: CongestionThresholds | None = None,
) -> Dict[str, Dict[str, float | str]]:
    threshold_config = thresholds or CongestionThresholds()
    current_value = _coerce_float(current_speed, "current_speed")
    ref_value = _coerce_float(reference_speed, "reference_speed")

    if ref_value <= 0:
        raise ValueError("reference_speed must be greater than zero.")

    def packet(speed: float, label: str, is_current: bool = False) -> Dict[str, float | str]:
        clipped_speed = clamp_speed_to_reference(speed, ref_value)
        congestion_score = calculate_congestion_score(clipped_speed, ref_value)
        severity = classify_congestion(congestion_score, threshold_config)
        if is_current:
            return {
                "speed": float(clipped_speed),
                "congestion_score": float(congestion_score),
                "severity": severity,
            }
        return {
            "predicted_speed": float(clipped_speed),
            "congestion_score": float(congestion_score),
            "severity": severity,
        }

    output: Dict[str, Dict[str, float | str]] = {
        "current": packet(current_value, "current", is_current=True),
    }

    for horizon in ["15_min", "30_min", "45_min", "60_min"]:
        if horizon not in predicted_speeds:
            raise ValueError(f"Predicted speed is missing for horizon {horizon}.")
        output[horizon] = packet(_coerce_float(predicted_speeds[horizon], f"predicted_speeds[{horizon}]"), horizon)

    return output


def derive_reference_speed_lookup(train_df: pd.DataFrame) -> Dict[int, float]:
    if train_df.empty:
        raise ValueError("Training data is empty; cannot derive a reference speed.")
    if "node_id" not in train_df.columns:
        raise ValueError("Training data must include node_id to derive node-level reference speeds.")

    observed_speed_column = "x_t+0_d0"
    if observed_speed_column in train_df.columns:
        speed_columns = [observed_speed_column]
    else:
        speed_columns = [
            col for col in train_df.columns
            if col.endswith("_d0") and (col.startswith("x_t") or col.startswith("y_t"))
        ]
    if not speed_columns:
        raise ValueError("No speed-like training columns were found to derive a reference speed.")

    observations = train_df[["node_id", *speed_columns]].copy()
    speed_values = observations[speed_columns].stack(future_stack=True)
    positive_values = speed_values[speed_values > 0]
    if positive_values.empty:
        raise ValueError("No positive speed observations were found in the training split.")

    node_reference = observations.groupby("node_id")[observed_speed_column].quantile(0.85)
    fallback_reference = float(positive_values.quantile(0.85))
    full_lookup = {int(node_id): float(value) for node_id, value in node_reference.items()}
    for node_id in train_df["node_id"].dropna().unique():
        full_lookup.setdefault(int(node_id), fallback_reference)
    return full_lookup


def resolve_reference_speed_for_node(node_id: Any, reference_lookup: Mapping[int, float]) -> float:
    if node_id is None:
        raise ValueError("node_id is required to resolve a reference speed.")
    try:
        node_key = int(node_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("node_id must be convertible to an integer.") from exc

    if node_key in reference_lookup:
        return float(reference_lookup[node_key])

    if reference_lookup:
        values = list(reference_lookup.values())
        return float(np.median(values))

    raise ValueError("reference_lookup is empty, so no reference speed can be resolved.")


def _iter_dataframe_chunks(df: pd.DataFrame, chunk_size: int = 250000) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size].copy()


def evaluate_horizon_nowcast(
    horizon_name: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_path: str | Path,
    thresholds: CongestionThresholds | None = None,
    chunk_size: int = 250000,
) -> Dict[str, Any]:
    if horizon_name not in HORIZONS:
        raise ValueError(f"Unsupported horizon '{horizon_name}'.")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    feature_cols = [
        col for col in build_feature_frame(train_df).columns
        if col not in {"node_id", "t0_timestamp"}
    ]
    target_column = f"y_t+{HORIZONS[horizon_name]}_d0"
    reference_lookup = derive_reference_speed_lookup(train_df)

    actual_speeds: list[float] = []
    predicted_speeds: list[float] = []
    actual_congestion: list[float] = []
    predicted_congestion: list[float] = []

    for chunk in _iter_dataframe_chunks(test_df, chunk_size=chunk_size):
        feature_frame = build_feature_frame(chunk)
        X_chunk = feature_frame[feature_cols].astype(float)
        y_chunk = chunk[target_column].astype(float).to_numpy()

        predictions = model.predict(X_chunk)
        for node_id, actual_speed, predicted_speed in zip(chunk["node_id"], y_chunk, predictions):
            ref_speed = resolve_reference_speed_for_node(node_id, reference_lookup)
            actual_speed_clipped = clamp_speed_to_reference(actual_speed, ref_speed)
            predicted_speed_clipped = clamp_speed_to_reference(predicted_speed, ref_speed)

            actual_speeds.append(float(actual_speed_clipped))
            predicted_speeds.append(float(predicted_speed_clipped))
            actual_congestion.append(calculate_congestion_score(actual_speed_clipped, ref_speed))
            predicted_congestion.append(calculate_congestion_score(predicted_speed_clipped, ref_speed))

    metrics = compute_metrics(np.asarray(actual_speeds), np.asarray(predicted_speeds))
    report = pd.DataFrame(
        {
            "actual_speed": actual_speeds,
            "predicted_speed": predicted_speeds,
            "actual_congestion": actual_congestion,
            "predicted_congestion": predicted_congestion,
        }
    )

    report_path = ARTIFACT_DIR / f"phase4_validation_{horizon_name}.csv"
    report.to_csv(report_path, index=False)

    return {
        "horizon": horizon_name,
        "model_path": str(model_path),
        "metrics": metrics,
        "validation_rows": len(report),
        "report_path": str(report_path),
        "actual_congestion_mean": float(np.mean(actual_congestion)),
        "predicted_congestion_mean": float(np.mean(predicted_congestion)),
        "thresholds": (thresholds or CongestionThresholds()).model_dump(),
    }


def generate_full_phase4_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: CongestionThresholds | None = None,
) -> Dict[str, Any]:
    threshold_config = thresholds or CongestionThresholds()
    report: Dict[str, Any] = {}
    for horizon_name in HORIZONS:
        model_path = MODEL_DIR / f"xgboost_{horizon_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing XGBoost model: {model_path}")

        report[horizon_name] = evaluate_horizon_nowcast(
            horizon_name=horizon_name,
            test_df=test_df,
            train_df=train_df,
            model_path=model_path,
            thresholds=threshold_config,
        )

    return report
