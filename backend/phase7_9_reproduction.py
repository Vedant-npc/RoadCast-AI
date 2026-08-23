from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.training import HORIZONS, build_feature_frame, compute_metrics, get_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "raw" / "metr-la"
MODEL_DIR = BASE_DIR / "models" / "phase7_9_reproduction"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "phase7_9"
SAMPLE_SIZES = {"train": 100_000, "val": 30_000, "test": 40_000}
RANDOM_STATE = 42


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"node_id", "t0_timestamp"}]


def load_metrla_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = []
    for split_name in ("train", "val", "test"):
        frame = pd.read_parquet(DATA_DIR / f"{split_name}.parquet")
        frame = frame.sample(n=SAMPLE_SIZES[split_name], random_state=RANDOM_STATE).copy()
        frame["t0_timestamp"] = pd.to_datetime(frame["t0_timestamp"], errors="coerce")
        splits.append(frame)
    return tuple(splits)  # type: ignore[return-value]


def load_versioned_model(horizon_name: str):
    if horizon_name not in HORIZONS:
        raise ValueError(f"Unsupported horizon '{horizon_name}'.")
    model_path = MODEL_DIR / f"xgboost_{horizon_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing reproduced model: {model_path}")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def reproduce_training() -> dict[str, Any]:
    train_df, val_df, test_df = load_metrla_splits()
    train_features = build_feature_frame(train_df)
    val_features = build_feature_frame(val_df)
    test_features = build_feature_frame(test_df)
    columns = feature_names(train_features)
    X_train = train_features[columns].astype(float)
    X_val = val_features[columns].astype(float)
    X_test = test_features[columns].astype(float)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "dataset": {
            "source": str(DATA_DIR),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "random_state": RANDOM_STATE,
            "feature_count": len(columns),
            "feature_names": columns,
        },
        "horizons": {},
    }
    for horizon_name, steps in HORIZONS.items():
        target = f"y_t+{steps}_d0"
        model = get_model("xgboost", random_state=RANDOM_STATE)
        model.fit(X_train, train_df[target].astype(float).to_numpy())
        val_prediction = model.predict(X_val)
        test_prediction = model.predict(X_test)
        model_path = MODEL_DIR / f"xgboost_{horizon_name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        metrics["horizons"][horizon_name] = {
            "target": target,
            "model_type": "XGBRegressor",
            "model_path": str(model_path),
            "validation_metrics": compute_metrics(val_df[target].astype(float).to_numpy(), val_prediction),
            "test_metrics": compute_metrics(test_df[target].astype(float).to_numpy(), test_prediction),
            "prediction_shape": list(test_prediction.shape),
        }
    metrics_path = ARTIFACT_DIR / "phase7_9_reproduction_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = reproduce_training()
    for horizon, summary in result["horizons"].items():
        print(horizon, summary["test_metrics"])
