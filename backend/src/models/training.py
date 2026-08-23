from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


MODEL_DIR = Path(r"d:\My_Projects\RoadCast AI\backend\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"d:\My_Projects\RoadCast AI\backend\artifacts\phase3")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = {
    "15min": 3,
    "30min": 6,
    "45min": 9,
    "60min": 12,
}


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "t0_timestamp" in out.columns:
        out["t0_timestamp"] = pd.to_datetime(out["t0_timestamp"], errors="coerce")
        out["hour"] = out["t0_timestamp"].dt.hour
        out["day_of_week"] = out["t0_timestamp"].dt.dayofweek
        out["weekend_flag"] = out["day_of_week"].ge(5).astype(int)
        out["minute_of_hour"] = out["t0_timestamp"].dt.minute
    return out


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in [1, 3, 5, 10]:
        prev_col = f"x_t-{lag}_d0"
        if prev_col in out.columns:
            out[f"trend_{lag}_d0"] = out["x_t+0_d0"] - out[prev_col]
    if "x_t+0_d0" in out.columns and "x_t-1_d0" in out.columns:
        out["rolling_mean_3_d0"] = out[[f"x_t-{lag}_d0" for lag in [1, 2, 3]]].mean(axis=1)
        out["rolling_mean_5_d0"] = out[[f"x_t-{lag}_d0" for lag in [1, 2, 3, 4, 5]]].mean(axis=1)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = build_time_features(df)
    out = build_trend_features(out)

    feature_cols = []
    for offset in range(-11, 1):
        offset_str = "+0" if offset == 0 else str(offset)
        for suffix in ["d0", "d1"]:
            col_name = f"x_t{offset_str}_{suffix}"
            if col_name in out.columns:
                feature_cols.append(col_name)

    feature_cols.extend(["hour", "day_of_week", "weekend_flag", "minute_of_hour"])
    for col in ["trend_1_d0", "trend_3_d0", "trend_5_d0", "trend_10_d0", "rolling_mean_3_d0", "rolling_mean_5_d0"]:
        if col in out.columns:
            feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    return out[feature_cols + ["node_id", "t0_timestamp"]]


def get_target_columns() -> Dict[str, str]:
    mapping = {}
    for name, steps in HORIZONS.items():
        target_col = f"y_t+{steps}_d0"
        mapping[name] = target_col
    return mapping


def get_model(name: str, random_state: int = 42):
    if name == "baseline":
        return "baseline"
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_pickle(obj, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, horizon: str, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_size = min(len(y_true), 2000)
    ax.scatter(y_true[:sample_size], y_pred[:sample_size], alpha=0.5, s=10)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    ax.set_title(f"{model_name.upper()} - {horizon} Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / f"{model_name}_{horizon}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: List[Dict[str, object]]) -> None:
    horizons = [res["horizon"] for res in results]
    model_names = sorted({res["model_name"] for res in results})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.22

    for idx, model_name in enumerate(model_names):
        values = []
        for horizon in horizons:
            metric_row = next(res for res in results if res["horizon"] == horizon and res["model_name"] == model_name)
            values.append(metric_row["rmse"])
        ax.bar(x + (idx - 1) * width, values, width=width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title("RMSE by Horizon and Model")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "model_comparison_rmse.png", dpi=150)
    plt.close(fig)


def train_and_evaluate(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    target_map = get_target_columns()
    results = []
    best_models = {}

    for horizon_name, steps in HORIZONS.items():
        target_col = target_map[horizon_name]
        feature_df_train = build_feature_frame(train_df)
        feature_df_val = build_feature_frame(val_df)
        feature_df_test = build_feature_frame(test_df)

        feature_cols = [
            col
            for col in feature_df_train.columns
            if col not in {"node_id", "t0_timestamp"} and not col.startswith("y_t+")
        ]

        X_train = feature_df_train[feature_cols].astype(float)
        X_val = feature_df_val[feature_cols].astype(float)
        X_test = feature_df_test[feature_cols].astype(float)

        y_train = train_df[target_col].astype(float).to_numpy()
        y_val = val_df[target_col].astype(float).to_numpy()
        y_test = test_df[target_col].astype(float).to_numpy()

        model_names = ["baseline", "random_forest", "xgboost"]
        horizon_results = []

        for model_name in model_names:
            model = get_model(model_name)
            if model_name == "baseline":
                val_pred = X_val["x_t+0_d0"].to_numpy()
                test_pred = X_test["x_t+0_d0"].to_numpy()
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
                save_pickle(model, MODEL_DIR / f"{model_name}_{horizon_name}.pkl")

            val_metrics = compute_metrics(y_val, val_pred)
            test_metrics = compute_metrics(y_test, test_pred)

            row = {
                "horizon": horizon_name,
                "model_name": model_name,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_pred": val_pred,
                "test_pred": test_pred,
            }
            horizon_results.append(row)

            plot_actual_vs_predicted(y_test, test_pred, horizon_name, model_name)

            if model_name != "baseline":
                rank_key = val_metrics["rmse"]
                current_best = best_models.get(horizon_name)
                if current_best is None or rank_key < current_best["validation_metrics"]["rmse"]:
                    best_models[horizon_name] = row

        results.append({"horizon": horizon_name, "models": horizon_results})

    return {"results": results, "best_models": best_models}
