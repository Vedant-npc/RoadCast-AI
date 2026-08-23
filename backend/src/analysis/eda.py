from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path(r"d:\My_Projects\RoadCast AI\backend\artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_plot_dir(name: str) -> Path:
    target = OUTPUT_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_speed_distribution(df: pd.DataFrame, name: str) -> None:
    speed_col = "x_t+0_d0"
    if speed_col not in df.columns:
        return
    sample = df[[speed_col]].sample(n=min(len(df), 200000), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(sample[speed_col], bins=50, kde=True, ax=ax)
    ax.set_title(f"{name}: Current Speed Distribution")
    ax.set_xlabel("Speed (x_t+0_d0)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "speed_distribution.png", dpi=150)
    plt.close(fig)


def plot_average_speed_over_time(df: pd.DataFrame, name: str, ts_col: str = "t0_timestamp") -> None:
    if ts_col not in df.columns:
        return
    speed_col = "x_t+0_d0"
    if speed_col not in df.columns:
        return
    sample = df[[ts_col, speed_col]].copy().sample(n=min(len(df), 200000), random_state=42)
    sample[ts_col] = pd.to_datetime(sample[ts_col])
    series = sample.groupby(pd.Grouper(key=ts_col, freq="H"))[speed_col].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    series.plot(ax=ax, color="steelblue")
    ax.set_title(f"{name}: Average Speed Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average speed")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "average_speed_over_time.png", dpi=150)
    plt.close(fig)


def plot_avg_by_hour(df: pd.DataFrame, name: str, ts_col: str = "t0_timestamp") -> None:
    if ts_col not in df.columns:
        return
    temp = df[[ts_col]].copy()
    temp[ts_col] = pd.to_datetime(temp[ts_col])
    temp["hour"] = temp[ts_col].dt.hour
    temp = temp.groupby("hour").size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=temp, x="hour", y="count", ax=ax, color="darkorange")
    ax.set_title(f"{name}: Average Activity by Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "activity_by_hour.png", dpi=150)
    plt.close(fig)


def plot_avg_by_day_of_week(df: pd.DataFrame, name: str, ts_col: str = "t0_timestamp") -> None:
    if ts_col not in df.columns:
        return
    temp = df[[ts_col]].copy()
    temp[ts_col] = pd.to_datetime(temp[ts_col])
    temp["day_of_week"] = temp[ts_col].dt.day_name()
    counts = temp["day_of_week"].value_counts().reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", color="mediumseagreen", ax=ax)
    ax.set_title(f"{name}: Activity by Day of Week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "activity_by_day_of_week.png", dpi=150)
    plt.close(fig)


def plot_sensor_activity(df: pd.DataFrame, name: str, sensor_col: str = "node_id") -> None:
    if sensor_col not in df.columns:
        return
    counts = df[sensor_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="slateblue")
    ax.set_title(f"{name}: Sensor/Node Activity")
    ax.set_xlabel("Sensor / node id")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "sensor_activity.png", dpi=150)
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, name: str) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    missing.plot(kind="bar", ax=ax, color="crimson")
    ax.set_title(f"{name}: Missing Value Counts")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing values")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "missing_values.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, name: str) -> None:
    selected = [
        "x_t-11_d0",
        "x_t-5_d0",
        "x_t+0_d0",
        "y_t+1_d0",
        "y_t+3_d0",
        "y_t+6_d0",
        "y_t+12_d0",
    ]
    numeric = df[[col for col in selected if col in df.columns]].copy()
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{name}: Speed Correlation Matrix")
    fig.tight_layout()
    fig.savefig(safe_plot_dir(name) / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, name: str) -> Dict[str, object]:
    plot_speed_distribution(df, name)
    plot_average_speed_over_time(df, name)
    plot_avg_by_hour(df, name)
    plot_avg_by_day_of_week(df, name)
    plot_sensor_activity(df, name)
    plot_missing_values(df, name)
    plot_correlation(df, name)

    stats = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "node_count": int(df["node_id"].nunique()) if "node_id" in df.columns else None,
        "timestamp_min": str(df["t0_timestamp"].min()) if "t0_timestamp" in df.columns else None,
        "timestamp_max": str(df["t0_timestamp"].max()) if "t0_timestamp" in df.columns else None,
    }
    return stats
