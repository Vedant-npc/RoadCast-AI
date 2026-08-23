from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class PreprocessResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, object]


def load_metrla_split(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df.copy()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def convert_timestamp(df: pd.DataFrame, ts_col: str = "t0_timestamp") -> pd.DataFrame:
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    # Remove impossible or invalid rows where timestamp is missing.
    if "t0_timestamp" in df.columns:
        df = df[df["t0_timestamp"].notna()].copy()

    # Fill numeric sensor values with a conservative strategy.
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def sort_chronologically(df: pd.DataFrame, ts_col: str = "t0_timestamp") -> pd.DataFrame:
    if ts_col in df.columns:
        df = df.sort_values(ts_col).reset_index(drop=True)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "node_id" in df.columns:
        df = df[df["node_id"].notna()].copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "t0_timestamp" in df.columns:
        df = df[df["t0_timestamp"].notna()].copy()

    return df


def preprocess_metrla_data(train_path: str | Path, val_path: str | Path, test_path: str | Path) -> PreprocessResult:
    train_df = load_metrla_split(train_path)
    val_df = load_metrla_split(val_path)
    test_df = load_metrla_split(test_path)

    processed = {}
    for name, df in {"train": train_df, "val": val_df, "test": test_df}.items():
        df = standardize_column_names(df)
        df = convert_timestamp(df)
        df = remove_duplicates(df)
        df = clean_missing_values(df)
        df = sort_chronologically(df)
        df = validate_data(df)
        processed[name] = df

    metadata = {
        "train_shape": processed["train"].shape,
        "val_shape": processed["val"].shape,
        "test_shape": processed["test"].shape,
        "timestamp_column": "t0_timestamp",
        "sensor_column": "node_id",
    }

    return PreprocessResult(
        train=processed["train"],
        val=processed["val"],
        test=processed["test"],
        metadata=metadata,
    )


def inspect_split(df: pd.DataFrame, name: str) -> Dict[str, object]:
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "timestamp_min": str(df["t0_timestamp"].min()) if "t0_timestamp" in df.columns else None,
        "timestamp_max": str(df["t0_timestamp"].max()) if "t0_timestamp" in df.columns else None,
        "node_count": int(df["node_id"].nunique()) if "node_id" in df.columns else None,
    }
