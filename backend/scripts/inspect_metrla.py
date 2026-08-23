from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la")


def summarize(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'=' * 120}\nFILE: {label}\n{'=' * 120}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate records: {int(df.duplicated().sum())}")

    ts_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower() or "timestamp" in col.lower()]
    print(f"\nTimestamp-like columns: {ts_cols}")
    for col in ts_cols:
        try:
            ts = pd.to_datetime(df[col], errors="coerce")
            valid = ts.notna()
            if valid.sum() > 0:
                print(f"  {col}: min={ts[valid].min()} max={ts[valid].max()} nulls={int((~valid).sum())}")
        except Exception:
            pass

    candidate_sensor_cols = [
        col for col in df.columns
        if any(token in col.lower() for token in ["sensor", "segment", "road", "node", "station", "id"])
    ]
    if candidate_sensor_cols:
        print("\nCandidate sensor/segment columns:")
        for col in candidate_sensor_cols:
            try:
                print(f"  {col}: unique={df[col].nunique(dropna=True)}")
            except Exception:
                pass

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    for file_name in ["train.parquet", "val.parquet", "test.parquet"]:
        path = BASE_DIR / file_name
        df = pd.read_parquet(path)
        summarize(df, file_name)
