from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.nowcasting import CongestionThresholds, generate_full_phase4_evaluation

DATA_DIR = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la")


def load_split(file_name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / file_name)
    if "t0_timestamp" in df.columns:
        df["t0_timestamp"] = pd.to_datetime(df["t0_timestamp"], errors="coerce")
    return df.copy()


def main() -> None:
    train_df = load_split("train.parquet")
    test_df = load_split("test.parquet")

    thresholds = CongestionThresholds(low=0.25, moderate=0.50, high=0.75, severe=1.00)
    evaluation = generate_full_phase4_evaluation(train_df=train_df, test_df=test_df, thresholds=thresholds)

    print("\n=== Phase 4 Nowcasting Validation Summary ===\n")
    for horizon_name, result in evaluation.items():
        metrics = result["metrics"]
        print(
            f"{horizon_name:7s} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
            f"R2={metrics['r2']:.4f} | Rows={result['validation_rows']}"
        )
        print(f"           report: {result['report_path']}")

    print("\nThresholds: ", thresholds.model_dump())
    print("\nCompleted Phase 4 nowcasting evaluation.")


if __name__ == "__main__":
    main()
