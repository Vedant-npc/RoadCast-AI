from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.training import train_and_evaluate


DATA_DIR = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la")


def load_split(file_name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / file_name)
    return df.copy()


def main() -> None:
    train_df = load_split("train.parquet").sample(n=100000, random_state=42)
    val_df = load_split("val.parquet").sample(n=30000, random_state=42)
    test_df = load_split("test.parquet").sample(n=40000, random_state=42)

    for df in [train_df, val_df, test_df]:
        df["t0_timestamp"] = pd.to_datetime(df["t0_timestamp"], errors="coerce")

    result_bundle = train_and_evaluate(train_df, val_df, test_df)

    print("\n=== Phase 3 Model Evaluation Summary ===\n")
    for horizon_summary in result_bundle["results"]:
        horizon = horizon_summary["horizon"]
        print(f"Horizon: {horizon}")
        for model_result in horizon_summary["models"]:
            model_name = model_result["model_name"]
            val_metrics = model_result["validation_metrics"]
            test_metrics = model_result["test_metrics"]
            print(
                f"  {model_name:12s} | val MAE={val_metrics['mae']:.4f} | "
                f"val RMSE={val_metrics['rmse']:.4f} | val R2={val_metrics['r2']:.4f} | "
                f"test MAE={test_metrics['mae']:.4f} | test RMSE={test_metrics['rmse']:.4f} | test R2={test_metrics['r2']:.4f}"
            )
        best = result_bundle["best_models"][horizon]
        print(f"  Best model for {horizon}: {best['model_name']}")
        print()


if __name__ == "__main__":
    main()
