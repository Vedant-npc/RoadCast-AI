from __future__ import annotations

from pathlib import Path

from src.analysis.eda import run_eda
from src.data.preprocessing import inspect_split, preprocess_metrla_data


DATA_DIR = Path(r"d:\My_Projects\RoadCast AI\data\raw\metr-la")


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "val.parquet"
    test_path = DATA_DIR / "test.parquet"

    print("Loading and preprocessing METR-LA datasets...")
    processed = preprocess_metrla_data(train_path, val_path, test_path)

    for name, df in {"train": processed.train, "val": processed.val, "test": processed.test}.items():
        report = inspect_split(df, name)
        print(f"\n{name}: {report['rows']} rows, {len(report['columns'])} columns")
        print(f"node_count={report['node_count']}")
        print(f"duplicate_rows={report['duplicate_rows']}")
        print(f"timestamp_min={report['timestamp_min']}")
        print(f"timestamp_max={report['timestamp_max']}")

        stats = run_eda(df, name)
        print(f"eda_summary={stats}")

    print("\nPreprocessing and EDA completed successfully.")


if __name__ == "__main__":
    main()
