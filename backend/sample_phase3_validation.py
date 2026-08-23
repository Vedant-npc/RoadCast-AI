import pandas as pd

from src.models.training import build_feature_frame, compute_metrics, get_model

DATA = r"d:\My_Projects\RoadCast AI\data\raw\metr-la"

train = pd.read_parquet(DATA + "\\train.parquet").sample(50000, random_state=42)
val = pd.read_parquet(DATA + "\\val.parquet").sample(20000, random_state=42)
test = pd.read_parquet(DATA + "\\test.parquet").sample(20000, random_state=42)

for df in [train, val, test]:
    df["t0_timestamp"] = pd.to_datetime(df["t0_timestamp"], errors="coerce")

for horizon_name, steps in {"15min": 3, "30min": 6, "45min": 9, "60min": 12}.items():
    target = f"y_t+{steps}_d0"
    feature_cols = [
        c for c in build_feature_frame(train).columns
        if c not in {"node_id", "t0_timestamp"}
    ]
    X_train = build_feature_frame(train)[feature_cols].astype(float)
    X_val = build_feature_frame(val)[feature_cols].astype(float)
    X_test = build_feature_frame(test)[feature_cols].astype(float)
    y_train = train[target].astype(float).to_numpy()
    y_val = val[target].astype(float).to_numpy()
    y_test = test[target].astype(float).to_numpy()

    print(f"\nHORIZON: {horizon_name}")
    for model_name in ["random_forest", "xgboost"]:
        model = get_model(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred)
        print(model_name, metrics)
