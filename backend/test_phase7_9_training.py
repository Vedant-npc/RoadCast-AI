from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase7_9_reproduction import MODEL_DIR, feature_names, load_versioned_model
from src.models.training import HORIZONS, build_feature_frame, compute_metrics


def test_feature_order_matches_existing_metrla_contract():
    frame = pd.DataFrame({
        "node_id": [1],
        "t0_timestamp": ["2012-01-01T00:00:00"],
        **{f"x_t{'+0' if offset == 0 else offset}_{suffix}": [1.0] for offset in range(-11, 1) for suffix in ("d0", "d1")},
    })
    names = feature_names(build_feature_frame(frame))
    assert len(names) == 34
    assert names[:2] == ["x_t-11_d0", "x_t-11_d1"]
    assert names[-6:] == ["trend_1_d0", "trend_3_d0", "trend_5_d0", "trend_10_d0", "rolling_mean_3_d0", "rolling_mean_5_d0"]


def test_versioned_models_load_after_reproduction():
    if not (MODEL_DIR / "xgboost_15min.pkl").exists():
        pytest.skip("Run phase7_9_reproduction.py before loading versioned models.")
    model = load_versioned_model("15min")
    assert model.n_features_in_ == 34


def test_unknown_horizon_is_rejected():
    with pytest.raises(ValueError):
        load_versioned_model("2min")


def test_prediction_shape_and_metric_contract():
    prediction = np.array([1.0, 2.0, 3.0])
    assert prediction.shape == (3,)
    metrics = compute_metrics(prediction, prediction)
    assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_invalid_timestamp_does_not_change_feature_contract():
    frame = pd.DataFrame({
        "node_id": [1],
        "t0_timestamp": ["invalid-timestamp"],
        **{f"x_t{'+0' if offset == 0 else offset}_{suffix}": [1.0] for offset in range(-11, 1) for suffix in ("d0", "d1")},
    })
    features = build_feature_frame(frame)
    assert len(feature_names(features)) == 34
    assert pd.isna(features.loc[0, "t0_timestamp"])


def test_all_required_horizons_are_defined():
    assert list(HORIZONS) == ["15min", "30min", "45min", "60min"]
    assert list(HORIZONS.values()) == [3, 6, 9, 12]
