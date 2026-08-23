import pytest

from src.models.nowcasting import (
    CongestionThresholds,
    build_nowcast_output,
    calculate_congestion_score,
    classify_congestion,
)


@pytest.mark.parametrize(
    ("speed", "reference_speed", "expected"),
    [
        (70.0, 70.0, 0.0),
        (35.0, 70.0, 0.5),
        (0.0, 70.0, 1.0),
        (80.0, 70.0, 0.0),
    ],
)
def test_congestion_score_calculation(speed, reference_speed, expected):
    assert calculate_congestion_score(speed, reference_speed) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "LOW"),
        (0.249, "LOW"),
        (0.25, "MODERATE"),
        (0.5, "HIGH"),
        (0.75, "SEVERE"),
        (1.0, "SEVERE"),
    ],
)
def test_severity_classification(score, expected):
    thresholds = CongestionThresholds()
    assert classify_congestion(score, thresholds) == expected


def test_prediction_output_structure():
    payload = build_nowcast_output(
        current_speed=58.0,
        predicted_speeds={
            "15_min": 54.5,
            "30_min": 50.0,
            "45_min": 47.0,
            "60_min": 44.0,
        },
        reference_speed=70.0,
    )

    assert set(payload.keys()) == {"current", "15_min", "30_min", "45_min", "60_min"}
    assert payload["current"]["speed"] == pytest.approx(58.0)
    assert payload["15_min"]["predicted_speed"] == pytest.approx(54.5)
    assert payload["15_min"]["severity"] in {"LOW", "MODERATE", "HIGH", "SEVERE"}


def test_negative_predictions_are_clipped_before_score_calc():
    output = build_nowcast_output(
        current_speed=58.0,
        predicted_speeds={"15_min": -10.0, "30_min": 50.0, "45_min": 47.0, "60_min": 44.0},
        reference_speed=70.0,
    )

    assert output["15_min"]["predicted_speed"] == pytest.approx(0.0)
    assert output["15_min"]["congestion_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("speed, reference_speed", [(None, 70.0), (-1.0, 70.0), (50.0, 0.0), (50.0, -5.0)])
def test_invalid_inputs_rejected(speed, reference_speed):
    with pytest.raises(ValueError):
        calculate_congestion_score(speed, reference_speed)


def test_boundary_values_for_thresholds():
    thresholds = CongestionThresholds(low=0.2, moderate=0.4, high=0.7, severe=1.0)
    assert classify_congestion(0.2, thresholds) == "MODERATE"
    assert classify_congestion(0.4, thresholds) == "HIGH"
    assert classify_congestion(0.7, thresholds) == "SEVERE"
