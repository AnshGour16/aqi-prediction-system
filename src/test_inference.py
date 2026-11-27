"""End-to-end inference checks for the signed champion model.

Unlike the previous version, this script asserts real expectations: the CPCB
category derived from the inputs, the model's predicted category, and the
agreement between the two. It exits non-zero on failure so CI can use it.
"""

import os
import sys
import datetime

import numpy as np

# Import secure loader and shared helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aqi_utils import compute_aqi_from_concentrations, map_aqi_to_category
from feature_builder import build_feature_frame
from security_utils import load_secure_model

#: Maximum tolerated gap, in AQI points, between the model and CPCB maths.
MODEL_TOLERANCE = 20.0


def run_tests():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, "artifacts", "model.joblib")

    print(f"[TEST INFERENCE] Loading signed champion model: {model_path}")
    model = load_secure_model(model_path, enforce_hash_check=True)

    test_cases = [
        {
            "name": "Case 1: Pristine Spring Air",
            "values": {
                "pm25": 15.0, "pm10": 28.0, "no2": 12.0, "so2": 8.0,
                "co": 0.5, "o3": 22.0, "temp": 22.0, "humidity": 45.0,
                "wind_speed": 5.5,
            },
            "date": datetime.date(2024, 4, 10),
            "expected_category": "Good",
        },
        {
            "name": "Case 2: Heavy Winter Smog",
            "values": {
                "pm25": 320.0, "pm10": 480.0, "no2": 120.0, "so2": 45.0,
                "co": 15.0, "o3": 35.0, "temp": 5.0, "humidity": 88.0,
                "wind_speed": 0.8,
            },
            "date": datetime.date(2024, 12, 18),
            "expected_category": "Severe",
        },
        {
            "name": "Case 3: Summer Dust Storm",
            "values": {
                "pm25": 75.0, "pm10": 180.0, "no2": 45.0, "so2": 22.0,
                "co": 2.5, "o3": 85.0, "temp": 38.0, "humidity": 35.0,
                "wind_speed": 4.2,
            },
            "date": datetime.date(2024, 6, 14),
            "expected_category": "Moderate",
        },
    ]

    failures = []
    print("\n--- Running End-to-End Inference Verification ---")
    for case in test_cases:
        values = case["values"]
        input_df = build_feature_frame(values, observation_date=case["date"])

        cpcb_aqi = compute_aqi_from_concentrations(values)
        cpcb_category = map_aqi_to_category(cpcb_aqi)

        raw_prediction = float(model.predict(input_df)[0])
        predicted_aqi = round(float(np.clip(raw_prediction, 0.0, 500.0)), 1)
        predicted_category = map_aqi_to_category(predicted_aqi)

        print(f"\nScenario: {case['name']}")
        print(f"  CPCB AQI      : {cpcb_aqi:.1f} ({cpcb_category})")
        print(f"  Model estimate: {predicted_aqi:.1f} ({predicted_category})")
        print(f"  Difference    : {abs(predicted_aqi - cpcb_aqi):.1f} AQI points")

        if cpcb_category != case["expected_category"]:
            failures.append(
                f"{case['name']}: CPCB category {cpcb_category!r} != "
                f"expected {case['expected_category']!r}"
            )
        if predicted_category != case["expected_category"]:
            failures.append(
                f"{case['name']}: model category {predicted_category!r} != "
                f"expected {case['expected_category']!r}"
            )
        if abs(predicted_aqi - cpcb_aqi) > MODEL_TOLERANCE:
            failures.append(
                f"{case['name']}: model differs from CPCB by "
                f"{abs(predicted_aqi - cpcb_aqi):.1f} points "
                f"(tolerance {MODEL_TOLERANCE})"
            )
        if not 0.0 <= predicted_aqi <= 500.0:
            failures.append(f"{case['name']}: prediction {predicted_aqi} outside 0-500")

    if failures:
        print("\n[FAILED] Inference checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\n[PASSED] All inference, category and cryptographic checks passed.")


if __name__ == "__main__":
    run_tests()
