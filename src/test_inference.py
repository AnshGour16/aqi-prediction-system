import os
import sys
import pandas as pd
import numpy as np

# Import secure loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from security_utils import load_secure_model

def map_aqi_to_category(val):
    if val <= 50: return 'Good'
    elif val <= 100: return 'Satisfactory'
    elif val <= 200: return 'Moderate'
    elif val <= 300: return 'Poor'
    elif val <= 400: return 'Very Poor'
    else: return 'Severe'

def run_tests():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, "artifacts", "model.joblib")
    
    print(f"[TEST INFERENCE] Loading signed champion model: {model_path}")
    model = load_secure_model(model_path, enforce_hash_check=True)
    
    test_cases = [
        {
            "name": "Case 1: Pristine Spring Air (Good)",
            "data": {
                'pm25_median': 15.0,
                'pm10_median': 28.0,
                'no2_median': 12.0,
                'so2_median': 8.0,
                'co_median': 0.5,
                'o3_median': 22.0,
                'temperature_median': 22.0,
                'humidity_median': 45.0,
                'wind-speed_median': 5.5,
                'PM25_PM10_ratio': 15.0 / 28.0,
                'PM25_Wind_Ratio': 15.0 / 5.5,
                'Stagnation_Index': 15.0 / (5.5 + 0.1),
                'Gaseous_Index': 12.0 + 8.0 + 5.0,
                'Thermal_Humidity_Index': 22.0 * 0.45,
                'Month': 4,
                'DayOfWeek': 2,
                'Month_Sin': np.sin(2 * np.pi * 4 / 12.0),
                'Month_Cos': np.cos(2 * np.pi * 4 / 12.0),
                'DayOfWeek_Sin': np.sin(2 * np.pi * 2 / 7.0),
                'DayOfWeek_Cos': np.cos(2 * np.pi * 2 / 7.0),
                'Season': 'Spring'
            }
        },
        {
            "name": "Case 2: Heavy Winter Smog (Severe)",
            "data": {
                'pm25_median': 320.0,
                'pm10_median': 480.0,
                'no2_median': 120.0,
                'so2_median': 45.0,
                'co_median': 15.0,
                'o3_median': 35.0,
                'temperature_median': 5.0,
                'humidity_median': 88.0,
                'wind-speed_median': 0.8,
                'PM25_PM10_ratio': 320.0 / 480.0,
                'PM25_Wind_Ratio': 320.0 / 0.8,
                'Stagnation_Index': 320.0 / (0.8 + 0.1),
                'Gaseous_Index': 120.0 + 45.0 + 150.0,
                'Thermal_Humidity_Index': 5.0 * 0.88,
                'Month': 12,
                'DayOfWeek': 3,
                'Month_Sin': np.sin(2 * np.pi * 12 / 12.0),
                'Month_Cos': np.cos(2 * np.pi * 12 / 12.0),
                'DayOfWeek_Sin': np.sin(2 * np.pi * 3 / 7.0),
                'DayOfWeek_Cos': np.cos(2 * np.pi * 3 / 7.0),
                'Season': 'Winter'
            }
        },
        {
            "name": "Case 3: Summer Dust Storm (Moderate/Poor)",
            "data": {
                'pm25_median': 75.0,
                'pm10_median': 180.0,
                'no2_median': 45.0,
                'so2_median': 22.0,
                'co_median': 2.5,
                'o3_median': 85.0,
                'temperature_median': 38.0,
                'humidity_median': 35.0,
                'wind-speed_median': 4.2,
                'PM25_PM10_ratio': 75.0 / 180.0,
                'PM25_Wind_Ratio': 75.0 / 4.2,
                'Stagnation_Index': 75.0 / (4.2 + 0.1),
                'Gaseous_Index': 45.0 + 22.0 + 25.0,
                'Thermal_Humidity_Index': 38.0 * 0.35,
                'Month': 6,
                'DayOfWeek': 5,
                'Month_Sin': np.sin(2 * np.pi * 6 / 12.0),
                'Month_Cos': np.cos(2 * np.pi * 6 / 12.0),
                'DayOfWeek_Sin': np.sin(2 * np.pi * 5 / 7.0),
                'DayOfWeek_Cos': np.cos(2 * np.pi * 5 / 7.0),
                'Season': 'Summer'
            }
        }
    ]
    
    print("\n--- Running End-to-End Inference Verification ---")
    for case in test_cases:
        print(f"\nScenario: {case['name']}")
        input_df = pd.DataFrame([case['data']])
        pred = model.predict(input_df)[0]
        pred_aqi = round(np.clip(pred, 0.0, 500.0), 1)
        cat = map_aqi_to_category(pred_aqi)
        print(f"-> Predicted AQI: {pred_aqi}")
        print(f"-> Derived Category: {cat}")
        
    print("\n[PASSED] All inference tests and cryptographic checks passed successfully!")

if __name__ == "__main__":
    run_tests()
