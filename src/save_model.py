import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Import secure saving utility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from security_utils import save_secure_model, load_secure_model

def load_data():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "processed", "engineered_air_quality.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}")
    return pd.read_csv(data_path)

def get_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def save_final_model():
    df = load_data()
    
    numeric_features = [
        'pm25_median', 'pm10_median', 'no2_median', 'so2_median', 'co_median', 'o3_median',
        'temperature_median', 'humidity_median', 'wind-speed_median', 
        'PM25_PM10_ratio', 'PM25_Wind_Ratio', 'Stagnation_Index', 
        'Gaseous_Index', 'Thermal_Humidity_Index',
        'Month', 'DayOfWeek', 'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos'
    ]
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = ['Season']
    
    X = df[numeric_features + categorical_features]
    y = df['AQI']
    
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Champion Model: Tuned Random Forest Regressor
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=150,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("[SAVE MODEL] Training champion model on complete dataset...")
    final_model.fit(X, y)
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_dir, "artifacts")
    dest_path = os.path.join(artifacts_dir, "model.joblib")
    
    # Securely save with SHA-256 integrity hash
    save_secure_model(final_model, dest_path)
    
    # Verify secure deserialization
    print("\n[SECURITY VERIFY] Verifying cryptographic model loading...")
    loaded_model = load_secure_model(dest_path, enforce_hash_check=True)
    
    # Test dummy inference
    sample_row = pd.DataFrame([{
        'pm25_median': 35.0,
        'pm10_median': 55.0,
        'no2_median': 12.0,
        'so2_median': 2.5,
        'co_median': 1.8,
        'o3_median': 25.0,
        'temperature_median': 22.0,
        'humidity_median': 60.0,
        'wind-speed_median': 3.2,
        'PM25_PM10_ratio': 35.0 / 55.0,
        'PM25_Wind_Ratio': 35.0 / 3.2,
        'Stagnation_Index': 35.0 / (3.2 + 0.1),
        'Gaseous_Index': 12.0 + 2.5 + 18.0,
        'Thermal_Humidity_Index': 22.0 * 0.60,
        'Month': 8,
        'DayOfWeek': 5,
        'Month_Sin': np.sin(2 * np.pi * 8 / 12.0),
        'Month_Cos': np.cos(2 * np.pi * 8 / 12.0),
        'DayOfWeek_Sin': np.sin(2 * np.pi * 5 / 7.0),
        'DayOfWeek_Cos': np.cos(2 * np.pi * 5 / 7.0),
        'Season': 'Summer'
    }])
    
    pred = loaded_model.predict(sample_row)[0]
    print(f"[TEST INFERENCE] Test prediction: AQI = {pred:.2f}")

if __name__ == "__main__":
    save_final_model()
