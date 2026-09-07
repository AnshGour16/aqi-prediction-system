import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "processed", "engineered_air_quality.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}")
    return pd.read_csv(data_path)

def get_preprocessing_pipeline(numeric_features, categorical_features):
    # Numeric pipeline: Impute with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def run_baseline_training():
    df = load_data()
    
    # Define features and target
    numeric_features = [
        'pm25_median', 'pm10_median', 'no2_median', 'so2_median', 'co_median', 
        'temperature_median', 'humidity_median', 'wind-speed_median', 
        'PM25_PM10_ratio', 'PM25_Wind_Ratio', 'Month', 'DayOfWeek'
    ]
    categorical_features = ['Season']
    
    X = df[numeric_features + categorical_features]
    y = df['AQI']
    
    # Train-test split (80/20, reproducible random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Baseline Model: Ridge Regression (Linear Regression with L2 Regularization)
    baseline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0, random_state=42))
    ])
    
    print("\nTraining Baseline Ridge Regression Model...")
    baseline_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = baseline_model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Baseline Model (Ridge Regression) Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    # Save the pipeline placeholder in artifacts
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "baseline_model.joblib")
    joblib.dump(baseline_model, model_path)
    print(f"\nBaseline model saved to: {model_path}")

if __name__ == "__main__":
    run_baseline_training()
