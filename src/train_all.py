import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to sys.path for security_utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from security_utils import save_secure_model

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

def train_and_evaluate_all():
    df = load_data()
    
    numeric_features = [
        'pm25_median', 'pm10_median', 'no2_median', 'so2_median', 'co_median', 'o3_median',
        'temperature_median', 'humidity_median', 'wind-speed_median', 
        'PM25_PM10_ratio', 'PM25_Wind_Ratio', 'Stagnation_Index', 
        'Gaseous_Index', 'Thermal_Humidity_Index',
        'Month', 'DayOfWeek', 'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos'
    ]
    # Filter features available in df
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = ['Season']
    
    X = df[numeric_features + categorical_features]
    y = df['AQI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    models = {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "Support Vector Regressor (SVR)": SVR(C=10.0, epsilon=0.1, max_iter=20000)
    }
    
    results = []
    trained_pipelines = {}
    
    print("[TRAIN ALL] Beginning multi-model training and evaluation...")
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")
        
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        inference_time = time.time() - start_time
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Evaluation: MAE = {mae:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f} ({r2*100:.2f}%)")
        
        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Train Time (s)": round(train_time, 2),
            "Inference Time (s)": round(inference_time, 2)
        })
        
        trained_pipelines[name] = pipeline
        
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison Table ===")
    print(results_df.to_string(index=False))
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(project_dir, "reports", "raw_model_comparison.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    results_df.to_csv(report_path, index=False)
    print(f"\nRaw results saved to: {report_path}")
    
    artifacts_dir = os.path.join(project_dir, "artifacts")
    for name, pipeline in trained_pipelines.items():
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_save_path = os.path.join(artifacts_dir, f"{clean_name}_raw.joblib")
        save_secure_model(pipeline, model_save_path)
        
    print("All raw model pipelines securely serialized with SHA-256 hashes.")

if __name__ == "__main__":
    train_and_evaluate_all()
