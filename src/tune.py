import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

# Import secure saving utility
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

def tune_models():
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Sub-sample for expensive kernels (SVR)
    sub_size = min(6000, len(X_train))
    X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=sub_size, random_state=42)
    
    tuned_results = []
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Tune Ridge Regression
    print("\n--- Tuning Ridge Regression ---")
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(random_state=42))
    ])
    ridge_grid = {'regressor__alpha': [0.01, 0.1, 1.0, 10.0]}
    ridge_search = GridSearchCV(ridge_pipeline, ridge_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_search.fit(X_train, y_train)
    y_pred = ridge_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tuned_results.append({"Model": "Ridge Regression (Tuned)", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Best Params": str(ridge_search.best_params_)})
    save_secure_model(ridge_search.best_estimator_, os.path.join(artifacts_dir, "ridge_tuned.joblib"))
    
    # 2. Tune Decision Tree
    print("\n--- Tuning Decision Tree ---")
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    dt_grid = {
        'regressor__max_depth': [15, 25, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }
    dt_search = GridSearchCV(dt_pipeline, dt_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    dt_search.fit(X_train, y_train)
    y_pred = dt_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tuned_results.append({"Model": "Decision Tree (Tuned)", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Best Params": str(dt_search.best_params_)})
    save_secure_model(dt_search.best_estimator_, os.path.join(artifacts_dir, "decision_tree_tuned.joblib"))
    
    # 3. Tune Random Forest
    print("\n--- Tuning Random Forest ---")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rf_dist = {
        'regressor__n_estimators': [100, 150, 200],
        'regressor__max_depth': [20, 30, None],
        'regressor__min_samples_split': [2, 4],
        'regressor__min_samples_leaf': [1, 2]
    }
    rf_search = RandomizedSearchCV(rf_pipeline, rf_dist, n_iter=6, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    y_pred = rf_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tuned_results.append({"Model": "Random Forest (Tuned)", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Best Params": str(rf_search.best_params_)})
    save_secure_model(rf_search.best_estimator_, os.path.join(artifacts_dir, "random_forest_tuned.joblib"))
    
    # 4. Tune Gradient Boosting
    print("\n--- Tuning Gradient Boosting ---")
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    gb_dist = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1, 0.15],
        'regressor__max_depth': [4, 6, 8]
    }
    gb_search = RandomizedSearchCV(gb_pipeline, gb_dist, n_iter=4, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    gb_search.fit(X_train, y_train)
    y_pred = gb_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tuned_results.append({"Model": "Gradient Boosting (Tuned)", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Best Params": str(gb_search.best_params_)})
    save_secure_model(gb_search.best_estimator_, os.path.join(artifacts_dir, "gradient_boosting_tuned.joblib"))
    
    # 5. Tune KNN
    print("\n--- Tuning KNN ---")
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(weights='distance'))
    ])
    knn_grid = {'regressor__n_neighbors': [3, 5, 7, 9]}
    knn_search = GridSearchCV(knn_pipeline, knn_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    knn_search.fit(X_train, y_train)
    y_pred = knn_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tuned_results.append({"Model": "K-Nearest Neighbors (Tuned)", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Best Params": str(knn_search.best_params_)})
    save_secure_model(knn_search.best_estimator_, os.path.join(artifacts_dir, "knn_tuned.joblib"))
    
    # 6. Tune SVR
    print("\n--- Tuning SVR ---")
    svr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(max_iter=20000))
    ])
    svr_dist = {
        'regressor__C': [1.0, 10.0, 50.0],
        'regressor__epsilon': [0.05, 0.1],
        'regressor__kernel': ['rbf']
    }
    svr_search = GridSearchCV(svr_pipeline, svr_dist, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    svr_search.fit(X_train_sub, y_train_sub)
    y_pred = svr_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tuned_results.append({"Model": "Support Vector Regressor (Tuned)", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Best Params": str(svr_search.best_params_)})
    save_secure_model(svr_search.best_estimator_, os.path.join(artifacts_dir, "svr_tuned.joblib"))
    
    tuned_df = pd.DataFrame(tuned_results)
    print("\n=== Tuned Model Comparison Table ===")
    print(tuned_df.to_string(index=False))
    
    tuned_report_path = os.path.join(project_dir, "reports", "tuned_model_comparison.csv")
    tuned_df.to_csv(tuned_report_path, index=False)
    print(f"\nTuned comparisons saved to: {tuned_report_path}")

if __name__ == "__main__":
    tune_models()
