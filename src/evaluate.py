import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Import secure loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from security_utils import load_secure_model

def load_data():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "processed", "engineered_air_quality.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}")
    return pd.read_csv(data_path)

def map_aqi_to_category(val):
    if pd.isna(val):
        return 'Missing'
    if 0 <= val <= 50:
        return 'Good'
    elif 51 <= val <= 100:
        return 'Satisfactory'
    elif 101 <= val <= 200:
        return 'Moderate'
    elif 201 <= val <= 300:
        return 'Poor'
    elif 301 <= val <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

def run_detailed_evaluation():
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
    y_test_categories = y_test.apply(map_aqi_to_category)
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_dir, "artifacts")
    
    models = {
        "Random Forest (Tuned)": "random_forest_tuned.joblib",
        "Gradient Boosting (Tuned)": "gradient_boosting_tuned.joblib",
        "Decision Tree (Tuned)": "decision_tree_tuned.joblib",
        "K-Nearest Neighbors (Tuned)": "knn_tuned.joblib",
        "Support Vector Regressor (Tuned)": "svr_tuned.joblib",
        "Ridge Regression (Tuned)": "ridge_tuned.joblib"
    }
    
    evaluation_results = []
    category_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    
    report_lines = []
    report_lines.append("# Detailed Model Evaluation Report\n")
    report_lines.append("This report evaluates the tuned models on both continuous regression metrics and derived CPCB AQI classification bands on the balanced dataset.\n")
    
    for model_name, filename in models.items():
        model_path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}, skipping.")
            continue
            
        print(f"[EVALUATE] Evaluating {model_name}...")
        pipeline = load_secure_model(model_path, enforce_hash_check=True)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        y_pred_categories = pd.Series(y_pred).apply(map_aqi_to_category)
        
        accuracy = accuracy_score(y_test_categories, y_pred_categories)
        precision = precision_score(y_test_categories, y_pred_categories, average='weighted', zero_division=0)
        recall = recall_score(y_test_categories, y_pred_categories, average='weighted', zero_division=0)
        f1 = f1_score(y_test_categories, y_pred_categories, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test_categories, y_pred_categories, labels=category_order)
        
        evaluation_results.append({
            "Model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Cat_Accuracy": round(accuracy, 4),
            "Cat_Precision": round(precision, 4),
            "Cat_Recall": round(recall, 4),
            "Cat_F1": round(f1, 4)
        })
        
        report_lines.append(f"## {model_name}\n")
        report_lines.append("### Regression Metrics")
        report_lines.append(f"- **MAE:** {mae:.4f}")
        report_lines.append(f"- **RMSE:** {rmse:.4f}")
        report_lines.append(f"- **R² Score:** {r2:.4f} ({r2*100:.2f}%)\n")
        
        report_lines.append("### Derived Classification Metrics")
        report_lines.append(f"- **Categorical Accuracy:** {accuracy*100:.2f}%")
        report_lines.append(f"- **Weighted Precision:** {precision:.4f}")
        report_lines.append(f"- **Weighted Recall:** {recall:.4f}")
        report_lines.append(f"- **Weighted F1-Score:** {f1:.4f}\n")
        
        report_lines.append("### Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in category_order], columns=[f"Pred {c}" for c in category_order])
        report_lines.append(cm_df.to_markdown())
        report_lines.append("\n---\n")
        
    summary_df = pd.DataFrame(evaluation_results)
    report_lines.insert(2, "## Overall Summary Table\n")
    report_lines.insert(3, summary_df.to_markdown() + "\n")
    
    report_path = os.path.join(project_dir, "reports", "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    csv_path = os.path.join(project_dir, "reports", "model_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    
    print(f"[EVALUATE] Evaluation complete!")
    print(f"- Report: {report_path}")
    print(f"- Summary CSV: {csv_path}")

if __name__ == "__main__":
    run_detailed_evaluation()
