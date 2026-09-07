import os
import pandas as pd
import numpy as np

def audit_dataset():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    synth_path = os.path.join(project_dir, "data", "raw", "synthetic_air_quality_data.csv")
    raw_path = os.path.join(project_dir, "data", "raw", "city_pollution_data.csv")
    
    if os.path.exists(synth_path):
        data_path = synth_path
    elif os.path.exists(raw_path):
        data_path = raw_path
    else:
        raise FileNotFoundError("No raw or synthetic dataset found.")
        
    df = pd.read_csv(data_path)
    total_rows, total_cols = df.shape
    duplicates = df.duplicated().sum()
    
    report = []
    report.append("# Comprehensive Dataset Audit Report\n")
    report.append(f"- **Dataset Source:** `{os.path.basename(data_path)}`")
    report.append(f"- **Total Rows:** {total_rows}")
    report.append(f"- **Total Columns:** {total_cols}")
    report.append(f"- **Duplicate Rows:** {duplicates}")
    report.append(f"- **Missing AQI Rows:** {df['AQI'].isna().sum() if 'AQI' in df.columns else 'N/A'}\n")
    
    report.append("## Feature Nullability & Data Types\n")
    null_info = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100.0
        null_info.append({
            "Column Name": f"`{col}`",
            "Data Type": str(df[col].dtype),
            "Non-Null Count": total_rows - null_count,
            "Missing Count": null_count,
            "Missing %": f"{null_pct:.2f}%"
        })
    null_df = pd.DataFrame(null_info)
    report.append(null_df.to_markdown(index=False) + "\n")
    
    report.append("## Statistical Feature Distributions\n")
    report.append(df.describe().to_markdown() + "\n")
    
    if 'AQI' in df.columns:
        def map_cat(val):
            if val <= 50: return 'Good (0-50)'
            elif val <= 100: return 'Satisfactory (51-100)'
            elif val <= 200: return 'Moderate (101-200)'
            elif val <= 300: return 'Poor (201-300)'
            elif val <= 400: return 'Very Poor (301-400)'
            else: return 'Severe (>400)'
            
        cats = df['AQI'].apply(map_cat).value_counts()
        cat_df = pd.DataFrame({
            "Category": cats.index,
            "Count": cats.values,
            "Percentage": [f"{v/total_rows*100:.2f}%" for v in cats.values]
        })
        report.append("## Target AQI Category Balance\n")
        report.append(cat_df.to_markdown(index=False) + "\n")
        
    report_path = os.path.join(project_dir, "reports", "dataset_audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"[AUDIT] Audit report successfully written to: {report_path}")

if __name__ == "__main__":
    audit_dataset()
