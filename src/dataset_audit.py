import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aqi_utils import BREAKPOINTS, calculate_sub_index, map_aqi_to_category

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
    if os.path.basename(data_path) == "synthetic_air_quality_data.csv":
        report.append("- **Dataset Type:** Locally generated synthetic examples; not measured sensor observations.")
        report.append("- **AQI Target:** Derived from CPCB pollutant sub-indices by `src/generate_dataset.py`.")
    else:
        report.append("- **Dataset Type:** Public DEAP-derived raw data; validate pollutant units before training.")
    report.append(f"- **Total Rows:** {total_rows}")
    report.append(f"- **Total Columns:** {total_cols}")
    report.append(f"- **Duplicate Rows:** {duplicates}")
    report.append(f"- **Missing AQI Rows:** {df['AQI'].isna().sum() if 'AQI' in df.columns else 'N/A'}\n")

    if 'AQI' in df.columns:
        report.append("## Target Integrity Checks\n")
        if 'Category' in df.columns:
            derived = df['AQI'].apply(map_aqi_to_category)
            mismatches = int((derived != df['Category']).sum())
            report.append(f"- **Category/AQI mismatches:** {mismatches}")
        report.append(f"- **Rows at the 500 AQI cap:** {int((df['AQI'] >= 499.95).sum())}")
        if {'pm25_median', 'pm10_median'}.issubset(df.columns):
            inverted = int((df['pm10_median'] < df['pm25_median']).sum())
            report.append(f"- **Rows with PM10 < PM2.5:** {inverted}")

        # A concentration below a pollutant's top breakpoint must never score
        # the 500 cap. This is the regression check for the old 0.1-wide gaps.
        gap_caps = 0
        for pollutant, column in (
            ('PM2.5', 'pm25_median'), ('PM10', 'pm10_median'), ('NO2', 'no2_median'),
            ('SO2', 'so2_median'), ('CO', 'co_median'), ('O3', 'o3_median'),
        ):
            if column not in df.columns:
                continue
            top = BREAKPOINTS[pollutant][-1][1]
            below_top = df[column] < top
            capped = df.loc[below_top, column].apply(
                lambda value, p=pollutant: calculate_sub_index(value, p) == 500.0
            )
            gap_caps += int(capped.sum())
        report.append(f"- **Sub-index 500 values below a top breakpoint (gap bug):** {gap_caps}\n")
    
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
        cats = df['AQI'].apply(map_aqi_to_category).value_counts()
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
