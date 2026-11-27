import os
import pandas as pd
import numpy as np

try:
    from .aqi_utils import compute_aqi_from_concentrations, map_aqi_to_category
except ImportError:  # Running as a script: `python src/data_preprocessing.py`
    from aqi_utils import compute_aqi_from_concentrations, map_aqi_to_category


def get_aqi(row):
    """Return the CPCB AQI for one dataframe row.

    At least three pollutants, including a particulate measure, must be
    present. Breakpoint maths lives in :mod:`aqi_utils` so the pipeline and
    the app can never drift apart.
    """
    values = {
        'PM2.5': row.get('pm25_median'),
        'PM10': row.get('pm10_median'),
        'NO2': row.get('no2_median'),
        'SO2': row.get('so2_median'),
        'CO': row.get('co_median'),
        'O3': row.get('o3_median'),
    }
    return compute_aqi_from_concentrations(values)

def clean_and_preprocess():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    synth_path = os.path.join(project_dir, "data", "raw", "synthetic_air_quality_data.csv")
    raw_path = os.path.join(project_dir, "data", "raw", "city_pollution_data.csv")
    
    if os.path.exists(synth_path):
        input_path = synth_path
        print(f"[PREPROCESS] Loading synthetic balanced dataset from: {input_path}")
    elif os.path.exists(raw_path):
        input_path = raw_path
        print(f"[PREPROCESS] Loading raw DEAP dataset from: {input_path}")
    else:
        raise FileNotFoundError(f"Neither synthetic nor raw dataset found in {os.path.join(project_dir, 'data', 'raw')}")
        
    processed_dir = os.path.join(project_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    dest_path = os.path.join(processed_dir, "cleaned_air_quality.csv")
    
    df = pd.read_csv(input_path)
    print(f"[PREPROCESS] Initial dataset shape: {df.shape}")
    
    # 1. Deduplication
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"[PREPROCESS] Dropped {duplicates_count} duplicate rows.")
        
    # 2. Date conversion
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    # 3. Handle physical limits (no negative concentrations)
    pollutant_cols = ['pm25_median', 'pm10_median', 'no2_median', 'so2_median', 'co_median', 'o3_median']
    for col in pollutant_cols:
        if col in df.columns:
            negative_mask = df[col] < 0
            if negative_mask.sum() > 0:
                df.loc[negative_mask, col] = np.nan
                print(f"[PREPROCESS] Replaced {negative_mask.sum()} negative values in `{col}` with NaN.")
                
    # 4. Meteorology validation
    if 'humidity_median' in df.columns:
        invalid_humidity = (df['humidity_median'] < 0) | (df['humidity_median'] > 100)
        if invalid_humidity.sum() > 0:
            df.loc[invalid_humidity, 'humidity_median'] = np.nan
            
    # 5. Calculate or verify AQI
    if 'AQI' not in df.columns or df['AQI'].isna().sum() > 0:
        print("[PREPROCESS] Calculating CPCB AQI target values...")
        df['AQI'] = df.apply(get_aqi, axis=1)
        
    df.dropna(subset=['AQI'], inplace=True)

    # 6. Keep the category label derived from the AQI target so downstream
    # evaluation and the app can never disagree with the numeric value.
    derived_categories = df['AQI'].apply(map_aqi_to_category)
    if 'Category' in df.columns:
        mismatches = int((derived_categories != df['Category']).sum())
        if mismatches:
            print(f"[PREPROCESS] Corrected {mismatches} Category labels to match AQI.")
    df['Category'] = derived_categories

    df.to_csv(dest_path, index=False)
    print(f"[PREPROCESS] Cleaned dataset saved successfully to: {dest_path}")
    print(f"[PREPROCESS] Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    clean_and_preprocess()
