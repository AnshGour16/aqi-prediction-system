import os
import pandas as pd
import numpy as np

# Official Indian CPCB Breakpoints definition
BREAKPOINTS = {
    'PM2.5': [
        (0.0, 30.0, 0.0, 50.0),
        (30.1, 60.0, 51.0, 100.0),
        (60.1, 90.0, 101.0, 200.0),
        (90.1, 120.0, 201.0, 300.0),
        (120.1, 250.0, 301.0, 400.0),
        (250.1, 99999.0, 401.0, 500.0)
    ],
    'PM10': [
        (0.0, 50.0, 0.0, 50.0),
        (50.1, 100.0, 51.0, 100.0),
        (100.1, 250.0, 101.0, 200.0),
        (250.1, 350.0, 201.0, 300.0),
        (350.1, 430.0, 301.0, 400.0),
        (430.1, 99999.0, 401.0, 500.0)
    ],
    'NO2': [
        (0.0, 40.0, 0.0, 50.0),
        (40.1, 80.0, 51.0, 100.0),
        (80.1, 180.0, 101.0, 200.0),
        (180.1, 280.0, 201.0, 300.0),
        (280.1, 400.0, 301.0, 400.0),
        (400.1, 99999.0, 401.0, 500.0)
    ],
    'SO2': [
        (0.0, 40.0, 0.0, 50.0),
        (40.1, 80.0, 51.0, 100.0),
        (80.1, 380.0, 101.0, 200.0),
        (381.0, 800.0, 201.0, 300.0),
        (801.0, 1600.0, 301.0, 400.0),
        (1601.0, 99999.0, 401.0, 500.0)
    ],
    'CO': [
        (0.0, 1.0, 0.0, 50.0),
        (1.01, 2.0, 51.0, 100.0),
        (2.01, 10.0, 101.0, 200.0),
        (10.01, 17.0, 201.0, 300.0),
        (17.01, 34.0, 301.0, 400.0),
        (34.01, 99999.0, 401.0, 500.0)
    ],
    'O3': [
        (0.0, 50.0, 0.0, 50.0),
        (50.1, 100.0, 51.0, 100.0),
        (100.1, 168.0, 101.0, 200.0),
        (168.1, 208.0, 201.0, 300.0),
        (208.1, 748.0, 301.0, 400.0),
        (748.1, 99999.0, 401.0, 500.0)
    ]
}

def calculate_sub_index(val, pollutant):
    if pd.isna(val) or val < 0:
        return np.nan
    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= val <= c_high:
            ans = i_low + (i_high - i_low) / (c_high - c_low) * (val - c_low)
            return round(ans, 2)
    return 500.0

def get_aqi(row):
    sub_indices = {}
    if not pd.isna(row.get('pm25_median')):
        sub_indices['PM2.5'] = calculate_sub_index(row['pm25_median'], 'PM2.5')
    if not pd.isna(row.get('pm10_median')):
        sub_indices['PM10'] = calculate_sub_index(row['pm10_median'], 'PM10')
    if not pd.isna(row.get('no2_median')):
        # If in ppb, scale factor 1.88, if already ug/m3, sub index applies directly
        sub_indices['NO2'] = calculate_sub_index(row['no2_median'], 'NO2')
    if not pd.isna(row.get('so2_median')):
        sub_indices['SO2'] = calculate_sub_index(row['so2_median'], 'SO2')
    if not pd.isna(row.get('co_median')):
        sub_indices['CO'] = calculate_sub_index(row['co_median'], 'CO')
    if not pd.isna(row.get('o3_median')):
        sub_indices['O3'] = calculate_sub_index(row['o3_median'], 'O3')
        
    has_pm = ('PM2.5' in sub_indices) or ('PM10' in sub_indices)
    if len(sub_indices) >= 3 and has_pm:
        return max(sub_indices.values())
    return np.nan

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
    
    df.to_csv(dest_path, index=False)
    print(f"[PREPROCESS] Cleaned dataset saved successfully to: {dest_path}")
    print(f"[PREPROCESS] Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    clean_and_preprocess()
