import os
import pandas as pd
import numpy as np

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def engineer_features():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "processed", "cleaned_air_quality.csv")
    dest_path = os.path.join(project_dir, "data", "processed", "engineered_air_quality.csv")
    
    print(f"[ENGINEERING] Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Date-based Features
    print("[ENGINEERING] Extracting calendar & cyclical features...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Season'] = df['Month'].apply(get_season)
    
    # Cyclical representations
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    
    # 2. Pollutant Ratio Features
    print("[ENGINEERING] Engineering pollutant ratios & physical indices...")
    # PM2.5 / PM10 ratio (fine to coarse particulate ratio)
    df['PM25_PM10_ratio'] = np.where(
        (df['pm10_median'] > 0) & (df['pm25_median'].notna()) & (df['pm10_median'].notna()),
        df['pm25_median'] / df['pm10_median'],
        0.5
    )
    df['PM25_PM10_ratio'] = df['PM25_PM10_ratio'].clip(0.0, 2.0)
    
    # 3. Atmospheric Dispersion & Stagnation Indices
    # Wind speed influence on PM2.5 dispersion
    df['PM25_Wind_Ratio'] = np.where(
        (df['wind-speed_median'] > 0) & (df['pm25_median'].notna()),
        df['pm25_median'] / df['wind-speed_median'],
        0.0
    )
    
    # Stagnation index: High PM with near-zero wind velocity indicates stagnant atmospheric inversion
    df['Stagnation_Index'] = np.where(
        df['pm25_median'].notna(),
        df['pm25_median'] / (df['wind-speed_median'].fillna(2.0) + 0.1),
        0.0
    )
    
    # 4. Gaseous Combustion Index
    # Combines toxic vehicular/industrial combustion gases: NO2, SO2, CO
    df['Gaseous_Index'] = (
        df['no2_median'].fillna(0.0) + 
        df['so2_median'].fillna(0.0) + 
        (df['co_median'].fillna(0.0) * 10.0)
    )
    
    # 5. Thermal Comfort / Humidity Interaction
    df['Thermal_Humidity_Index'] = df['temperature_median'].fillna(20.0) * (df['humidity_median'].fillna(50.0) / 100.0)
    
    # Save the engineered dataset
    df.to_csv(dest_path, index=False)
    print(f"[ENGINEERING] Engineered dataset saved to: {dest_path}")
    print(f"[ENGINEERING] Feature columns: {list(df.columns)}")
    print(f"[ENGINEERING] Shape: {df.shape}")
    return df

if __name__ == "__main__":
    engineer_features()
