import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    # Setup paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "processed", "cleaned_air_quality.csv")
    fig_dir = os.path.join(project_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Mapped columns
    key_pollutants = {
        'pm25_median': 'PM2.5',
        'pm10_median': 'PM10',
        'no2_median': 'NO2',
        'so2_median': 'SO2',
        'co_median': 'CO',
        'o3_median': 'O3'
    }
    
    weather_params = {
        'temperature_median': 'Temperature',
        'humidity_median': 'Humidity',
        'wind-speed_median': 'Wind Speed'
    }
    
    # 1. AQI Distribution
    print("Generating AQI distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AQI'], kde=True, bins=50, color='skyblue')
    plt.title('Distribution of Air Quality Index (AQI)')
    plt.xlabel('AQI')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, 'aqi_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pollutant Distributions
    print("Generating pollutant distribution plots...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
    for i, (col, name) in enumerate(key_pollutants.items()):
        sns.histplot(df[col].dropna(), kde=True, bins=40, ax=axes[i], color='forestgreen')
        axes[i].set_title(f'Distribution of {name}')
        axes[i].set_xlabel('Concentration')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pollutant_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Boxplots for Outliers
    print("Generating boxplots for pollutants...")
    plt.figure(figsize=(12, 6))
    df_melted = df[list(key_pollutants.keys())].rename(columns=key_pollutants).melt()
    sns.boxplot(x='variable', y='value', data=df_melted, palette='Set2')
    plt.title('Outliers in Pollutant Concentrations')
    plt.xlabel('Pollutant')
    plt.ylabel('Concentration')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(fig_dir, 'pollutant_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Heatmap
    print("Generating correlation heatmap...")
    cols_for_corr = list(key_pollutants.keys()) + list(weather_params.keys()) + ['AQI']
    rename_dict = {**key_pollutants, **weather_params, 'AQI': 'AQI'}
    corr_matrix = df[cols_for_corr].rename(columns=rename_dict).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
    plt.title('Correlation Matrix of Pollutants, Weather & AQI')
    plt.savefig(os.path.join(fig_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter plots: Pollutants vs AQI (taking a random sample of 2000 points to keep plots clean)
    print("Generating scatter plots: Pollutants vs AQI...")
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
    for i, (col, name) in enumerate(key_pollutants.items()):
        sns.scatterplot(x=col, y='AQI', data=sample_df, ax=axes[i], alpha=0.5, color='coral')
        sns.regplot(x=col, y='AQI', data=sample_df, ax=axes[i], scatter=False, color='red')
        axes[i].set_title(f'{name} vs AQI')
        axes[i].set_xlabel(f'{name} Concentration')
        axes[i].set_ylabel('AQI')
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pollutant_vs_aqi.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Scatter plots: Weather vs AQI
    print("Generating scatter plots: Weather vs AQI...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (col, name) in enumerate(weather_params.items()):
        sns.scatterplot(x=col, y='AQI', data=sample_df, ax=axes[i], alpha=0.5, color='purple')
        sns.regplot(x=col, y='AQI', data=sample_df, ax=axes[i], scatter=False, color='black')
        axes[i].set_title(f'{name} vs AQI')
        axes[i].set_xlabel(name)
        axes[i].set_ylabel('AQI')
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'weather_vs_aqi.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Time-based trends: Monthly Average AQI
    print("Generating time-based trend plots...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_aqi = df.groupby('Month')['AQI'].mean().reset_index()
    monthly_aqi['Month'] = monthly_aqi['Month'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Month', y='AQI', data=monthly_aqi, marker='o', color='darkorange', linewidth=2.5)
    plt.title('Monthly Average AQI Trend')
    plt.xlabel('Month')
    plt.ylabel('Average AQI')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, 'monthly_aqi_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    generate_plots()
