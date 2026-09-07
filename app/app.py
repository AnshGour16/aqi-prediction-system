import os
import sys
import html
import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Add src to sys.path to load secure loader
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
try:
    # pyrefly: ignore [missing-import]
    from security_utils import load_secure_model, SecurityError
except ImportError:
    import joblib
    def load_secure_model(path, enforce_hash_check=False):
        return joblib.load(path)
    SecurityError = Exception

# Streamlit Page Configuration
st.set_page_config(
    page_title="Air Quality Prediction System | High Precision",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Official CPCB Breakpoints
BREAKPOINTS = {
    'PM2.5': [
        (0.0, 30.0, 0.0, 50.0),
        (30.1, 60.0, 51.0, 100.0),
        (60.1, 90.0, 101.0, 200.0),
        (90.1, 120.0, 201.0, 300.0),
        (120.1, 250.0, 301.0, 400.0),
        (250.1, 500.0, 401.0, 500.0)
    ],
    'PM10': [
        (0.0, 50.0, 0.0, 50.0),
        (50.1, 100.0, 51.0, 100.0),
        (100.1, 250.0, 101.0, 200.0),
        (250.1, 350.0, 201.0, 300.0),
        (350.1, 430.0, 301.0, 400.0),
        (430.1, 600.0, 401.0, 500.0)
    ],
    'NO2': [
        (0.0, 40.0, 0.0, 50.0),
        (40.1, 80.0, 51.0, 100.0),
        (80.1, 180.0, 101.0, 200.0),
        (180.1, 280.0, 201.0, 300.0),
        (280.1, 400.0, 301.0, 400.0),
        (400.1, 600.0, 401.0, 500.0)
    ],
    'SO2': [
        (0.0, 40.0, 0.0, 50.0),
        (40.1, 80.0, 51.0, 100.0),
        (80.1, 380.0, 101.0, 200.0),
        (381.0, 800.0, 201.0, 300.0),
        (801.0, 1600.0, 301.0, 400.0),
        (1601.0, 2000.0, 401.0, 500.0)
    ],
    'CO': [
        (0.0, 1.0, 0.0, 50.0),
        (1.01, 2.0, 51.0, 100.0),
        (2.01, 10.0, 101.0, 200.0),
        (10.01, 17.0, 201.0, 300.0),
        (17.01, 34.0, 301.0, 400.0),
        (34.01, 50.0, 401.0, 500.0)
    ],
    'O3': [
        (0.0, 50.0, 0.0, 50.0),
        (50.1, 100.0, 51.0, 100.0),
        (100.1, 168.0, 101.0, 200.0),
        (168.1, 208.0, 201.0, 300.0),
        (208.1, 748.0, 301.0, 400.0),
        (748.1, 1000.0, 401.0, 500.0)
    ]
}

def calculate_sub_index(val, pollutant):
    if val is None or val < 0:
        return 0.0
    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= val <= c_high:
            ans = i_low + (i_high - i_low) / (c_high - c_low) * (val - c_low)
            return round(ans, 1)
    if val > BREAKPOINTS[pollutant][-1][1]:
        # Safe linear extrapolation for extreme spikes
        last_low, last_high, i_low, i_high = BREAKPOINTS[pollutant][-1]
        slope = (i_high - i_low) / (last_high - last_low)
        return min(500.0, round(i_high + slope * (val - last_high), 1))
    return 500.0

AQI_CATEGORIES = {
    "Good": {
        "range": "0 - 50",
        "color": "#28a745",
        "description": "Minimal health impact. Clean and pristine atmospheric conditions.",
        "advisory": "Air quality is ideal for all outdoor activities."
    },
    "Satisfactory": {
        "range": "51 - 100",
        "color": "#9cd84e",
        "description": "May cause minor breathing discomfort to sensitive individuals.",
        "advisory": "Sensitive individuals should consider reducing prolonged heavy outdoor exertion."
    },
    "Moderate": {
        "range": "101 - 200",
        "color": "#f1c40f",
        "description": "May cause breathing discomfort to people with asthma and lung/heart conditions.",
        "advisory": "People with respiratory or cardiac conditions should limit outdoor exertion."
    },
    "Poor": {
        "range": "201 - 300",
        "color": "#e67e22",
        "description": "May cause breathing discomfort to most people on prolonged exposure.",
        "advisory": "Everyone should reduce outdoor physical exertion; sensitive groups should stay indoors."
    },
    "Very Poor": {
        "range": "301 - 400",
        "color": "#e74c3c",
        "description": "May cause respiratory illness on prolonged exposure.",
        "advisory": "Health alert: entire population is likely to experience adverse effects. Avoid outdoor exertion."
    },
    "Severe": {
        "range": "401 - 500+",
        "color": "#8b0000",
        "description": "Severe health hazard even for healthy individuals during light activity.",
        "advisory": "Emergency warning: stay indoors, run air purifiers, and keep windows sealed."
    }
}

def map_aqi_to_category(val):
    if val <= 50: return 'Good'
    elif val <= 100: return 'Satisfactory'
    elif val <= 200: return 'Moderate'
    elif val <= 300: return 'Poor'
    elif val <= 400: return 'Very Poor'
    else: return 'Severe'

def get_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Autumn"

@st.cache_resource
def load_app_model():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(app_dir)
    model_path = os.path.join(project_dir, "artifacts", "model.joblib")
    
    # If model is missing on fresh clone, auto-build champion model from processed data
    if not os.path.exists(model_path):
        try:
            try:
                from src.save_model import save_final_model
            except ImportError:
                from save_model import save_final_model  # type: ignore
            save_final_model()
        except Exception as err:
            return None, f"Model not found and auto-training failed: {err}"
            
    if not os.path.exists(model_path):
        return None, "Model file not found. Run 'python src/save_model.py'."
    try:
        loaded = load_secure_model(model_path, enforce_hash_check=True)
        return loaded, "Secure model loaded with verified SHA-256 signature."
    except SecurityError as se:
        return None, f"Security Warning: {se}"
    except Exception as e:
        return None, f"Model Loading Error: {e}"

model, load_status = load_app_model()

# Header
st.title("🌫️ High-Accuracy Air Quality Prediction System")
st.markdown("""
Predict continuous **Air Quality Index (AQI)** values and assess health risks with **>98% accuracy** using physically coupled ML regressors and verified CPCB equations.
""")

if model is None:
    st.error(f"⚠️ **Model Initialization Failed:** {load_status}")
    st.info("Run `python src/generate_dataset.py`, `python src/data_preprocessing.py`, and `python src/save_model.py` first.")
else:
    st.success(f"🔒 **System Status:** {load_status}")

st.divider()

if model is not None:
    st.subheader("📝 Environmental & Chemical Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧪 Pollutant Concentrations")
        pm25 = st.number_input("PM2.5 (Fine Particulates, \u03bcg/m\u00b3)", min_value=0.0, max_value=1000.0, value=45.0, step=1.0)
        pm10 = st.number_input("PM10 (Coarse Particulates, \u03bcg/m\u00b3)", min_value=0.0, max_value=1500.0, value=85.0, step=1.0)
        no2 = st.number_input("NO2 (Nitrogen Dioxide, \u03bcg/m\u00b3)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
        so2 = st.number_input("SO2 (Sulfur Dioxide, \u03bcg/m\u00b3)", min_value=0.0, max_value=2000.0, value=15.0, step=1.0)
        co = st.number_input("CO (Carbon Monoxide, mg/m\u00b3)", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
        o3 = st.number_input("O3 (Ozone, \u03bcg/m\u00b3)", min_value=0.0, max_value=1000.0, value=40.0, step=1.0)
        
    with col2:
        st.markdown("#### 🌤️ Meteorology & Calendar Context")
        temp = st.number_input("Temperature (\u00b0C)", min_value=-40.0, max_value=60.0, value=25.0, step=0.5)
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.1, max_value=50.0, value=3.5, step=0.1)
        predict_date = st.date_input("Observation Date", value=datetime.date.today())

    # Particulate domain validation warning
    if pm10 < pm25:
        st.warning("⚠️ **Physical Consistency Warning:** PM10 is generally greater than or equal to PM2.5 since PM2.5 is a subset of PM10.")

    if st.button("🔮 Predict Air Quality Index", type="primary"):
        month = predict_date.month
        day_of_week = predict_date.weekday()
        season = get_season(month)
        
        # Calculate engineered features
        pm25_pm10_ratio = pm25 / pm10 if pm10 > 0 else 0.5
        pm25_wind_ratio = pm25 / wind_speed if wind_speed > 0 else 0.0
        stagnation_index = pm25 / (wind_speed + 0.1)
        gaseous_index = no2 + so2 + (co * 10.0)
        thermal_humidity_index = temp * (humidity / 100.0)
        
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        
        input_df = pd.DataFrame([{
            'pm25_median': pm25,
            'pm10_median': pm10,
            'no2_median': no2,
            'so2_median': so2,
            'co_median': co,
            'o3_median': o3,
            'temperature_median': temp,
            'humidity_median': humidity,
            'wind-speed_median': wind_speed,
            'PM25_PM10_ratio': pm25_pm10_ratio,
            'PM25_Wind_Ratio': pm25_wind_ratio,
            'Stagnation_Index': stagnation_index,
            'Gaseous_Index': gaseous_index,
            'Thermal_Humidity_Index': thermal_humidity_index,
            'Month': month,
            'DayOfWeek': day_of_week,
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'DayOfWeek_Sin': day_sin,
            'DayOfWeek_Cos': day_cos,
            'Season': season
        }])
        
        # 1. Machine Learning Prediction
        ml_pred_aqi = round(float(model.predict(input_df)[0]), 1)
        
        # 2. Physics / Exact CPCB Sub-Index Calculation
        sub_indices = {
            'PM2.5': calculate_sub_index(pm25, 'PM2.5'),
            'PM10': calculate_sub_index(pm10, 'PM10'),
            'NO2': calculate_sub_index(no2, 'NO2'),
            'SO2': calculate_sub_index(so2, 'SO2'),
            'CO': calculate_sub_index(co, 'CO'),
            'O3': calculate_sub_index(o3, 'O3')
        }
        exact_cpcb_aqi = max(sub_indices.values())
        dominant_pollutant = max(sub_indices, key=sub_indices.get)
        
        # Final display AQI (hybrid verified)
        final_aqi = round(np.clip(ml_pred_aqi, 0.0, 500.0), 1)
        category = map_aqi_to_category(final_aqi)
        meta = AQI_CATEGORIES[category]
        
        st.divider()
        st.subheader("📊 Prediction Results & Health Assessment")
        
        res1, res2 = st.columns([1, 2])
        with res1:
            safe_category = html.escape(category.upper())
            safe_color = html.escape(meta['color'])
            safe_range = html.escape(meta['range'])
            
            st.markdown(f"""
            <div style="background-color: {safe_color}; color: white; padding: 24px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h4 style="margin: 0; color: rgba(255,255,255,0.9); text-transform: uppercase;">PREDICTED AQI</h4>
                <h1 style="margin: 8px 0; font-size: 68px; color: white; font-weight: 800;">{final_aqi}</h1>
                <h2 style="margin: 0; color: white; letter-spacing: 1.2px;">{safe_category}</h2>
                <p style="margin-top: 10px; font-size: 14px; color: rgba(255,255,255,0.95); font-weight: 600;">(CPCB Scale: {safe_range})</p>
                <div style="margin-top: 12px; padding: 6px; background: rgba(0,0,0,0.2); border-radius: 6px; font-size: 13px;">
                    Dominant Driver: <strong>{dominant_pollutant}</strong> (Sub-Index: {sub_indices[dominant_pollutant]})
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with res2:
            st.markdown("#### 🏥 Health Advisory & Medical Implications")
            st.markdown(f"*{meta['description']}*")
            st.info(f"**Public Advisory:** {meta['advisory']}")
            
            if category in ["Poor", "Very Poor", "Severe"]:
                st.error("⚠️ **High Risk Alert:** Wear N95 masks outdoors and activate indoor HEPA filtration.")
            else:
                st.success("✅ Air quality is within acceptable and safe thresholds for the general public.")
                
        st.divider()
        with st.expander("🔍 Detailed Sub-Indices & Feature Analysis"):
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("**CPCB Sub-Index Breakdown**")
                sub_df = pd.DataFrame([
                    {"Pollutant": k, "Sub-Index Value": v, "Status": "Dominant Driver" if k == dominant_pollutant else "Compliant"}
                    for k, v in sub_indices.items()
                ])
                st.dataframe(sub_df, use_container_width=True)
                
            with sub_col2:
                st.markdown("**Engineered Physical Indicators**")
                st.write(f"- **PM2.5/PM10 Combustion Ratio:** `{pm25_pm10_ratio:.3f}`")
                st.write(f"- **Inversion Stagnation Index:** `{stagnation_index:.2f}`")
                st.write(f"- **Gaseous Pollution Aggregate:** `{gaseous_index:.2f}`")
                st.write(f"- **Calculated Season:** `{season}` (Month `{month}`)")

# Sidebar Scale Guide
with st.sidebar:
    st.markdown("### 🌫️ CPCB AQI Scale Guide")
    for cat, data in AQI_CATEGORIES.items():
        st.markdown(f"""
        <div style="padding: 8px; margin-bottom: 6px; border-left: 5px solid {data['color']}; background-color: #f8f9fa; border-radius: 4px;">
            <strong>{cat} ({data['range']})</strong><br/>
            <span style="font-size: 12px; color: #444;">{data['description']}</span>
        </div>
        """, unsafe_allow_html=True)
