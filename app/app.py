import os
import sys
import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Ensure src directory is in sys.path for secure loading
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from security_utils import load_secure_model, SecurityError

# Streamlit Page Configuration - Executive Dark/Light Responsive
st.set_page_config(
    page_title="AQI Intelligence System | AI & CPCB Analytics",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS CSS
st.markdown("""
<style>
    /* Metric and Card Styling */
    .hero-card {
        padding: 24px 28px;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .hero-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
    }
    .metric-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.2);
        letter-spacing: 0.5px;
    }
    .sub-indicator-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: rgba(128, 128, 128, 0.05);
        margin-bottom: 12px;
    }
    /* Scale Guide in Sidebar */
    .scale-card {
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 10px;
        border-left: 5px solid;
        background: rgba(128, 128, 128, 0.08);
        border-top: 1px solid rgba(128, 128, 128, 0.1);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }
    /* Interactive Gradient Gauge */
    .gauge-bar {
        width: 100%;
        height: 16px;
        border-radius: 8px;
        background: linear-gradient(90deg, 
            #00b894 0%, #00b894 10%, 
            #55efc4 10.1%, #55efc4 20%, 
            #f1c40f 20.1%, #f1c40f 40%, 
            #e67e22 40.1%, #e67e22 60%, 
            #e74c3c 60.1%, #e74c3c 80%, 
            #8b0000 80.1%, #8b0000 100%);
        position: relative;
        margin: 12px 0 24px 0;
    }
    .gauge-indicator {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 28px;
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 3px;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.6);
        transform: translateX(-50%);
    }
    /* Quick Scenario Chip Styling */
    .scenario-title {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #888888;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

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

CPCB_STANDARDS = {
    'PM2.5': 60.0,
    'PM10': 100.0,
    'NO2': 80.0,
    'SO2': 80.0,
    'CO': 2.0,
    'O3': 100.0
}

def calculate_sub_index(val, pollutant):
    if val is None or val < 0:
        return 0.0
    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= val <= c_high:
            ans = i_low + (i_high - i_low) / (c_high - c_low) * (val - c_low)
            return round(ans, 1)
    if val > BREAKPOINTS[pollutant][-1][1]:
        last_low, last_high, i_low, i_high = BREAKPOINTS[pollutant][-1]
        slope = (i_high - i_low) / (last_high - last_low)
        return min(500.0, round(i_high + slope * (val - last_high), 1))
    return 500.0

AQI_CATEGORIES = {
    "Good": {
        "range": "0 - 50",
        "color": "#00b894",
        "gradient": "linear-gradient(135deg, #00b894 0%, #00cec9 100%)",
        "icon": "🌿",
        "description": "Clean and pristine air quality. Minimal to zero health hazard for the entire population.",
        "advisory": "Air quality is ideal for all outdoor sports, running, and ventilation.",
        "mask": "Not Required",
        "purifier": "Optional"
    },
    "Satisfactory": {
        "range": "51 - 100",
        "color": "#2ed573",
        "gradient": "linear-gradient(135deg, #2ed573 0%, #10ac84 100%)",
        "icon": "🌤️",
        "description": "Acceptable air quality. Minor breathing discomfort possible only for exceptionally sensitive persons.",
        "advisory": "Sensitive individuals should consider taking normal rest intervals during prolonged outdoor exertion.",
        "mask": "Not Required",
        "purifier": "Optional"
    },
    "Moderate": {
        "range": "101 - 200",
        "color": "#f1c40f",
        "gradient": "linear-gradient(135deg, #f39c12 0%, #e67e22 100%)",
        "icon": "😷",
        "description": "May cause breathing discomfort to individuals with asthma, heart ailments, and elderly groups.",
        "advisory": "Children and elderly individuals should avoid continuous heavy outdoor physical training.",
        "mask": "Recommended for Sensitive Groups",
        "purifier": "Recommended indoors"
    },
    "Poor": {
        "range": "201 - 300",
        "color": "#e67e22",
        "gradient": "linear-gradient(135deg, #e67e22 0%, #d35400 100%)",
        "icon": "⚠️",
        "description": "May cause breathing discomfort to most people on prolonged exposure. Triggers acute respiratory symptoms.",
        "advisory": "General public should significantly curtail outdoor activities. Sensitive groups should remain indoors.",
        "mask": "N95 / FFP2 Strongly Advised",
        "purifier": "Active HEPA Filtration Required"
    },
    "Very Poor": {
        "range": "301 - 400",
        "color": "#e74c3c",
        "gradient": "linear-gradient(135deg, #eb4d4b 0%, #c0392b 100%)",
        "icon": "🚨",
        "description": "Health Alert: Causes respiratory illness on prolonged exposure. Significant risk to cardiac health.",
        "advisory": "Avoid all outdoor physical activity. Keep windows shut. Sensitive groups must remain strictly indoors.",
        "mask": "N95 Mask Mandatory Outdoors",
        "purifier": "High-Efficiency Air Filtration Essential"
    },
    "Severe": {
        "range": "401 - 500+",
        "color": "#8b0000",
        "gradient": "linear-gradient(135deg, #741b47 0%, #4a0018 100%)",
        "icon": "🛑",
        "description": "Emergency Hazard: Severe health impacts on healthy individuals; life-threatening for pulmonary/cardiac patients.",
        "advisory": "Public health emergency. Stay indoors. Run commercial HEPA air purifiers continuously.",
        "mask": "N95/N99 Mask Essential Outdoors",
        "purifier": "Continuous Emergency Air Purification"
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

@st.cache_resource(show_spinner=False)
def load_app_model():
    model_path = os.path.join(PROJECT_DIR, "artifacts", "model.joblib")
    if not os.path.exists(model_path):
        try:
            try:
                from src.save_model import save_final_model
            except ImportError:
                from save_model import save_final_model  # type: ignore
            save_final_model()
        except Exception as err:
            return None, f"Model auto-build failed: {err}"
            
    if not os.path.exists(model_path):
        return None, "Model file not found. Run 'python src/save_model.py'."
    try:
        loaded = load_secure_model(model_path, enforce_hash_check=True)
        return loaded, "Cryptographically verified model (SHA-256 authenticated)."
    except SecurityError as se:
        return None, f"Security Warning: {se}"
    except Exception as e:
        return None, f"Model Loading Error: {e}"

def get_global_feature_importances(loaded_model):
    if loaded_model is None:
        return None

    model_steps = getattr(loaded_model, "named_steps", {})
    preprocessor = model_steps.get("preprocessor")
    regressor = model_steps.get("regressor")
    if (
        preprocessor is None
        or regressor is None
        or not hasattr(regressor, "feature_importances_")
    ):
        return None

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_
    if len(feature_names) != len(importances):
        raise ValueError("Model feature names do not match its feature importances.")

    importance_df = pd.DataFrame({
        "Feature": [
            name.split("__", 1)[-1].replace("_median", "").replace("_", " ")
            for name in feature_names
        ],
        "Importance": importances
    })
    return importance_df.nlargest(8, "Importance").sort_values("Importance")

# Load model instantly from cache
model, load_status = load_app_model()
global_feature_importances = get_global_feature_importances(model)

# Sidebar: CPCB AQI Scale Guide (Theme adaptive)
with st.sidebar:
    st.markdown("### 🍃 CPCB AQI Standard Scale")
    st.caption("Official Central Pollution Control Board (CPCB) India Reference Matrix")
    
    for cat, data in AQI_CATEGORIES.items():
        st.markdown(f"""
        <div class="scale-card" style="border-left-color: {data['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 14px;">{data['icon']} {cat}</span>
                <span style="font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: {data['color']}22; color: {data['color']};">{data['range']}</span>
            </div>
            <div style="font-size: 11.5px; opacity: 0.85; line-height: 1.4;">
                {data['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.markdown("#### 🛡️ System Integrity")
    st.caption(f"Status: `{load_status}`")
    st.caption("Architecture: `Tuned Random Forest Regressor`")
    st.caption("Training data: generated synthetic examples, not live sensor measurements.")

# Header Section
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🌬️ Air Quality Prediction & Environmental Intelligence")
    st.markdown(
        "Explore how pollutant and weather conditions affect air quality. Adjust the inputs or choose a scenario "
        "to see the results update immediately."
    )
    st.caption(
        "This is an interactive estimate based on your inputs, not a live sensor reading or a substitute "
        "for official local air-quality guidance. It estimates same-day AQI; it does not forecast future conditions."
    )
with header_col2:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if model is not None:
        st.success("🟢 **Model Online**\n\nSHA-256 Verified")
    else:
        st.error("🔴 **Model Offline**")

# Interactive Scenario Presets
SCENARIOS = {
    "🌿 Clean Mountain / Forest": {
        "pm25": 18.0, "pm10": 32.0, "no2": 12.0, "so2": 6.0, "co": 0.4, "o3": 22.0,
        "temp": 22.0, "humidity": 48.0, "wind_speed": 4.8
    },
    "🏙️ Typical Urban Day": {
        "pm25": 48.0, "pm10": 88.0, "no2": 32.0, "so2": 14.0, "co": 1.2, "o3": 38.0,
        "temp": 27.0, "humidity": 55.0, "wind_speed": 3.4
    },
    "🏭 Industrial Traffic Hub": {
        "pm25": 110.0, "pm10": 210.0, "no2": 72.0, "so2": 45.0, "co": 3.2, "o3": 68.0,
        "temp": 31.0, "humidity": 42.0, "wind_speed": 1.6
    },
    "🌫️ Severe Winter Smog": {
        "pm25": 290.0, "pm10": 440.0, "no2": 140.0, "so2": 78.0, "co": 8.8, "o3": 95.0,
        "temp": 13.5, "humidity": 84.0, "wind_speed": 0.7
    }
}

# State initialization
if "scenario_applied" not in st.session_state:
    st.session_state.scenario_applied = "🏙️ Typical Urban Day"
    for k, v in SCENARIOS["🏙️ Typical Urban Day"].items():
        st.session_state[f"input_{k}"] = v
if "input_predict_date" not in st.session_state:
    st.session_state.input_predict_date = datetime.date.today()

def mark_custom_inputs():
    st.session_state.scenario_applied = "✏️ Custom inputs"

# Navigation Tabs
tab_predict, tab_analytics, tab_governance = st.tabs([
    "🎛️ Live Prediction & Health Advisory",
    "📊 Pollutant Concentrations vs CPCB Limits",
    "🧠 Model Governance & Benchmarks"
])

# ----------------- TAB 1: PREDICTION & ADVISORY -----------------
with tab_predict:
    st.markdown('<div class="scenario-title">⚡ Instant Scenario Presets (1-Click Demo)</div>', unsafe_allow_html=True)
    preset_cols = st.columns(4)
    for idx, (s_name, s_data) in enumerate(SCENARIOS.items()):
        with preset_cols[idx]:
            if st.button(s_name, use_container_width=True, key=f"btn_scen_{idx}"):
                st.session_state.scenario_applied = s_name
                for k, v in s_data.items():
                    st.session_state[f"input_{k}"] = v
                st.rerun()

    st.caption(
        f"Current preset: **{st.session_state.scenario_applied}**. "
        "Changing any control switches to a custom scenario."
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    with st.expander("🛠️ Customize Environmental & Meteorological Inputs", expanded=True):
        input_c1, input_c2 = st.columns(2)
        
        with input_c1:
            st.markdown("##### 🧪 Pollutant Concentrations")
            pm25 = st.slider(
                "PM2.5 (Fine Particulates, \u03bcg/m\u00b3)", 
                0.0, 500.0, step=1.0,
                help="CPCB 24-hour reference: 60 \u03bcg/m\u00b3",
                key="input_pm25",
                on_change=mark_custom_inputs
            )
            pm10 = st.slider(
                "PM10 (Coarse Particulates, \u03bcg/m\u00b3)", 
                0.0, 600.0, step=1.0,
                help="CPCB 24-hour reference: 100 \u03bcg/m\u00b3",
                key="input_pm10",
                on_change=mark_custom_inputs
            )
            no2 = st.slider(
                "NO2 (Nitrogen Dioxide, \u03bcg/m\u00b3)", 
                0.0, 400.0, step=1.0,
                help="CPCB 24-hour reference: 80 \u03bcg/m\u00b3",
                key="input_no2",
                on_change=mark_custom_inputs
            )
            so2 = st.slider(
                "SO2 (Sulfur Dioxide, \u03bcg/m\u00b3)", 
                0.0, 400.0, step=1.0,
                help="CPCB 24-hour reference: 80 \u03bcg/m\u00b3",
                key="input_so2",
                on_change=mark_custom_inputs
            )
            co = st.slider(
                "CO (Carbon Monoxide, mg/m\u00b3)", 
                0.0, 30.0, step=0.1,
                help="CPCB 8-hour reference: 2.0 mg/m\u00b3",
                key="input_co",
                on_change=mark_custom_inputs
            )
            o3 = st.slider(
                "O3 (Ozone, \u03bcg/m\u00b3)", 
                0.0, 300.0, step=1.0,
                help="CPCB 8-hour reference: 100 \u03bcg/m\u00b3",
                key="input_o3",
                on_change=mark_custom_inputs
            )
            
        with input_c2:
            st.markdown("##### 🌤️ Meteorology & Observation Context")
            temp = st.slider(
                "Ambient Temperature (\u00b0C)", 
                -10.0, 50.0, step=0.5,
                key="input_temp",
                on_change=mark_custom_inputs
            )
            humidity = st.slider(
                "Relative Humidity (%)", 
                5.0, 100.0, step=1.0,
                key="input_humidity",
                on_change=mark_custom_inputs
            )
            wind_speed = st.slider(
                "Wind Speed (m/s)", 
                0.1, 25.0, step=0.1,
                help="Lower wind speed (<1.5 m/s) exacerbates air stagnation.",
                key="input_wind_speed",
                on_change=mark_custom_inputs
            )
            predict_date = st.date_input(
                "Observation Date",
                key="input_predict_date",
                on_change=mark_custom_inputs
            )

    # Consistency Warning
    if pm10 < pm25:
        st.warning("⚠️ **Physical Consistency Notice:** PM10 is measured lower than PM2.5. In atmospheric physics, PM2.5 is a physical sub-fraction of PM10.")

    # Execute Instant Real-Time Calculation
    month = predict_date.month
    day_of_week = predict_date.weekday()
    season = get_season(month)

    pm25_pm10_ratio = pm25 / pm10 if pm10 > 0 else 0.5
    pm25_wind_ratio = pm25 / wind_speed if wind_speed > 0 else 0.0
    stagnation_index = pm25 / (wind_speed + 0.1)
    gaseous_index = no2 + so2 + (co * 10.0)
    thermal_humidity_index = temp * (humidity / 100.0)

    month_sin = float(np.sin(2 * np.pi * month / 12.0))
    month_cos = float(np.cos(2 * np.pi * month / 12.0))
    day_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
    day_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))

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
    ml_pred_aqi = None
    if model is not None:
        try:
            ml_pred_aqi = round(float(model.predict(input_df)[0]), 1)
        except (KeyError, TypeError, ValueError) as err:
            st.error(f"The machine-learning estimate could not be calculated: {err}")

    # 2. CPCB Sub-Index Calculation
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

    aqi_values = {}
    if ml_pred_aqi is not None:
        aqi_values["Machine-learning estimate"] = ml_pred_aqi
    aqi_values["CPCB calculation"] = exact_cpcb_aqi
    result_source = st.radio(
        "Choose the AQI shown in the main result",
        list(aqi_values),
        horizontal=True,
        help=(
            "The CPCB calculation is the highest pollutant sub-index from the entered concentrations. "
            "The machine-learning estimate is the trained model's prediction for the same inputs."
        )
    )
    final_aqi = aqi_values[result_source]
    final_aqi = round(float(np.clip(final_aqi, 0.0, 500.0)), 1)
    category = map_aqi_to_category(final_aqi)
    meta = AQI_CATEGORIES[category]

    # Calculate needle percentage on 0-500 scale
    gauge_pct = min(100.0, max(0.0, (final_aqi / 500.0) * 100.0))

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.subheader("🎯 AQI Result & Health Assessment")
    st.caption(f"Main result uses: **{result_source}**. Change the selection above to compare methods.")

    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        st.markdown(f"""
        <div class="hero-card" style="background: {meta['gradient']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-pill">{result_source.upper()}</span>
                <span style="font-size: 28px;">{meta['icon']}</span>
            </div>
            <div style="font-size: 72px; font-weight: 800; line-height: 1.1; margin: 8px 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">
                {final_aqi}
            </div>
            <div style="font-size: 24px; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px;">
                {category.upper()}
            </div>
            <div style="font-size: 13px; opacity: 0.95; font-weight: 600;">
                CPCB Standard Index Range: {meta['range']}
            </div>
            <div class="gauge-bar">
                <div class="gauge-indicator" style="left: {gauge_pct}%;"></div>
            </div>
            <div style="padding: 10px 14px; background: rgba(0, 0, 0, 0.25); border-radius: 8px; font-size: 13px;">
                Primary Driver: <strong>{dominant_pollutant}</strong> (Sub-Index: <strong>{sub_indices[dominant_pollutant]}</strong>)
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Official CPCB calculation", f"{exact_cpcb_aqi:.1f}")
        if ml_pred_aqi is not None:
            st.metric("Machine-learning estimate", f"{ml_pred_aqi:.1f}")
        else:
            st.info("The model estimate is unavailable. The CPCB calculation remains available.")

    with res_col2:
        st.markdown(f"#### 🏥 Health Advisory: **{category} Air Quality**")
        st.caption(f"Category and advisory are based on the selected {result_source.lower()}.")
        st.markdown(f"*{meta['description']}*")
        st.caption(
            f"The highest CPCB sub-index for these inputs is {dominant_pollutant} "
            f"({sub_indices[dominant_pollutant]:.1f}). This identifies the CPCB calculation driver, "
            "not a causal explanation of the model prediction."
        )
        
        # Action cards
        act1, act2 = st.columns(2)
        with act1:
            st.markdown(f"""
            <div class="sub-indicator-card">
                <div style="font-weight: 700; font-size: 13px; color: #888;">😷 MASK RECOMMENDATION</div>
                <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">{meta['mask']}</div>
            </div>
            """, unsafe_allow_html=True)
        with act2:
            st.markdown(f"""
            <div class="sub-indicator-card">
                <div style="font-weight: 700; font-size: 13px; color: #888;">💨 HEPA AIR PURIFICATION</div>
                <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">{meta['purifier']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.info(f"📋 **Public Health Directive:** {meta['advisory']}")

        # Physical indicators
        ind1, ind2, ind3 = st.columns(3)
        with ind1:
            st.metric("PM2.5/PM10 Ratio", f"{pm25_pm10_ratio:.2f}", help=">0.5 indicates combustion/vehicular source dominance")
        with ind2:
            st.metric("Stagnation Index", f"{stagnation_index:.1f}", help="Higher values indicate low ventilation and entrapment")
        with ind3:
            st.metric("Gaseous Burden", f"{gaseous_index:.1f}", help="Weighted aggregate of NO2, SO2, and CO")

    with st.expander("How to interpret this result"):
        st.markdown(
            "- **Machine-learning estimate:** the trained model estimates same-day AQI from the entered readings.\n"
            "- **CPCB calculation:** the highest calculated pollutant sub-index for the entered concentrations.\n"
            "- **Dominant pollutant:** the pollutant with that highest CPCB sub-index; it is not a causal model explanation.\n"
            "- **Data limitation:** the current model uses generated synthetic examples, not live sensor observations."
        )

    report = {
        "observation_date": predict_date.isoformat(),
        "result_source": result_source,
        "aqi": final_aqi,
        "category": category,
        "cpcb_calculated_aqi": round(float(exact_cpcb_aqi), 1),
        "machine_learning_estimate": ml_pred_aqi,
        "dominant_pollutant": dominant_pollutant,
        "inputs": {
            "PM2.5 (µg/m³)": pm25,
            "PM10 (µg/m³)": pm10,
            "NO2 (µg/m³)": no2,
            "SO2 (µg/m³)": so2,
            "CO (mg/m³)": co,
            "O3 (µg/m³)": o3,
            "temperature (°C)": temp,
            "humidity (%)": humidity,
            "wind_speed (m/s)": wind_speed
        }
    }
    st.download_button(
        "⬇️ Download this result as JSON",
        data=json.dumps(report, indent=2),
        file_name=f"aqi-result-{predict_date.isoformat()}.json",
        mime="application/json"
    )

# ----------------- TAB 2: POLLUTANT ANALYTICS -----------------
with tab_analytics:
    st.subheader("📊 Pollutant Concentrations vs CPCB Reference Thresholds")
    st.caption("PM2.5, PM10, NO2, and SO2 use 24-hour references; CO and O3 use 8-hour references.")
    
    chart_data = pd.DataFrame([
        {
            "Pollutant": k,
            "Concentration": float(val),
            "CPCB_Limit": CPCB_STANDARDS[k],
            "Sub_Index": sub_indices[k],
            "Status": "Exceeded" if float(val) > CPCB_STANDARDS[k] else "Safe"
        }
        for k, val in [("PM2.5", pm25), ("PM10", pm10), ("NO2", no2), ("SO2", so2), ("CO", co), ("O3", o3)]
    ])

    c_col1, c_col2 = st.columns([1.6, 1.4])
    
    with c_col1:
        # Altair bar chart with threshold
        base = alt.Chart(chart_data).encode(
            x=alt.X("Pollutant:N", title="Criteria Pollutants", sort=None),
            tooltip=["Pollutant", "Concentration", "CPCB_Limit", "Status"]
        )
        
        bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            y=alt.Y("Concentration:Q", title="Concentration (\u03bcg/m\u00b3 or mg/m\u00b3)"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Safe", "Exceeded"], range=["#2ecc71", "#e74c3c"]),
                legend=alt.Legend(title="Regulatory Status")
            )
        )
        
        st.altair_chart(bars, use_container_width=True)

    with c_col2:
        st.markdown("##### 📋 CPCB Sub-Index Breakdown Matrix")
        display_sub_df = chart_data[["Pollutant", "Concentration", "CPCB_Limit", "Sub_Index", "Status"]].copy()
        display_sub_df.rename(columns={
            "CPCB_Limit": "Safe Threshold",
            "Sub_Index": "CPCB Sub-Index",
            "Status": "Safety Status"
        }, inplace=True)
        st.dataframe(display_sub_df, use_container_width=True, hide_index=True)
        
        st.caption("ℹ️ *Under official CPCB Indian Air Quality standards, the overall AQI is governed by the maximum sub-index across all monitored criteria pollutants.*")

# ----------------- TAB 3: MODEL GOVERNANCE -----------------
with tab_governance:
    st.subheader("🧠 Machine Learning Governance & Model Performance")
    
    gov_col1, gov_col2 = st.columns(2)
    with gov_col1:
        st.markdown("##### 🏆 Champion Model Architecture")
        st.markdown("""
        * **Algorithm:** `Random Forest Regressor`
        * **Ensemble Size:** `150 trees`
        * **Input Fields:** `21 same-day pollutant, weather, and calendar values`
        * **Verification Hash:** `SHA-256 Cryptographic Checksum Active`
        * **Training Data:** Generated synthetic examples; no live sensor feed
        """)
        
    with gov_col2:
        st.markdown("##### 📈 Chronological Holdout Metrics")
        temporal_results_path = os.path.join(
            PROJECT_DIR, "reports", "temporal_evaluation.csv"
        )
        if os.path.exists(temporal_results_path):
            temporal_results = pd.read_csv(temporal_results_path)
            champion_result = temporal_results[
                temporal_results["Model"] == "Random Forest (chronological holdout)"
            ]
            if not champion_result.empty:
                metrics = champion_result.iloc[0]
                metric_cols = st.columns(3)
                metric_cols[0].metric("Test MAE", f"{metrics['MAE']:.2f} AQI")
                metric_cols[1].metric("Test R²", f"{metrics['R2']:.4f}")
                metric_cols[2].metric(
                    "Category accuracy", f"{metrics['Category Accuracy'] * 100:.2f}%"
                )
                st.caption(
                    f"Trained through {metrics['Training Through']}; tested from "
                    f"{metrics['Test From']} on later dates."
                )
                st.caption(
                    "These metrics are from a separate model fit on the training-date partition; "
                    "the interactive app's model is fitted on all available training rows."
                )
            temporal_results_display = temporal_results.copy()
            temporal_results_display["Category Accuracy"] = (
                temporal_results_display["Category Accuracy"].map("{:.2%}".format)
            )
            st.dataframe(temporal_results_display, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Run `python src/evaluate_temporal.py` to generate date-based holdout metrics."
            )

    st.markdown("---")
    st.markdown("##### 🔎 Global Model Feature Importance")
    if global_feature_importances is not None:
        importance_chart = alt.Chart(global_feature_importances).mark_bar().encode(
            x=alt.X("Importance:Q", title="Relative feature importance"),
            y=alt.Y("Feature:N", sort=None, title=None),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")]
        ).properties(height=260)
        st.altair_chart(importance_chart, use_container_width=True)
        st.caption(
            "These Random Forest importances summarize feature use across training data; "
            "they are not causal explanations or per-scenario contributions."
        )
    else:
        st.info("Global feature importance is not available for the loaded model.")
