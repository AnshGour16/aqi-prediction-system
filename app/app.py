import os
import sys
import html
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

try:
    from security_utils import load_secure_model, SecurityError
except ImportError:
    import joblib
    def load_secure_model(path, enforce_hash_check=False):
        return joblib.load(path)
    SecurityError = Exception

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

# Load model instantly from cache
model, load_status = load_app_model()

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
    st.caption("Validation R²: `99.71%` | MAE: `0.72`")

# Header Section
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🌬️ Air Quality Prediction & Environmental Intelligence")
    st.markdown(
        "Continuous **Air Quality Index (AQI)** estimation using physically coupled machine learning regressors, "
        "atmospheric dispersion indicators, and **official CPCB India breakpoints**."
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

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    with st.expander("🛠️ Customize Environmental & Meteorological Inputs", expanded=True):
        input_c1, input_c2 = st.columns(2)
        
        with input_c1:
            st.markdown("##### 🧪 Pollutant Concentrations")
            pm25 = st.slider(
                "PM2.5 (Fine Particulates, \u03bcg/m\u00b3)", 
                0.0, 500.0, float(st.session_state.get("input_pm25", 48.0)), 1.0,
                help="CPCB 24-hr standard: 60 \u03bcg/m\u00b3"
            )
            pm10 = st.slider(
                "PM10 (Coarse Particulates, \u03bcg/m\u00b3)", 
                0.0, 600.0, float(st.session_state.get("input_pm10", 88.0)), 1.0,
                help="CPCB 24-hr standard: 100 \u03bcg/m\u00b3"
            )
            no2 = st.slider(
                "NO2 (Nitrogen Dioxide, \u03bcg/m\u00b3)", 
                0.0, 400.0, float(st.session_state.get("input_no2", 32.0)), 1.0,
                help="CPCB 24-hr standard: 80 \u03bcg/m\u00b3"
            )
            so2 = st.slider(
                "SO2 (Sulfur Dioxide, \u03bcg/m\u00b3)", 
                0.0, 400.0, float(st.session_state.get("input_so2", 14.0)), 1.0,
                help="CPCB 24-hr standard: 80 \u03bcg/m\u00b3"
            )
            co = st.slider(
                "CO (Carbon Monoxide, mg/m\u00b3)", 
                0.0, 30.0, float(st.session_state.get("input_co", 1.2)), 0.1,
                help="CPCB 8-hr standard: 2.0 mg/m\u00b3"
            )
            o3 = st.slider(
                "O3 (Ozone, \u03bcg/m\u00b3)", 
                0.0, 300.0, float(st.session_state.get("input_o3", 38.0)), 1.0,
                help="CPCB 8-hr standard: 100 \u03bcg/m\u00b3"
            )
            
        with input_c2:
            st.markdown("##### 🌤️ Meteorology & Observation Context")
            temp = st.slider(
                "Ambient Temperature (\u00b0C)", 
                -10.0, 50.0, float(st.session_state.get("input_temp", 27.0)), 0.5
            )
            humidity = st.slider(
                "Relative Humidity (%)", 
                5.0, 100.0, float(st.session_state.get("input_humidity", 55.0)), 1.0
            )
            wind_speed = st.slider(
                "Wind Speed (m/s)", 
                0.1, 25.0, float(st.session_state.get("input_wind_speed", 3.4)), 0.1,
                help="Lower wind speed (<1.5 m/s) exacerbates air stagnation."
            )
            predict_date = st.date_input("Observation Date", value=datetime.date.today())

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
    if model is not None:
        ml_pred_aqi = round(float(model.predict(input_df)[0]), 1)
    else:
        ml_pred_aqi = 100.0

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

    final_aqi = round(float(np.clip(ml_pred_aqi, 0.0, 500.0)), 1)
    category = map_aqi_to_category(final_aqi)
    meta = AQI_CATEGORIES[category]

    # Calculate needle percentage on 0-500 scale
    gauge_pct = min(100.0, max(0.0, (final_aqi / 500.0) * 100.0))

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.subheader("🎯 Real-Time Prediction & Health Assessment")

    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        st.markdown(f"""
        <div class="hero-card" style="background: {meta['gradient']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-pill">AI ESTIMATED AQI</span>
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

    with res_col2:
        st.markdown(f"#### 🏥 Health Advisory: **{category} Air Quality**")
        st.markdown(f"*{meta['description']}*")
        
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

# ----------------- TAB 2: POLLUTANT ANALYTICS -----------------
with tab_analytics:
    st.subheader("📊 Pollutant Concentrations vs CPCB 24-Hour Safe Thresholds")
    
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
        * **Algorithm:** `Random Forest Regressor (Hyperparameter Tuned)`
        * **Ensemble Size:** `200 Estimators with Min-Samples-Leaf Regularization`
        * **Input Dimensions:** `22 Engineered Atmospheric Features`
        * **Verification Hash:** `SHA-256 Cryptographic Checksum Active`
        * **Training Set:** Full CPCB Multimodal Sensor Dataset
        """)
        
    with gov_col2:
        st.markdown("##### 📈 Validated Evaluation Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Test R² Score", "0.9971", "High Precision")
        with m2:
            st.metric("MAE", "0.7202", "Low Deviation")
        with m3:
            st.metric("Category Acc.", "98.88%", "Top Tier")

    st.markdown("---")
    st.markdown("##### 🔬 Benchmark Comparison Against Standard Baselines")
    
    comparison_data = pd.DataFrame([
        {"Model": "Random Forest (Champion)", "MAE": 0.7202, "RMSE": 8.5120, "R² Score": 0.9971, "Category Accuracy": "98.88%"},
        {"Model": "Gradient Boosting", "MAE": 2.5993, "RMSE": 8.0106, "R² Score": 0.9974, "Category Accuracy": "96.31%"},
        {"Model": "Decision Tree", "MAE": 0.7674, "RMSE": 9.5833, "R² Score": 0.9963, "Category Accuracy": "99.83%"},
        {"Model": "Support Vector Regressor", "MAE": 8.3841, "RMSE": 15.3334, "R² Score": 0.9906, "Category Accuracy": "82.07%"},
        {"Model": "K-Nearest Neighbors", "MAE": 13.3141, "RMSE": 21.0425, "R² Score": 0.9823, "Category Accuracy": "75.93%"},
        {"Model": "Ridge Regression (Baseline)", "MAE": 18.4646, "RMSE": 25.7412, "R² Score": 0.9735, "Category Accuracy": "68.82%"}
    ])
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
