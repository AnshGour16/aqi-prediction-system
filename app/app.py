import datetime
import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths and shared modules
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aqi_utils import (  # noqa: E402
    CPCB_STANDARDS,
    POLLUTANT_UNITS,
    calculate_sub_index,
    compute_aqi_from_concentrations,
    map_aqi_to_category,
    sub_index_breakdown,
)
from feature_builder import build_feature_frame  # noqa: E402
from security_utils import SecurityError, load_secure_model  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AQI Intelligence System | AI & CPCB Analytics",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 2.2rem; }

    /* Hero result card */
    .hero-card {
        padding: 24px 28px;
        border-radius: 18px;
        color: #ffffff;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .hero-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 82% 8%, rgba(255,255,255,0.22), transparent 45%);
        pointer-events: none;
    }
    .hero-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.30);
    }
    .metric-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.22);
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .sub-indicator-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        background: rgba(128, 128, 128, 0.06);
        margin-bottom: 12px;
        transition: border-color 0.18s ease, background 0.18s ease;
    }
    .sub-indicator-card:hover {
        border-color: rgba(46, 204, 113, 0.55);
        background: rgba(46, 204, 113, 0.07);
    }

    /* Sidebar CPCB scale cards */
    .scale-card {
        padding: 10px 14px;
        margin-bottom: 9px;
        border-radius: 10px;
        border-left: 5px solid;
        background: rgba(128, 128, 128, 0.08);
        border-top: 1px solid rgba(128, 128, 128, 0.10);
        border-right: 1px solid rgba(128, 128, 128, 0.10);
        border-bottom: 1px solid rgba(128, 128, 128, 0.10);
    }

    /* Gauge */
    .gauge-bar {
        width: 100%;
        height: 18px;
        border-radius: 9px;
        background: linear-gradient(90deg,
            #00b894 0%, #00b894 10%,
            #55efc4 10.1%, #55efc4 20%,
            #f1c40f 20.1%, #f1c40f 40%,
            #e67e22 40.1%, #e67e22 60%,
            #e74c3c 60.1%, #e74c3c 80%,
            #8b0000 80.1%, #8b0000 100%);
        position: relative;
        margin: 16px 0 6px 0;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.35);
    }
    .gauge-indicator {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 30px;
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 3px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.65);
        transform: translateX(-50%);
        transition: left 0.35s ease;
    }
    .gauge-ticks {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        opacity: 0.8;
        margin-bottom: 14px;
    }

    /* Presets */
    .scenario-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        color: #8a8a8a;
        margin-bottom: 8px;
    }

    /* Footer */
    .app-footer {
        font-size: 12px;
        opacity: 0.65;
        text-align: center;
        padding-top: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
AQI_CATEGORIES = {
    "Good": {
        "range": "0 - 50",
        "color": "#00b894",
        "gradient": "linear-gradient(135deg, #00b894 0%, #00cec9 100%)",
        "icon": "🌿",
        "description": "Clean and pristine air quality. Minimal to zero health hazard for the entire population.",
        "advisory": "Air quality is ideal for all outdoor sports, running, and ventilation.",
        "mask": "Not Required",
        "purifier": "Optional",
    },
    "Satisfactory": {
        "range": "51 - 100",
        "color": "#2ed573",
        "gradient": "linear-gradient(135deg, #2ed573 0%, #10ac84 100%)",
        "icon": "🌤️",
        "description": "Acceptable air quality. Minor breathing discomfort possible only for exceptionally sensitive persons.",
        "advisory": "Sensitive individuals should consider normal rest intervals during prolonged outdoor exertion.",
        "mask": "Not Required",
        "purifier": "Optional",
    },
    "Moderate": {
        "range": "101 - 200",
        "color": "#f1c40f",
        "gradient": "linear-gradient(135deg, #f39c12 0%, #e67e22 100%)",
        "icon": "😷",
        "description": "May cause breathing discomfort to people with asthma, heart ailments, and elderly groups.",
        "advisory": "Children and elderly individuals should avoid continuous heavy outdoor physical training.",
        "mask": "Recommended for Sensitive Groups",
        "purifier": "Recommended indoors",
    },
    "Poor": {
        "range": "201 - 300",
        "color": "#e67e22",
        "gradient": "linear-gradient(135deg, #e67e22 0%, #d35400 100%)",
        "icon": "⚠️",
        "description": "May cause breathing discomfort to most people on prolonged exposure and trigger acute respiratory symptoms.",
        "advisory": "The general public should significantly curtail outdoor activities. Sensitive groups should remain indoors.",
        "mask": "N95 / FFP2 Strongly Advised",
        "purifier": "Active HEPA Filtration Required",
    },
    "Very Poor": {
        "range": "301 - 400",
        "color": "#e74c3c",
        "gradient": "linear-gradient(135deg, #eb4d4b 0%, #c0392b 100%)",
        "icon": "🚨",
        "description": "Health alert: respiratory illness on prolonged exposure and significant risk to cardiac health.",
        "advisory": "Avoid all outdoor physical activity. Keep windows shut. Sensitive groups must remain strictly indoors.",
        "mask": "N95 Mask Mandatory Outdoors",
        "purifier": "High-Efficiency Air Filtration Essential",
    },
    "Severe": {
        "range": "401 - 500+",
        "color": "#8b0000",
        "gradient": "linear-gradient(135deg, #741b47 0%, #4a0018 100%)",
        "icon": "🛑",
        "description": "Emergency hazard: severe health impacts on healthy individuals; life-threatening for pulmonary and cardiac patients.",
        "advisory": "Public health emergency. Stay indoors. Run commercial HEPA air purifiers continuously.",
        "mask": "N95/N99 Mask Essential Outdoors",
        "purifier": "Continuous Emergency Air Purification",
    },
}

SCENARIOS = {
    "🌿 Clean Mountain / Forest": {
        "pm25": 18.0, "pm10": 32.0, "no2": 12.0, "so2": 6.0, "co": 0.4, "o3": 22.0,
        "temp": 22.0, "humidity": 48.0, "wind_speed": 4.8,
    },
    "🏙️ Typical Urban Day": {
        "pm25": 48.0, "pm10": 88.0, "no2": 32.0, "so2": 14.0, "co": 1.2, "o3": 38.0,
        "temp": 27.0, "humidity": 55.0, "wind_speed": 3.4,
    },
    "🏭 Industrial Traffic Hub": {
        "pm25": 110.0, "pm10": 210.0, "no2": 72.0, "so2": 45.0, "co": 3.2, "o3": 68.0,
        "temp": 31.0, "humidity": 42.0, "wind_speed": 1.6,
    },
    "🌫️ Severe Winter Smog": {
        "pm25": 290.0, "pm10": 440.0, "no2": 140.0, "so2": 78.0, "co": 8.8, "o3": 95.0,
        "temp": 13.5, "humidity": 84.0, "wind_speed": 0.7,
    },
}
DEFAULT_SCENARIO = "🏙️ Typical Urban Day"

INPUT_LABELS = {
    "pm25": "PM2.5 (Fine Particulates, µg/m³)",
    "pm10": "PM10 (Coarse Particulates, µg/m³)",
    "no2": "NO2 (Nitrogen Dioxide, µg/m³)",
    "so2": "SO2 (Sulfur Dioxide, µg/m³)",
    "co": "CO (Carbon Monoxide, mg/m³)",
    "o3": "O3 (Ozone, µg/m³)",
    "temp": "Ambient Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "wind_speed": "Wind Speed (m/s)",
}
INPUT_BOUNDS = {
    "pm25": (0.0, 500.0, 1.0),
    "pm10": (0.0, 600.0, 1.0),
    "no2": (0.0, 400.0, 1.0),
    "so2": (0.0, 400.0, 1.0),
    "co": (0.0, 30.0, 0.1),
    "o3": (0.0, 300.0, 1.0),
    "temp": (-10.0, 50.0, 0.5),
    "humidity": (5.0, 100.0, 1.0),
    "wind_speed": (0.1, 25.0, 0.1),
}
INPUT_HELP = {
    "pm25": "CPCB 24-hour reference: 60 µg/m³",
    "pm10": "CPCB 24-hour reference: 100 µg/m³",
    "no2": "CPCB 24-hour reference: 80 µg/m³",
    "so2": "CPCB 24-hour reference: 80 µg/m³",
    "co": "CPCB 8-hour reference: 2.0 mg/m³",
    "o3": "CPCB 8-hour reference: 100 µg/m³",
    "wind_speed": "Below 1.5 m/s, pollutants stagnate near the surface.",
}
POLLUTANT_INPUTS = ("pm25", "pm10", "no2", "so2", "co", "o3")
METEOROLOGY_INPUTS = ("temp", "humidity", "wind_speed")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
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
        except Exception as err:  # pragma: no cover - environment dependent
            return None, f"Model auto-build failed: {err}"

    if not os.path.exists(model_path):
        return None, "Model file not found. Run 'python src/save_model.py'."
    try:
        loaded = load_secure_model(model_path, enforce_hash_check=True)
        return loaded, "Cryptographically verified model (SHA-256 authenticated)."
    except SecurityError as se:
        return None, f"Security warning: {se}"
    except Exception as e:
        return None, f"Model loading error: {e}"


def get_global_feature_importances(loaded_model):
    """Return the top importances, or ``None`` when the model cannot supply them."""
    if loaded_model is None:
        return None
    try:
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
            return None

        importance_df = pd.DataFrame(
            {
                "Feature": [
                    name.split("__", 1)[-1].replace("_median", "").replace("_", " ")
                    for name in feature_names
                ],
                "Importance": importances,
            }
        )
        return importance_df.nlargest(8, "Importance").sort_values("Importance")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_temporal_results(path: str, mtime: float) -> pd.DataFrame | None:
    """Read the temporal evaluation CSV, keyed on its modification time."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


model, load_status = load_app_model()
global_feature_importances = get_global_feature_importances(model)


# ---------------------------------------------------------------------------
# Session state and callbacks
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "scenario_applied" not in st.session_state:
        st.session_state.scenario_applied = DEFAULT_SCENARIO
    for key, value in SCENARIOS[DEFAULT_SCENARIO].items():
        state_key = f"input_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = value
    if "input_predict_date" not in st.session_state:
        st.session_state.input_predict_date = datetime.date.today()


def apply_scenario(name: str) -> None:
    """Set every input to a preset, or back to the default scenario."""
    for key, value in SCENARIOS[name].items():
        st.session_state[f"input_{key}"] = value
    st.session_state.scenario_applied = name


def mark_custom_inputs() -> None:
    st.session_state.scenario_applied = "✏️ Custom inputs"


def fix_particulate_order() -> None:
    """Raise PM10 to at least PM2.5 so the inputs are physically possible."""
    pm25 = float(st.session_state.get("input_pm25", 0.0))
    pm10 = float(st.session_state.get("input_pm10", 0.0))
    st.session_state.input_pm10 = min(600.0, max(pm10, pm25))
    st.session_state.scenario_applied = "✏️ Custom inputs"


def reload_model() -> None:
    load_app_model.clear()


_init_state()


def current_values() -> dict[str, float]:
    return {key: float(st.session_state[f"input_{key}"]) for key in INPUT_BOUNDS}


def current_date() -> datetime.date:
    value = st.session_state.get("input_predict_date", datetime.date.today())
    if isinstance(value, (tuple, list)):  # st.date_input can return a tuple
        value = value[0]
    return value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🍃 CPCB AQI Standard Scale")
    st.caption("Official Central Pollution Control Board (India) reference matrix")
    for cat, data in AQI_CATEGORIES.items():
        st.markdown(
            f"""
        <div class="scale-card" style="border-left-color: {data['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 14px;">{data['icon']} {cat}</span>
                <span style="font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: {data['color']}22; color: {data['color']};">{data['range']}</span>
            </div>
            <div style="font-size: 11.5px; opacity: 0.85; line-height: 1.4;">
                {data['description']}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### ⚡ Quick Actions")
    st.button(
        "↺ Reset to typical urban day",
        width="stretch",
        on_click=apply_scenario,
        args=(DEFAULT_SCENARIO,),
    )
    st.button("🔄 Reload model from disk", width="stretch", on_click=reload_model)

    st.divider()
    st.markdown("#### 🛡️ System Integrity")
    st.caption(f"Status: `{load_status}`")
    st.caption("Architecture: `Tuned Random Forest Regressor`")
    st.caption("Training data: generated synthetic examples, not live sensor measurements.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🌬️ Air Quality Prediction & Environmental Intelligence")
    st.markdown(
        "Explore how pollutant and weather conditions affect air quality. Adjust the inputs or pick a "
        "scenario — every result updates instantly."
    )
    st.caption(
        "This is an input-driven estimate, not a live sensor reading or a substitute for official local "
        "air-quality guidance. It estimates same-day AQI; it does not forecast future conditions."
    )
with header_col2:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if model is not None:
        st.success("🟢 **Model Online**\n\nSHA-256 verified")
    else:
        st.error("🔴 **Model Offline**")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_analytics, tab_lab, tab_governance = st.tabs(
    [
        "🎛️ Live Prediction & Health Advisory",
        "📊 Pollutant Analytics",
        "🔬 Scenario Lab",
        "🧠 Model Governance",
    ]
)

# ===========================================================================
# TAB 1 — LIVE PREDICTION
# ===========================================================================
with tab_predict:
    st.markdown(
        '<div class="scenario-title">⚡ Instant scenario presets (1-click demo)</div>',
        unsafe_allow_html=True,
    )
    preset_cols = st.columns(4)
    for idx, name in enumerate(SCENARIOS):
        with preset_cols[idx]:
            st.button(
                name,
                key=f"btn_scen_{idx}",
                width="stretch",
                type="primary" if st.session_state.scenario_applied == name else "secondary",
                on_click=apply_scenario,
                args=(name,),
            )

    st.caption(
        f"Current preset: **{st.session_state.scenario_applied}**. "
        "Changing any control switches to a custom scenario."
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with st.expander("🛠️ Customize environmental & meteorological inputs", expanded=True):
        input_c1, input_c2 = st.columns(2)

        with input_c1:
            st.markdown("##### 🧪 Pollutant Concentrations")
            for key in POLLUTANT_INPUTS:
                low, high, step = INPUT_BOUNDS[key]
                st.slider(
                    INPUT_LABELS[key],
                    low,
                    high,
                    step=step,
                    help=INPUT_HELP.get(key),
                    key=f"input_{key}",
                    on_change=mark_custom_inputs,
                )

        with input_c2:
            st.markdown("##### 🌤️ Meteorology & Observation Context")
            for key in METEOROLOGY_INPUTS:
                low, high, step = INPUT_BOUNDS[key]
                st.slider(
                    INPUT_LABELS[key],
                    low,
                    high,
                    step=step,
                    help=INPUT_HELP.get(key),
                    key=f"input_{key}",
                    on_change=mark_custom_inputs,
                )
            st.date_input(
                "Observation Date",
                min_value=datetime.date(2020, 1, 1),
                max_value=datetime.date.today(),
                key="input_predict_date",
                on_change=mark_custom_inputs,
                help="Calendar features (month, weekday, season) are derived from this date.",
            )

    values = current_values()
    predict_date = current_date()

    if values["pm10"] < values["pm25"]:
        st.warning(
            "⚠️ **Physical consistency notice:** PM10 is lower than PM2.5. In atmospheric physics, "
            "PM2.5 is a sub-fraction of PM10, so PM10 must be the larger value."
        )
        st.button(
            "🛠️ Fix automatically (set PM10 = PM2.5)",
            on_click=fix_particulate_order,
        )

    # --- Core calculations -------------------------------------------------
    feature_frame = build_feature_frame(values, predict_date)

    ml_pred_aqi = None
    prediction_error = None
    if model is not None:
        try:
            raw_prediction = float(model.predict(feature_frame)[0])
            ml_pred_aqi = round(float(np.clip(raw_prediction, 0.0, 500.0)), 1)
        except Exception as err:  # pragma: no cover - defensive
            prediction_error = str(err)

    breakdown = sub_index_breakdown(values)
    if breakdown:
        exact_cpcb_aqi = round(float(max(breakdown.values())), 1)
        dominant_pollutant = max(breakdown, key=breakdown.get)
    else:
        exact_cpcb_aqi = float("nan")
        dominant_pollutant = None

    aqi_values: dict[str, float] = {}
    if ml_pred_aqi is not None:
        aqi_values["Machine-learning estimate"] = ml_pred_aqi
    if breakdown:
        aqi_values["CPCB calculation"] = exact_cpcb_aqi

    if not aqi_values:
        st.error(
            "No AQI value could be computed. The model is unavailable and the pollutant "
            "inputs are incomplete."
        )
        st.stop()

    result_source = st.radio(
        "Choose the AQI shown in the main result",
        list(aqi_values),
        horizontal=True,
        help=(
            "The CPCB calculation is the highest pollutant sub-index from the entered concentrations. "
            "The machine-learning estimate is the trained model's prediction for the same inputs."
        ),
    )
    final_aqi = round(float(np.clip(aqi_values[result_source], 0.0, 500.0)), 1)
    category = map_aqi_to_category(final_aqi)
    meta = AQI_CATEGORIES[category]
    gauge_pct = min(100.0, max(0.0, (final_aqi / 500.0) * 100.0))

    if prediction_error:
        st.error(f"The machine-learning estimate could not be calculated: {prediction_error}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.subheader("🎯 AQI Result & Health Assessment")
    st.caption(
        f"Main result uses **{result_source}**. Change the selection above to compare methods."
    )

    res_col1, res_col2 = st.columns([1.15, 1.85])

    with res_col1:
        st.markdown(
            f"""
        <div class="hero-card" style="background: {meta['gradient']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-pill">{result_source}</span>
                <span style="font-size: 28px;">{meta['icon']}</span>
            </div>
            <div style="font-size: 72px; font-weight: 800; line-height: 1.05; margin: 8px 0; text-shadow: 0 2px 10px rgba(0,0,0,0.22);">
                {final_aqi}
            </div>
            <div style="font-size: 24px; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px;">
                {category.upper()}
            </div>
            <div style="font-size: 13px; opacity: 0.95; font-weight: 600;">
                CPCB standard index range: {meta['range']}
            </div>
            <div class="gauge-bar">
                <div class="gauge-indicator" style="left: {gauge_pct}%;"></div>
            </div>
            <div class="gauge-ticks">
                <span>0</span><span>100</span><span>200</span><span>300</span><span>400</span><span>500</span>
            </div>
            <div style="padding: 10px 14px; background: rgba(0, 0, 0, 0.25); border-radius: 8px; font-size: 13px;">
                Primary driver: <strong>{dominant_pollutant or '—'}</strong>
                (sub-index: <strong>{breakdown.get(dominant_pollutant, float('nan')):.1f}</strong>)
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(2)
        metric_cols[0].metric("CPCB calculation", f"{exact_cpcb_aqi:.1f}")
        if ml_pred_aqi is not None:
            delta = ml_pred_aqi - exact_cpcb_aqi
            metric_cols[1].metric(
                "Model estimate",
                f"{ml_pred_aqi:.1f}",
                delta=f"{delta:+.1f} vs CPCB",
                delta_color="inverse",
                help="A positive delta means the model estimates a higher (worse) AQI than the CPCB maths.",
            )
        else:
            metric_cols[1].info("Model estimate unavailable.")

        if ml_pred_aqi is not None:
            gap = abs(ml_pred_aqi - exact_cpcb_aqi)
            if gap <= 10:
                st.markdown(
                    f'<span class="chip" style="background:#2ecc7122;color:#2ecc71;">✓ Methods agree (Δ {gap:.1f})</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="chip" style="background:#e67e2222;color:#e67e22;">⚠ Methods differ by {gap:.1f} points</span>',
                    unsafe_allow_html=True,
                )

    with res_col2:
        st.markdown(f"#### 🏥 Health Advisory: **{category} Air Quality**")
        st.caption(f"Category and advisory are based on the selected {result_source.lower()}.")
        st.markdown(f"*{meta['description']}*")
        if dominant_pollutant:
            st.caption(
                f"The highest CPCB sub-index for these inputs is {dominant_pollutant} "
                f"({breakdown[dominant_pollutant]:.1f}). This identifies the CPCB calculation driver, "
                "not a causal explanation of the model prediction."
            )

        act1, act2 = st.columns(2)
        with act1:
            st.markdown(
                f"""
            <div class="sub-indicator-card">
                <div style="font-weight: 700; font-size: 12px; color: #8a8a8a; letter-spacing: 0.5px;">😷 MASK RECOMMENDATION</div>
                <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">{meta['mask']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with act2:
            st.markdown(
                f"""
            <div class="sub-indicator-card">
                <div style="font-weight: 700; font-size: 12px; color: #8a8a8a; letter-spacing: 0.5px;">💨 HEPA AIR PURIFICATION</div>
                <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">{meta['purifier']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.info(f"📋 **Public health directive:** {meta['advisory']}")

        pm25_pm10_ratio = values["pm25"] / values["pm10"] if values["pm10"] > 0 else 0.5
        stagnation_index = values["pm25"] / (values["wind_speed"] + 0.1)
        gaseous_index = values["no2"] + values["so2"] + (values["co"] * 10.0)

        ind1, ind2, ind3 = st.columns(3)
        ind1.metric(
            "PM2.5/PM10 ratio",
            f"{pm25_pm10_ratio:.2f}",
            help="Above 0.5 suggests combustion or vehicular source dominance.",
        )
        ind2.metric(
            "Stagnation index",
            f"{stagnation_index:.1f}",
            help="Higher values indicate poor ventilation and pollutant entrapment.",
        )
        ind3.metric(
            "Gaseous burden",
            f"{gaseous_index:.1f}",
            help="Weighted aggregate of NO2, SO2 and CO.",
        )

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
        "sub_indices": {name: round(float(value), 2) for name, value in breakdown.items()},
        "inputs": {
            "PM2.5 (µg/m³)": values["pm25"],
            "PM10 (µg/m³)": values["pm10"],
            "NO2 (µg/m³)": values["no2"],
            "SO2 (µg/m³)": values["so2"],
            "CO (mg/m³)": values["co"],
            "O3 (µg/m³)": values["o3"],
            "temperature (°C)": values["temp"],
            "humidity (%)": values["humidity"],
            "wind_speed (m/s)": values["wind_speed"],
        },
    }

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download result as JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name=f"aqi-result-{predict_date.isoformat()}.json",
            mime="application/json",
            width="stretch",
        )
    with dl2:
        report_table = pd.DataFrame(
            [
                {
                    "Pollutant": name,
                    "Concentration": values[key],
                    "Unit": POLLUTANT_UNITS[name],
                    "CPCB reference": CPCB_STANDARDS[name],
                    "Sub-index": round(float(breakdown.get(name, float("nan"))), 2),
                }
                for name, key in zip(
                    ("PM2.5", "PM10", "NO2", "SO2", "CO", "O3"), POLLUTANT_INPUTS
                )
            ]
        )
        st.download_button(
            "⬇️ Download pollutant table as CSV",
            data=report_table.to_csv(index=False),
            file_name=f"aqi-pollutants-{predict_date.isoformat()}.csv",
            mime="text/csv",
            width="stretch",
        )

# ===========================================================================
# TAB 2 — POLLUTANT ANALYTICS
# ===========================================================================
with tab_analytics:
    st.subheader("📊 Pollutant Concentrations vs CPCB Reference Thresholds")
    st.caption(
        "PM2.5, PM10, NO2 and SO2 use 24-hour references; CO and O3 use 8-hour references. "
        "Because CO is measured in mg/m³ and the others in µg/m³, the chart normalises every "
        "pollutant to its own reference value instead of mixing units on one axis."
    )

    chart_rows = []
    for name, key in zip(("PM2.5", "PM10", "NO2", "SO2", "CO", "O3"), POLLUTANT_INPUTS):
        concentration = float(values[key])
        limit = CPCB_STANDARDS[name]
        chart_rows.append(
            {
                "Pollutant": name,
                "Concentration": concentration,
                "Unit": POLLUTANT_UNITS[name],
                "CPCB reference": limit,
                "Share of reference (%)": round(concentration / limit * 100.0, 1),
                "Sub-index": round(float(breakdown.get(name, float("nan"))), 1),
                "Status": "Above reference" if concentration > limit else "Within reference",
            }
        )
    chart_data = pd.DataFrame(chart_rows)

    view_mode = st.radio(
        "Chart view",
        ["Share of CPCB reference", "Sub-index contribution"],
        horizontal=True,
        help="Both views are unit-safe; raw concentrations stay in the table below.",
    )

    c_col1, c_col2 = st.columns([1.55, 1.45])

    with c_col1:
        if view_mode == "Share of CPCB reference":
            bars = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Pollutant:N", sort=None, title=None),
                    y=alt.Y("Share of reference (%):Q", title="% of CPCB reference value"),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=["Within reference", "Above reference"],
                            range=["#2ecc71", "#e74c3c"],
                        ),
                        legend=alt.Legend(title="Regulatory status"),
                    ),
                    tooltip=[
                        "Pollutant",
                        alt.Tooltip("Concentration:Q", format=".1f"),
                        "Unit",
                        alt.Tooltip("CPCB reference:Q", format=".1f"),
                        alt.Tooltip("Share of reference (%):Q", format=".1f"),
                        "Status",
                    ],
                )
            )
            threshold = (
                alt.Chart(pd.DataFrame({"y": [100.0]}))
                .mark_rule(strokeDash=[6, 4], color="#888888")
                .encode(y="y:Q")
            )
            st.altair_chart(bars + threshold, width="stretch")
            st.caption("The dashed line marks 100% of the CPCB reference concentration.")
        else:
            sub_bars = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusEnd=6)
                .encode(
                    y=alt.Y("Pollutant:N", sort="-x", title=None),
                    x=alt.X("Sub-index:Q", title="CPCB sub-index (0–500)", scale=alt.Scale(domain=[0, 500])),
                    color=alt.Color(
                        "Sub-index:Q",
                        scale=alt.Scale(
                            domain=[0, 50, 100, 200, 300, 400, 500],
                            range=["#00b894", "#2ed573", "#f1c40f", "#e67e22", "#e74c3c", "#8b0000", "#4a0018"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        "Pollutant",
                        alt.Tooltip("Sub-index:Q", format=".1f"),
                        alt.Tooltip("Concentration:Q", format=".1f"),
                        "Unit",
                    ],
                )
            )
            st.altair_chart(sub_bars, width="stretch")
            st.caption("The overall AQI is the maximum of these six sub-indices.")

    with c_col2:
        st.markdown("##### 📋 CPCB sub-index breakdown")
        display_df = chart_data[
            ["Pollutant", "Concentration", "Unit", "CPCB reference", "Share of reference (%)", "Sub-index", "Status"]
        ].copy()
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Share of reference (%)": st.column_config.ProgressColumn(
                    "Share of reference",
                    min_value=0,
                    max_value=200,
                    format="%.0f%%",
                    help="Values above 100% exceed the CPCB reference concentration.",
                ),
                "Sub-index": st.column_config.ProgressColumn(
                    "CPCB sub-index",
                    min_value=0,
                    max_value=500,
                    format="%.1f",
                ),
            },
        )
        if dominant_pollutant:
            st.info(
                f"**Dominant pollutant:** {dominant_pollutant} at sub-index "
                f"{breakdown[dominant_pollutant]:.1f} ({exact_cpcb_aqi:.1f} overall AQI)."
            )
        st.caption(
            "Under CPCB standards the overall AQI is governed by the maximum sub-index across "
            "all monitored criteria pollutants."
        )

# ===========================================================================
# TAB 3 — SCENARIO LAB
# ===========================================================================
with tab_lab:
    st.subheader("🔬 Scenario Lab")
    st.caption(
        "Compare every preset side by side, then stress-test the current inputs. "
        "All values are recomputed live from the same model and CPCB formulas."
    )

    comparison_rows = []
    for name, scenario_values in SCENARIOS.items():
        scenario_frame = build_feature_frame(scenario_values, predict_date)
        scenario_breakdown = sub_index_breakdown(scenario_values)
        scenario_cpcb = round(float(max(scenario_breakdown.values())), 1)
        scenario_ml = None
        if model is not None:
            try:
                scenario_ml = round(
                    float(np.clip(model.predict(scenario_frame)[0], 0.0, 500.0)), 1
                )
            except Exception:
                scenario_ml = None
        comparison_rows.append(
            {
                "Scenario": name,
                "CPCB AQI": scenario_cpcb,
                "Model AQI": scenario_ml,
                "Category": map_aqi_to_category(scenario_cpcb),
                "Dominant pollutant": max(scenario_breakdown, key=scenario_breakdown.get),
                "Delta (model − CPCB)": (
                    round(scenario_ml - scenario_cpcb, 1) if scenario_ml is not None else None
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    lab_col1, lab_col2 = st.columns([1.35, 1.65])
    with lab_col1:
        st.markdown("##### 🧭 Preset comparison")
        st.dataframe(
            comparison_df.drop(columns=["Delta (model − CPCB)"]),
            width="stretch",
            hide_index=True,
            column_config={
                "CPCB AQI": st.column_config.NumberColumn(format="%.1f"),
                "Model AQI": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.download_button(
            "⬇️ Download comparison as CSV",
            data=comparison_df.to_csv(index=False),
            file_name="aqi-scenario-comparison.csv",
            mime="text/csv",
            width="stretch",
        )

    with lab_col2:
        long_df = comparison_df.melt(
            id_vars=["Scenario", "Category"],
            value_vars=["CPCB AQI", "Model AQI"],
            var_name="Method",
            value_name="AQI",
        ).dropna(subset=["AQI"])
        grouped_bars = (
            alt.Chart(long_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Scenario:N", sort=None, title=None, axis=alt.Axis(labelAngle=-15)),
                xOffset="Method:N",
                y=alt.Y("AQI:Q", scale=alt.Scale(domain=[0, 500]), title="AQI"),
                color=alt.Color(
                    "Method:N",
                    scale=alt.Scale(
                        domain=["CPCB AQI", "Model AQI"],
                        range=["#3498db", "#9b59b6"],
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["Scenario", "Method", alt.Tooltip("AQI:Q", format=".1f"), "Category"],
            )
        )
        st.altair_chart(grouped_bars, width="stretch")

    st.divider()
    st.markdown("##### 🎚️ Sensitivity analysis for the current inputs")
    st.caption(
        "Scale every pollutant up or down to see how quickly the CPCB and model estimates respond. "
        "Weather values are held constant."
    )

    multiplier = st.slider(
        "Pollutant multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="1.0 = current inputs. 1.5 = every pollutant 50% higher.",
        key="multiplier",
    )

    curve_rows = []
    for factor in np.round(np.arange(0.5, 2.01, 0.1), 2):
        scaled = {
            key: (values[key] * factor if key in POLLUTANT_INPUTS else values[key])
            for key in INPUT_BOUNDS
        }
        scaled_cpcb = compute_aqi_from_concentrations(scaled)
        scaled_ml = None
        if model is not None:
            try:
                scaled_ml = float(
                    np.clip(
                        model.predict(build_feature_frame(scaled, predict_date))[0],
                        0.0,
                        500.0,
                    )
                )
            except Exception:
                scaled_ml = None
        curve_rows.append(
            {"Multiplier": float(factor), "CPCB AQI": scaled_cpcb, "Model AQI": scaled_ml}
        )
    curve_df = pd.DataFrame(curve_rows).melt(
        id_vars=["Multiplier"], var_name="Method", value_name="AQI"
    ).dropna(subset=["AQI"])

    curve = (
        alt.Chart(curve_df)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5)
        .encode(
            x=alt.X("Multiplier:Q", title="Pollutant multiplier", scale=alt.Scale(domain=[0.5, 2.0])),
            y=alt.Y("AQI:Q", title="AQI", scale=alt.Scale(domain=[0, 500])),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(
                    domain=["CPCB AQI", "Model AQI"], range=["#3498db", "#9b59b6"]
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Method", alt.Tooltip("Multiplier:Q", format=".2f"), alt.Tooltip("AQI:Q", format=".1f")],
        )
    )
    current_rule = (
        alt.Chart(pd.DataFrame({"x": [multiplier]}))
        .mark_rule(strokeDash=[5, 4], color="#888888")
        .encode(x="x:Q")
    )
    st.altair_chart(curve + current_rule, width="stretch")

    scaled_current = {
        key: (values[key] * multiplier if key in POLLUTANT_INPUTS else values[key])
        for key in INPUT_BOUNDS
    }
    scaled_aqi = compute_aqi_from_concentrations(scaled_current)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Scaled CPCB AQI", f"{scaled_aqi:.1f}")
    summary_cols[1].metric(
        "Scaled category", map_aqi_to_category(scaled_aqi)
    )
    summary_cols[2].metric("Multiplier", f"{multiplier:.2f}×")

# ===========================================================================
# TAB 4 — MODEL GOVERNANCE
# ===========================================================================
with tab_governance:
    st.subheader("🧠 Machine Learning Governance & Model Performance")

    gov_col1, gov_col2 = st.columns(2)
    with gov_col1:
        st.markdown("##### 🏆 Champion model architecture")
        st.markdown(
            """
        * **Algorithm:** `Random Forest Regressor`
        * **Ensemble size:** `150 trees`
        * **Input fields:** `21 same-day pollutant, weather and calendar values`
        * **Verification:** `SHA-256 cryptographic checksum enforced on load`
        * **Training data:** generated synthetic examples; no live sensor feed
        """
        )
        st.caption(
            "Feature construction is shared with the pipeline through `src/feature_builder.py`, "
            "so the app cannot drift from the training schema."
        )

    with gov_col2:
        st.markdown("##### 📈 Chronological holdout metrics")
        temporal_results_path = os.path.join(
            PROJECT_DIR, "reports", "temporal_evaluation.csv"
        )
        mtime = (
            os.path.getmtime(temporal_results_path)
            if os.path.exists(temporal_results_path)
            else 0.0
        )
        temporal_results = load_temporal_results(temporal_results_path, mtime)
        if temporal_results is not None and not temporal_results.empty:
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
                    "These metrics come from a separate model fit on the training-date partition; "
                    "the interactive app's model is fitted on all available training rows."
                )

            display_temporal = temporal_results.copy()
            display_temporal["Category Accuracy"] = display_temporal[
                "Category Accuracy"
            ].map("{:.2%}".format)
            st.dataframe(display_temporal, width="stretch", hide_index=True)
        else:
            st.info(
                "Run `python src/evaluate_temporal.py` to generate date-based holdout metrics."
            )

    st.markdown("---")
    st.markdown("##### 🔎 Global model feature importance")
    if global_feature_importances is not None:
        importance_chart = (
            alt.Chart(global_feature_importances)
            .mark_bar(cornerRadiusEnd=5, color="#4c78a8")
            .encode(
                x=alt.X("Importance:Q", title="Relative feature importance"),
                y=alt.Y("Feature:N", sort=None, title=None),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
            )
            .properties(height=280)
        )
        st.altair_chart(importance_chart, width="stretch")
        st.caption(
            "Random Forest importances summarise feature use across training data; they are not "
            "causal explanations or per-scenario contributions."
        )
    else:
        st.info("Global feature importance is not available for the loaded model.")

    st.markdown("---")
    st.markdown("##### 📚 Data provenance & limitations")
    st.markdown(
        "- The committed model is trained on `data/raw/synthetic_air_quality_data.csv`, "
        "a locally generated dataset of 51,000 simulated records.\n"
        "- Pollutant targets are derived from CPCB sub-indices using contiguous breakpoints; "
        "no concentration can fall between bands.\n"
        "- The model estimates same-day AQI from same-day inputs. It is **not** a forecast and "
        "does not use live sensor data.\n"
        "- Tree ensembles cannot extrapolate beyond their training range; predictions are capped "
        "at 500."
    )

st.divider()
st.markdown(
    '<div class="app-footer">Air Quality Prediction System · CPCB reference logic · '
    "Synthetic training data · Built with Streamlit</div>",
    unsafe_allow_html=True,
)
