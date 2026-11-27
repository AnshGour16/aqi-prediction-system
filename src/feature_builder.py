"""Build the model's 21-feature input frame from raw scenario values.

Keeping this in one place means the Streamlit app, the inference tests and any
future caller construct features in exactly the same order and with the same
formulas used during training.
"""

from __future__ import annotations

import datetime as _datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from .aqi_utils import get_season
except ImportError:  # Running from `src/` as a script
    from aqi_utils import get_season

#: Canonical scenario keys in the order the app exposes them.
SCENARIO_KEYS = (
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "temp",
    "humidity",
    "wind_speed",
)

#: Accepted aliases so callers can pass either short names or dataset columns.
_ALIASES = {
    "pm25": ("pm25", "pm25_median"),
    "pm10": ("pm10", "pm10_median"),
    "no2": ("no2", "no2_median"),
    "so2": ("so2", "so2_median"),
    "co": ("co", "co_median"),
    "o3": ("o3", "o3_median"),
    "temp": ("temp", "temperature", "temperature_median"),
    "humidity": ("humidity", "humidity_median"),
    "wind_speed": ("wind_speed", "wind-speed_median", "wind_speed_median"),
}


def normalise_values(values: Mapping[str, Any]) -> dict[str, float]:
    """Return a dict keyed by :data:`SCENARIO_KEYS` with float values."""
    normalised: dict[str, float] = {}
    for key in SCENARIO_KEYS:
        for alias in _ALIASES[key]:
            if alias in values and values[alias] is not None:
                normalised[key] = float(values[alias])
                break
        else:
            raise KeyError(f"Missing scenario value for {key!r}")
    return normalised


def build_feature_frame(
    values: Mapping[str, Any],
    observation_date: _datetime.date | None = None,
) -> pd.DataFrame:
    """Return a one-row dataframe ready for ``model.predict``."""
    data = normalise_values(values)
    date = observation_date or _datetime.date.today()
    month = date.month
    day_of_week = date.weekday()
    season = get_season(month)

    pm25 = data["pm25"]
    pm10 = data["pm10"]
    wind_speed = data["wind_speed"]

    pm25_pm10_ratio = pm25 / pm10 if pm10 > 0 else 0.5
    pm25_wind_ratio = pm25 / wind_speed if wind_speed > 0 else 0.0
    stagnation_index = pm25 / (wind_speed + 0.1)
    gaseous_index = data["no2"] + data["so2"] + (data["co"] * 10.0)
    thermal_humidity_index = data["temp"] * (data["humidity"] / 100.0)

    return pd.DataFrame(
        [
            {
                "pm25_median": pm25,
                "pm10_median": pm10,
                "no2_median": data["no2"],
                "so2_median": data["so2"],
                "co_median": data["co"],
                "o3_median": data["o3"],
                "temperature_median": data["temp"],
                "humidity_median": data["humidity"],
                "wind-speed_median": wind_speed,
                "PM25_PM10_ratio": pm25_pm10_ratio,
                "PM25_Wind_Ratio": pm25_wind_ratio,
                "Stagnation_Index": stagnation_index,
                "Gaseous_Index": gaseous_index,
                "Thermal_Humidity_Index": thermal_humidity_index,
                "Month": month,
                "DayOfWeek": day_of_week,
                "Month_Sin": float(np.sin(2 * np.pi * month / 12.0)),
                "Month_Cos": float(np.cos(2 * np.pi * month / 12.0)),
                "DayOfWeek_Sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
                "DayOfWeek_Cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
                "Season": season,
            }
        ]
    )
