"""Shared CPCB AQI calculations.

This module is the single source of truth for the CPCB breakpoint tables,
sub-index interpolation, category mapping and season labels used by both the
data pipeline (``src/``) and the Streamlit app (``app/``).

The breakpoint concentration bands are **contiguous**. Earlier versions of the
project used rounded lower bounds such as ``30.1`` after a band ending at
``30.0``. Any concentration landing in the resulting 0.1-wide hole (for
example ``PM10 = 50.05``) silently fell through to the 500-point cap and
produced a false "Severe" AQI. Keeping the bands contiguous removes that
failure mode entirely.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: Model/data column name for each criteria pollutant.
POLLUTANT_COLUMNS = {
    "PM2.5": "pm25_median",
    "PM10": "pm10_median",
    "NO2": "no2_median",
    "SO2": "so2_median",
    "CO": "co_median",
    "O3": "o3_median",
}

#: Accepted keys for each pollutant: display name, short app key, dataset column.
POLLUTANT_ALIASES = {
    "PM2.5": ("PM2.5", "pm25", "pm25_median"),
    "PM10": ("PM10", "pm10", "pm10_median"),
    "NO2": ("NO2", "no2", "no2_median"),
    "SO2": ("SO2", "so2", "so2_median"),
    "CO": ("CO", "co", "co_median"),
    "O3": ("O3", "o3", "o3_median"),
}

#: Display unit for each criteria pollutant.
POLLUTANT_UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "µg/m³",
    "SO2": "µg/m³",
    "CO": "mg/m³",
    "O3": "µg/m³",
}

#: CPCB 24-hour (particulates, NO2, SO2) / 8-hour (CO, O3) reference values.
CPCB_STANDARDS = {
    "PM2.5": 60.0,
    "PM10": 100.0,
    "NO2": 80.0,
    "SO2": 80.0,
    "CO": 2.0,
    "O3": 100.0,
}

#: Ordered CPCB AQI categories.
CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

#: Contiguous concentration bands mapped to index bands.
#: Each row is ``(C_low, C_high, I_low, I_high)``.
BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    "PM2.5": [
        (0.0, 30.0, 0.0, 50.0),
        (30.0, 60.0, 51.0, 100.0),
        (60.0, 90.0, 101.0, 200.0),
        (90.0, 120.0, 201.0, 300.0),
        (120.0, 250.0, 301.0, 400.0),
        (250.0, 500.0, 401.0, 500.0),
    ],
    "PM10": [
        (0.0, 50.0, 0.0, 50.0),
        (50.0, 100.0, 51.0, 100.0),
        (100.0, 250.0, 101.0, 200.0),
        (250.0, 350.0, 201.0, 300.0),
        (350.0, 430.0, 301.0, 400.0),
        (430.0, 600.0, 401.0, 500.0),
    ],
    "NO2": [
        (0.0, 40.0, 0.0, 50.0),
        (40.0, 80.0, 51.0, 100.0),
        (80.0, 180.0, 101.0, 200.0),
        (180.0, 280.0, 201.0, 300.0),
        (280.0, 400.0, 301.0, 400.0),
        (400.0, 600.0, 401.0, 500.0),
    ],
    "SO2": [
        (0.0, 40.0, 0.0, 50.0),
        (40.0, 80.0, 51.0, 100.0),
        (80.0, 380.0, 101.0, 200.0),
        (380.0, 800.0, 201.0, 300.0),
        (800.0, 1600.0, 301.0, 400.0),
        (1600.0, 2000.0, 401.0, 500.0),
    ],
    "CO": [
        (0.0, 1.0, 0.0, 50.0),
        (1.0, 2.0, 51.0, 100.0),
        (2.0, 10.0, 101.0, 200.0),
        (10.0, 17.0, 201.0, 300.0),
        (17.0, 34.0, 301.0, 400.0),
        (34.0, 50.0, 401.0, 500.0),
    ],
    "O3": [
        (0.0, 50.0, 0.0, 50.0),
        (50.0, 100.0, 51.0, 100.0),
        (100.0, 168.0, 101.0, 200.0),
        (168.0, 208.0, 201.0, 300.0),
        (208.0, 748.0, 301.0, 400.0),
        (748.0, 1000.0, 401.0, 500.0),
    ],
}


def _as_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is unusable."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def calculate_sub_index(value: Any, pollutant: str) -> float:
    """Return the CPCB sub-index for one pollutant concentration.

    Values below zero or missing return ``NaN``. Values above the last
    breakpoint are capped at 500. No concentration can fall between bands,
    because the bands are contiguous.
    """
    if pollutant not in BREAKPOINTS:
        raise KeyError(f"Unknown pollutant: {pollutant!r}")

    number = _as_float(value)
    if number is None or number < 0:
        return float("nan")

    table = BREAKPOINTS[pollutant]
    for c_low, c_high, i_low, i_high in table:
        # Inclusive on both edges with the first matching band winning. Bands
        # are contiguous, so a value can never fall through to the 500 cap.
        if c_low <= number <= c_high:
            if c_high == c_low:
                return float(i_high)
            sub_index = i_low + (i_high - i_low) / (c_high - c_low) * (number - c_low)
            return round(sub_index, 2)

    # Above the highest breakpoint: CPCB caps the index at 500.
    return 500.0


def sub_index_breakdown(values: Mapping[str, Any]) -> dict[str, float]:
    """Return the non-null sub-index for every pollutant in ``values``.

    ``values`` may use display names (``"PM2.5"``), short app keys
    (``"pm25"``) or dataset column names (``"pm25_median"``).
    """
    breakdown: dict[str, float] = {}
    for pollutant, aliases in POLLUTANT_ALIASES.items():
        raw = next((values[key] for key in aliases if key in values), None)
        sub_index = calculate_sub_index(raw, pollutant)
        if not math.isnan(sub_index):
            breakdown[pollutant] = sub_index
    return breakdown


def compute_aqi_from_concentrations(
    values: Mapping[str, Any],
    min_pollutants: int = 3,
    require_particulate: bool = True,
) -> float:
    """Compute the overall CPCB AQI as the maximum sub-index.

    Mirrors the CPCB rule that at least three pollutants, including a
    particulate measure, must be available before an overall index is
    reported.
    """
    breakdown = sub_index_breakdown(values)
    has_particulate = "PM2.5" in breakdown or "PM10" in breakdown
    if len(breakdown) < min_pollutants or (require_particulate and not has_particulate):
        return float("nan")
    return max(breakdown.values())


def map_aqi_to_category(value: Any) -> str:
    """Map a numeric AQI value to its CPCB category label."""
    number = _as_float(value)
    if number is None:
        return "Unknown"
    if number <= 50:
        return "Good"
    if number <= 100:
        return "Satisfactory"
    if number <= 200:
        return "Moderate"
    if number <= 300:
        return "Poor"
    if number <= 400:
        return "Very Poor"
    return "Severe"


def get_season(month: int) -> str:
    """Return the season label used by the feature-engineering pipeline."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"
