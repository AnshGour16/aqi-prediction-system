"""Unit tests for the shared CPCB AQI maths.

These lock in the fix for the historical breakpoint-gap bug, where a
concentration such as ``PM10 = 50.05`` fell between the rounded bands and was
scored as 500 ("Severe") instead of ~51.
"""

import math
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aqi_utils import (  # noqa: E402
    BREAKPOINTS,
    CATEGORY_ORDER,
    calculate_sub_index,
    compute_aqi_from_concentrations,
    map_aqi_to_category,
)


class SubIndexTests(unittest.TestCase):
    def test_reference_concentrations_map_to_100(self):
        """The CPCB 24h/8h reference values sit at the top of Satisfactory."""
        references = {
            "PM2.5": 60.0,
            "PM10": 100.0,
            "NO2": 80.0,
            "SO2": 80.0,
            "CO": 2.0,
            "O3": 100.0,
        }
        for pollutant, concentration in references.items():
            with self.subTest(pollutant=pollutant):
                self.assertAlmostEqual(
                    calculate_sub_index(concentration, pollutant), 100.0, places=1
                )

    def test_gap_values_are_not_capped(self):
        """Values inside the old 0.1-wide holes must interpolate normally."""
        cases = [
            ("PM10", 50.05, 51.0),
            ("PM10", 100.05, 101.0),
            ("PM2.5", 30.05, 51.0),
            ("NO2", 40.05, 51.0),
            ("SO2", 80.05, 101.0),
            ("CO", 1.005, 51.0),
            ("O3", 50.05, 51.0),
        ]
        for pollutant, concentration, expected in cases:
            with self.subTest(pollutant=pollutant, concentration=concentration):
                result = calculate_sub_index(concentration, pollutant)
                self.assertAlmostEqual(result, expected, delta=0.5)
                self.assertLess(result, 200.0)

    def test_no_hidden_gaps_or_spikes(self):
        """Sweeping each pollutant must be smooth and monotonic."""
        step = 0.05
        for pollutant, table in BREAKPOINTS.items():
            upper = table[-1][1]
            max_slope = max(
                (i_high - i_low) / (c_high - c_low)
                for c_low, c_high, i_low, i_high in table
            )
            # A band boundary adds exactly 1 index point; allow that plus the
            # slope of the steepest band over one sweep step.
            allowed_jump = max_slope * step + 1.5
            previous = -math.inf
            value = 0.0
            while value <= upper:
                sub_index = calculate_sub_index(value, pollutant)
                with self.subTest(pollutant=pollutant, value=round(value, 3)):
                    self.assertFalse(math.isnan(sub_index))
                    self.assertGreaterEqual(sub_index, previous - 1e-9)
                    self.assertLessEqual(sub_index, 500.0)
                    if previous > -math.inf:
                        self.assertLess(sub_index - previous, allowed_jump)
                previous = sub_index
                value = round(value + step, 4)

    def test_above_max_caps_at_500(self):
        for pollutant, table in BREAKPOINTS.items():
            with self.subTest(pollutant=pollutant):
                self.assertEqual(
                    calculate_sub_index(table[-1][1] + 1000.0, pollutant), 500.0
                )

    def test_missing_and_negative_return_nan(self):
        for value in (None, float("nan"), -1.0, "not-a-number"):
            with self.subTest(value=value):
                self.assertTrue(math.isnan(calculate_sub_index(value, "PM2.5")))

    def test_unknown_pollutant_raises(self):
        with self.assertRaises(KeyError):
            calculate_sub_index(10.0, "CH4")


class OverallAqiTests(unittest.TestCase):
    def test_requires_three_pollutants_and_a_particulate(self):
        two_only = {"PM2.5": 10.0, "PM10": 20.0}
        self.assertTrue(math.isnan(compute_aqi_from_concentrations(two_only)))

        no_particulate = {"NO2": 10.0, "SO2": 10.0, "CO": 0.5}
        self.assertTrue(math.isnan(compute_aqi_from_concentrations(no_particulate)))

        valid = {"PM2.5": 10.0, "NO2": 10.0, "CO": 0.5}
        # Sub-indices: PM2.5 = 16.67, NO2 = 12.5, CO = 25.0 -> overall 25.0.
        self.assertAlmostEqual(compute_aqi_from_concentrations(valid), 25.0, places=1)

    def test_overall_is_max_sub_index(self):
        values = {"PM2.5": 10.0, "PM10": 120.0, "NO2": 10.0, "SO2": 10.0}
        expected = max(
            calculate_sub_index(values["PM2.5"], "PM2.5"),
            calculate_sub_index(values["PM10"], "PM10"),
            calculate_sub_index(values["NO2"], "NO2"),
            calculate_sub_index(values["SO2"], "SO2"),
        )
        self.assertAlmostEqual(compute_aqi_from_concentrations(values), expected)


class CategoryTests(unittest.TestCase):
    def test_boundaries(self):
        cases = {
            0: "Good",
            50: "Good",
            50.1: "Satisfactory",
            100: "Satisfactory",
            100.1: "Moderate",
            200: "Moderate",
            200.1: "Poor",
            300: "Poor",
            300.1: "Very Poor",
            400: "Very Poor",
            400.1: "Severe",
            500: "Severe",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(map_aqi_to_category(value), expected)

    def test_unknown_for_missing(self):
        self.assertEqual(map_aqi_to_category(None), "Unknown")
        self.assertEqual(map_aqi_to_category(float("nan")), "Unknown")

    def test_category_order_is_complete(self):
        self.assertEqual(len(CATEGORY_ORDER), 6)
        self.assertEqual(CATEGORY_ORDER[0], "Good")
        self.assertEqual(CATEGORY_ORDER[-1], "Severe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
