import unittest

import pandas as pd

from src.evaluate_temporal import split_by_date


class ChronologicalSplitTests(unittest.TestCase):
    def test_keeps_duplicate_dates_in_one_split(self):
        frame = pd.DataFrame({
            "Date": [
                "2024-01-01", "2024-01-01",
                "2024-01-02", "2024-01-02",
                "2024-01-03", "2024-01-03",
                "2024-01-04", "2024-01-04",
                "2024-01-05", "2024-01-05"
            ],
            "value": range(10)
        })

        train_df, test_df, cutoff = split_by_date(frame, test_fraction=0.4)

        self.assertEqual(cutoff, pd.Timestamp("2024-01-04"))
        self.assertLess(pd.to_datetime(train_df["Date"]).max(), cutoff)
        self.assertGreaterEqual(pd.to_datetime(test_df["Date"]).min(), cutoff)
        self.assertEqual(len(train_df), 6)
        self.assertEqual(len(test_df), 4)

    def test_rejects_missing_or_invalid_dates(self):
        with self.assertRaisesRegex(ValueError, "Date column"):
            split_by_date(pd.DataFrame({"value": [1, 2]}))
        with self.assertRaisesRegex(ValueError, "invalid dates"):
            split_by_date(pd.DataFrame({"Date": ["2024-01-01", "not-a-date"]}))

    def test_rejects_invalid_test_fraction(self):
        frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"]})
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            split_by_date(frame, test_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
