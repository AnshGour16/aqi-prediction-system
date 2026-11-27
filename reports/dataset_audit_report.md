# Comprehensive Dataset Audit Report

- **Dataset Source:** `synthetic_air_quality_data.csv`
- **Dataset Type:** Locally generated synthetic examples; not measured sensor observations.
- **AQI Target:** Derived from CPCB pollutant sub-indices by `src/generate_dataset.py`.
- **Total Rows:** 51000
- **Total Columns:** 12
- **Duplicate Rows:** 0
- **Missing AQI Rows:** 0

## Target Integrity Checks

- **Category/AQI mismatches:** 0
- **Rows at the 500 AQI cap:** 12
- **Rows with PM10 < PM2.5:** 0
- **Sub-index 500 values below a top breakpoint (gap bug):** 0

## Feature Nullability & Data Types

| Column Name          | Data Type   |   Non-Null Count |   Missing Count | Missing %   |
|:---------------------|:------------|-----------------:|----------------:|:------------|
| `Date`               | str         |            51000 |               0 | 0.00%       |
| `pm25_median`        | float64     |            51000 |               0 | 0.00%       |
| `pm10_median`        | float64     |            51000 |               0 | 0.00%       |
| `no2_median`         | float64     |            51000 |               0 | 0.00%       |
| `so2_median`         | float64     |            51000 |               0 | 0.00%       |
| `co_median`          | float64     |            51000 |               0 | 0.00%       |
| `o3_median`          | float64     |            51000 |               0 | 0.00%       |
| `temperature_median` | float64     |            51000 |               0 | 0.00%       |
| `humidity_median`    | float64     |            51000 |               0 | 0.00%       |
| `wind-speed_median`  | float64     |            51000 |               0 | 0.00%       |
| `AQI`                | float64     |            51000 |               0 | 0.00%       |
| `Category`           | str         |            51000 |               0 | 0.00%       |

## Statistical Feature Distributions

|       |   pm25_median |   pm10_median |   no2_median |   so2_median |   co_median |   o3_median |   temperature_median |   humidity_median |   wind-speed_median |       AQI |
|:------|--------------:|--------------:|-------------:|-------------:|------------:|------------:|---------------------:|------------------:|--------------------:|----------:|
| count |     51000     |    51000      |    51000     |    51000     |  51000      |   51000     |          51000       |        51000      |         51000       | 51000     |
| mean  |       139.82  |      247.087  |      209.386 |      649.831 |     14.6725 |     291.158 |             22.2843  |           62.3224 |             4.02146 |   250.702 |
| std   |       114.644 |      174.893  |      161.64  |      662.915 |     14.7525 |     296.548 |              9.33141 |           16.3268 |             2.73897 |   155.755 |
| min   |         2.34  |        5.01   |        3     |        1.01  |      0.1    |       5     |             -5       |           25      |             0.2     |     8.6   |
| 25%   |        47.43  |       75.1675 |       59.85  |       60.51  |      1.51   |      74.8   |             16       |           50.5    |             1.87    |    96.4   |
| 50%   |       101.695 |      250.05   |      180.04  |      380.495 |     10.005  |     168.05  |             22.9     |           62.2    |             3.29    |   239.1   |
| 75%   |       250     |      390.055  |      339.465 |     1194.74  |     25.49   |     470.635 |             29.2     |           74.6    |             5.58    |   400     |
| max   |       480     |      599.99   |      549.97  |     2000     |     48      |     950     |             45       |          100      |            14.04    |   500     |

## Target AQI Category Balance

| Category     |   Count | Percentage   |
|:-------------|--------:|:-------------|
| Good         |    8500 | 16.67%       |
| Satisfactory |    8500 | 16.67%       |
| Very Poor    |    8500 | 16.67%       |
| Moderate     |    8500 | 16.67%       |
| Poor         |    8500 | 16.67%       |
| Severe       |    8500 | 16.67%       |
