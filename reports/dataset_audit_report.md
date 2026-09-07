# Comprehensive Dataset Audit Report

- **Dataset Source:** `synthetic_air_quality_data.csv`
- **Total Rows:** 51000
- **Total Columns:** 12
- **Duplicate Rows:** 0
- **Missing AQI Rows:** 0

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
| count |     51000     |     51000     |    51000     |    51000     |  51000      |  51000      |          51000       |        51000      |         51000       | 51000     |
| mean  |       131.946 |       268.174 |      209.544 |      648.951 |     14.6578 |    292.146  |             22.2586  |           62.2934 |             4.02815 |   249.92  |
| std   |       121.313 |       205.523 |      161.474 |      662.076 |     14.7671 |    299.515  |              9.31483 |           16.2917 |             2.74482 |   157.914 |
| min   |         2     |         5.25  |        3     |        1.01  |      0.1    |      4.53   |             -5       |           25      |             0.2     |    11.6   |
| 25%   |        45.22  |        83.37  |       60.1   |       60.09  |      1.51   |     74.2175 |             16       |           50.5    |             1.88    |    95.8   |
| 50%   |        90.05  |       250.055 |      180.045 |      380.55  |     10.005  |    162.4    |             22.8     |           62.3    |             3.3     |   242.1   |
| 75%   |       185.41  |       395.277 |      340.5   |     1186.06  |     25.42   |    474.745  |             29.2     |           74.5    |             5.56    |   390.2   |
| max   |       479.98  |      1063.68  |      550     |     1999.97  |     48      |   1044.97   |             45       |          100      |            15.16    |   500     |

## Target AQI Category Balance

| Category              |   Count | Percentage   |
|:----------------------|--------:|:-------------|
| Severe (>400)         |    9637 | 18.90%       |
| Very Poor (301-400)   |    9476 | 18.58%       |
| Moderate (101-200)    |    8963 | 17.57%       |
| Satisfactory (51-100) |    7818 | 15.33%       |
| Good (0-50)           |    7679 | 15.06%       |
| Poor (201-300)        |    7427 | 14.56%       |
