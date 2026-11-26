# Chronological AQI Evaluation

## Evaluation design
- **Configured source:** `synthetic_air_quality_data.csv` (the preprocessing pipeline prioritizes this generated dataset when it is present).
- **Rows evaluated:** 51,000.
- **Training rows:** 40,676 through 2024-03-13.
- **Test rows:** 10,324 from 2024-03-14 onward.
- **Date groups:** 1,168 training dates and 292 test dates; no date appears in both sets.
- **Target meaning:** same-day AQI derived from pollutant sub-indices. This evaluates AQI estimation on later dates; it is not a forecast of future air quality.
- **Synthetic-data caveat:** the default generated dataset is simulated, not measured sensor data. Strong performance on it does not establish real-world accuracy.

## Results
| Model                                 |   Training Rows |   Test Rows | Training Through   | Test From   |      MAE |     RMSE |      R2 |   Category Accuracy |
|:--------------------------------------|----------------:|------------:|:-------------------|:------------|---------:|---------:|--------:|--------------------:|
| Mean baseline                         |           40676 |       10324 | 2024-03-13         | 2024-03-14  | 139.8314 | 157.6600 | -0.0000 |              0.1424 |
| Random Forest (chronological holdout) |           40676 |       10324 | 2024-03-13         | 2024-03-14  |   0.6093 |   5.9093 |  0.9986 |              0.9900 |

MAE and RMSE are measured in AQI points. Category accuracy compares the derived CPCB bands after clipping estimates to the 0–500 range.
