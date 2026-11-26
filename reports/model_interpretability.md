# Model Interpretability & Selection Report

## Selected Champion Model
- **Algorithm:** Random Forest Regressor (`n_estimators=150`, `min_samples_split=2`, `min_samples_leaf=1`)
- **Cryptographic Security:** Signed with SHA-256 integrity manifest (`model.joblib.sha256`).
- **Training data:** The committed model uses the generated synthetic dataset, not observed sensor readings.
- **Interpretation:** The target is same-day AQI derived from pollutant sub-indices. Random row-split scores do not establish future-date forecasting performance; see `reports/temporal_evaluation_report.md` for a date-grouped holdout.

## Feature Importances Ranking
These are global Random Forest importances across training examples. They summarize feature use; they are not causal explanations or per-scenario contributions.

|    | Feature                |   Importance |
|---:|:-----------------------|-------------:|
|  1 | pm10_median            |  0.469492    |
|  0 | pm25_median            |  0.154948    |
|  2 | no2_median             |  0.135174    |
| 12 | Gaseous_Index          |  0.127883    |
|  3 | so2_median             |  0.0685686   |
|  4 | co_median              |  0.0375535   |
|  5 | o3_median              |  0.00539898  |
|  7 | humidity_median        |  0.000198201 |
|  6 | temperature_median     |  0.000184958 |
|  8 | wind-speed_median      |  0.000106232 |
|  9 | PM25_PM10_ratio        |  9.84473e-05 |
| 13 | Thermal_Humidity_Index |  8.66522e-05 |
| 10 | PM25_Wind_Ratio        |  7.68314e-05 |
| 14 | Month                  |  5.89137e-05 |
| 11 | Stagnation_Index       |  5.10123e-05 |
| 16 | Month_Sin              |  2.70583e-05 |
| 18 | DayOfWeek_Sin          |  2.68014e-05 |
| 15 | DayOfWeek              |  2.54844e-05 |
| 17 | Month_Cos              |  2.04695e-05 |
| 19 | DayOfWeek_Cos          |  8.96185e-06 |
| 20 | Season_Autumn          |  6.41778e-06 |
| 22 | Season_Summer          |  2.41908e-06 |
| 23 | Season_Winter          |  2.34612e-06 |
| 21 | Season_Spring          |  1.5977e-07  |

## Interpreting the Importances
The synthetic target is calculated from pollutant sub-indices, so pollutant concentrations dominate the learned AQI estimate. These importances do not show that the model discovered causal relationships, nor do they identify the contribution of a feature for an individual scenario.