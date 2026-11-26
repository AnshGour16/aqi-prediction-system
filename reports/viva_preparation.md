# Viva Preparation Guide — Air Quality Prediction System

This guide contains 25+ likely viva/presentation questions with concise, technically rigorous, and beginner-readable answers to prepare for your project demonstration.

---

### Section 1: Dataset & Preprocessing

#### Q1: Why is Air Quality Index (AQI) prediction modeled as a regression problem?
**Answer:** AQI is a continuous numerical value representing pollution concentration scales. Since our goal is to predict the exact numerical index (e.g. 156.4) from multiple continuous environmental inputs, it is a regression problem. Mapping this continuous prediction to a discrete category (like 'Moderate') is a post-processing step, not a classification model.

#### Q2: How did you select your dataset, and what variables are included?
**Answer:** The checked-in model uses 51,000 locally generated synthetic examples because preprocessing prioritizes `synthetic_air_quality_data.csv` when it is present. The repository also contains a public DEAP-derived CSV, but it is only a fallback and was not used for the committed processed data. The synthetic rows do not represent actual cities or sensor observations.

#### Q3: Your raw dataset did not contain AQI. How did you handle this?
**Answer:** The synthetic generator calculates each pollutant sub-index using the project's CPCB breakpoint table and takes the maximum as the same-day AQI target. This target is derived from the pollutant inputs; it is not an independently measured future outcome.

#### Q4: Why is it critical to check units during pollutant calculations?
**Answer:** The generated dataset stores PM2.5, PM10, NO2, SO2, and O3 in $\mu g/m^3$, and CO in $mg/m^3$. The optional DEAP-derived fallback has different source conventions; its unit conversions are not currently applied by the preprocessing path and must be validated before using it.

#### Q5: Why did you drop rows with missing AQI values instead of imputing them?
**Answer:** The checked-in generated dataset contains 51,000 rows with no missing AQI targets. The preprocessing code drops rows when a valid AQI cannot be calculated rather than imputing target values.

#### Q6: How did you handle missing values in features like PM10 (52% missing)?
**Answer:** We embedded a `SimpleImputer` using the `median` strategy inside a Scikit-Learn `Pipeline` (within a `ColumnTransformer`). This ensures that missing feature values are imputed automatically on-the-fly during training and inference.

#### Q7: What is data leakage, and how did you prevent it during preprocessing?
**Answer:** Scikit-Learn pipelines keep imputation, scaling, and encoding fitted on training rows only. The older model comparisons still use random row splits, however, so dates may overlap. The dedicated temporal evaluation groups complete dates and tests only on later dates.

#### Q8: How did you handle outliers in your dataset?
**Answer:** The synthetic dataset is bounded by the generator's pollutant ranges and is not an empirical sample of real-world outliers. Numeric preprocessing is kept inside the model pipeline; tree models split on feature thresholds, while scaling is primarily useful for scale-sensitive models such as Ridge and SVR.

---

### Section 2: Machine Learning Algorithms & Training

#### Q9: What train-test split strategy did you use and why?
**Answer:** The original comparison scripts use random row splits and are exploratory. `python src/evaluate_temporal.py` holds out the latest 20% of unique dates, keeping all records for a date together. For the current generated data, training ends on 2024-03-13 and testing starts on 2024-03-14. This is a later-date same-day AQI estimation test, not a future forecast.

#### Q10: Why did you include Ridge Regression as a baseline?
**Answer:** Ridge Regression serves as a baseline to evaluate linear relationships. Since Ridge penalizes large coefficients (L2 regularization), it stabilizes the linear weights against multicollinearity (high correlations between PM2.5 and PM10, for example).

#### Q11: Explain the core difference between a Decision Tree and a Random Forest.
**Answer:** A Decision Tree splits data recursively into subsets based on feature thresholds, which can easily overfit the training set. A Random Forest is an *ensemble* method that trains multiple independent decision trees on bootstrapped data samples and averages their predictions, which significantly reduces variance and prevents overfitting.

#### Q12: Why did Random Forest perform the best in your experiments?
**Answer:** The Random Forest approximates the CPCB-derived same-day target from pollutant and engineered inputs. On the chronological holdout it achieved MAE 0.6093, RMSE 5.9093, and $R^2$ 0.9986 on synthetic data. This result measures agreement with the generated rule-based target and does not establish real-world performance.

#### Q13: Why did Support Vector Regressor (SVR) perform poorly on this task?
**Answer:** SVR is one of the exploratory comparison models. Its reported score comes from the older random-row evaluation, so compare it only within that setup; the chronological evaluation currently compares the deployed Random Forest with a mean baseline.

#### Q14: Explain the difference between GridSearchCV and RandomizedSearchCV.
**Answer:** `GridSearchCV` performs an exhaustive search across every possible combination of specified parameters, which can be computationally slow. `RandomizedSearchCV` samples a fixed number of parameter settings randomly from specified distributions, which keeps computational overhead low while finding near-optimal parameters.

#### Q15: Why did you regularize your tuned Random Forest if it slightly lowered the R² score?
**Answer:** The deployed Random Forest uses `n_estimators=150`, `min_samples_split=2`, and `min_samples_leaf=1`. The temporal evaluation uses the same estimator settings but fits it only on the earlier-date partition. These settings should not be described as proof of performance on real user or sensor data.

---

### Section 3: Model Evaluation & Metrics

#### Q16: What do MAE, RMSE, and R² score measure?
**Answer:** 
- **MAE (Mean Absolute Error):** Average absolute error; the temporal holdout MAE is 0.6093 AQI points.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily; the temporal holdout RMSE is 5.9093.
- **$R^2$ (R-squared/Coefficient of Determination):** Measures explained target variance; the temporal holdout $R^2$ is 0.9986. These are results on generated synthetic data.

#### Q17: How did you compute classification metrics (like Accuracy and F1-score) for a regression model?
**Answer:** We did not train a classifier. Instead, we predicted continuous AQI values on the test set, mapped both true and predicted values to CPCB risk categories, and calculated standard classification metrics (accuracy, precision, recall) on the mapped categories.

#### Q18: What did your confusion matrix reveal about Ridge Regression?
**Answer:** The older random-split report includes a confusion matrix for Ridge Regression, but it should be interpreted as an exploratory comparison on the synthetic target. The newer chronological report focuses on the deployed Random Forest and a simple mean baseline.

---

### Section 4: Feature Importance & Interpretability

#### Q19: What does feature importance represent in tree-based models?
**Answer:** It measures how frequently a feature is selected for splits and how much it reduces impurity (variance) across all trees. It indicates *predictive association*, not a physical causal relationship.

#### Q20: Which features were most important, and does this align with environmental science?
**Answer:** The committed Random Forest's global importances are led by PM10 (about 46.9%), PM2.5 (15.5%), NO2 (13.5%), and `Gaseous_Index` (12.8%). Since the synthetic target is calculated from pollutant sub-indices, this ranking mainly reflects that target construction; it is not causal evidence or a per-input explanation.

#### Q21: What engineered features did you create and how did they perform?
**Answer:** We created:
- `PM25_PM10_ratio` to differentiate combustion-derived fine particles from mechanical dust.
- `PM25_Wind_Ratio` (PM2.5 / Wind Speed) to represent atmospheric stagnation.
Both are engineered features, but their small global importance values do not demonstrate that they improve generalization. Feature importance summarizes model use across training examples, not an individual prediction.

---

### Section 5: Model Deployment & Streamlit

#### Q22: What file format did you use to save the model, and what does it contain?
**Answer:** We used **joblib** to save a single serialized file: `artifacts/model.joblib`. It contains the entire Scikit-Learn `Pipeline` (the numeric imputer, scaler, categorical encoder, and the final fitted Random Forest estimator).

#### Q23: Why did you save the preprocessing steps together with the model estimator?
**Answer:** Saving them together guarantees that raw input entered by a user in the Streamlit app is cleaned, scaled, and encoded in the *exact same way* as the training data, preventing runtime exceptions and feature mismatch bugs.

#### Q24: How does the Streamlit app handle input validation?
**Answer:** The app uses bounded sliders for pollutant and weather inputs and warns when PM10 is lower than PM2.5. It does not accept negative concentrations or humidity above 100% through those sliders.

#### Q25: How are calendar context features handled in Streamlit?
**Answer:** The app features a date picker. When a user selects a date, the app automatically extracts the `Month`, `DayOfWeek`, and `Season` fields, passing them to the pipeline in the background.

#### Q26: What are the main limitations of your deployed system?
**Answer:** 
1. **Extrapolation Limit:** Being an ensemble of trees, Random Forest cannot extrapolate predictions outside the range of targets seen in training (capped at 500).
2. **Tabular Only:** The model predicts based on current parameters and does not capture chronological lag or spatial shifts.

#### Q27: Is this system a forecast of future air quality?
**Answer:** No. It estimates same-day AQI from user-entered pollutant and weather values. The observation date supplies calendar features, but the model has no historical lag inputs or future target.
