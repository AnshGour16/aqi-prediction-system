# Viva Preparation Guide — Air Quality Prediction System

This guide contains 25+ likely viva/presentation questions with concise, technically rigorous, and beginner-readable answers to prepare for your project demonstration.

---

### Section 1: Dataset & Preprocessing

#### Q1: Why is Air Quality Index (AQI) prediction modeled as a regression problem?
**Answer:** AQI is a continuous numerical value representing pollution concentration scales. Since our goal is to predict the exact numerical index (e.g. 156.4) from multiple continuous environmental inputs, it is a regression problem. Mapping this continuous prediction to a discrete category (like 'Moderate') is a post-processing step, not a classification model.

#### Q2: How did you select your dataset, and what variables are included?
**Answer:** We selected the public **DEAP** dataset. It tracks daily air quality across major cities and contains key pollutants (PM2.5, PM10, NO2, SO2, CO, O3) alongside meteorological parameters (temperature, relative humidity, wind speed).

#### Q3: Your raw dataset did not contain AQI. How did you handle this?
**Answer:** We calculated AQI from raw pollutant concentrations using the **Indian Central Pollution Control Board (CPCB)** guidelines. We computed individual sub-indices for each pollutant using official linear interpolation formulas and took the maximum sub-index as the final daily AQI.

#### Q4: Why is it critical to check units during pollutant calculations?
**Answer:** Breakpoint standards assume specific units. The CPCB standard uses $\mu g/m^3$ for particulate matter and most gases, but uses $mg/m^3$ for Carbon Monoxide (CO). The raw dataset measured gases (NO2, SO2, O3, CO) in parts-per-billion (ppb) or parts-per-million (ppm). We applied conversion factors (e.g., $1\text{ ppm CO} = 1.15\text{ mg/m}^3$ at $25^\circ C$) to ensure calculation accuracy.

#### Q5: Why did you drop rows with missing AQI values instead of imputing them?
**Answer:** Under ML best practices, **we never impute the target variable ($y$)**. Imputing the target fabricates ground truth, introducing synthetic bias and invalidating model evaluation. If a row lacked enough pollutants to calculate a valid CPCB AQI, we dropped it. We retained 27,938 clean rows, which is more than sufficient.

#### Q6: How did you handle missing values in features like PM10 (52% missing)?
**Answer:** We embedded a `SimpleImputer` using the `median` strategy inside a Scikit-Learn `Pipeline` (within a `ColumnTransformer`). This ensures that missing feature values are imputed automatically on-the-fly during training and inference.

#### Q7: What is data leakage, and how did you prevent it during preprocessing?
**Answer:** Data leakage occurs when information from the test set is accidentally used to fit training parameters (like scaling means or imputation medians), leading to over-optimistic test scores. We prevented this by wrapping all preprocessing steps (scaling, encoding, imputation) inside a Scikit-Learn `Pipeline` rather than applying them globally to the pandas DataFrame.

#### Q8: How did you handle outliers in your dataset?
**Answer:** We performed an IQR (Interquartile Range) outlier analysis showing extreme right-skewed tails (e.g. PM2.5 max = 507). Linear baseline models and SVR are highly sensitive to outliers, so we applied feature scaling (`StandardScaler`). Tree-based models (Random Forest, Decision Tree) are naturally robust to outliers since they partition data with binary threshold splits.

---

### Section 2: Machine Learning Algorithms & Training

#### Q9: What train-test split strategy did you use and why?
**Answer:** We used an **80% training and 20% testing split** using `random_state=42`. Since we are mapping instantaneous, daily environmental metrics directly to daily AQI (no historical lag inputs), a random split is mathematically sound and ensures both splits contain a representative distribution of seasons and cities.

#### Q10: Why did you include Ridge Regression as a baseline?
**Answer:** Ridge Regression serves as a baseline to evaluate linear relationships. Since Ridge penalizes large coefficients (L2 regularization), it stabilizes the linear weights against multicollinearity (high correlations between PM2.5 and PM10, for example).

#### Q11: Explain the core difference between a Decision Tree and a Random Forest.
**Answer:** A Decision Tree splits data recursively into subsets based on feature thresholds, which can easily overfit the training set. A Random Forest is an *ensemble* method that trains multiple independent decision trees on bootstrapped data samples and averages their predictions, which significantly reduces variance and prevents overfitting.

#### Q12: Why did Random Forest perform the best in your experiments?
**Answer:** AQI calculations are threshold-bound step functions (e.g. CPCB bands). Decision splits in Random Forest naturally partition the feature space into rectangular threshold bins. Additionally, by aggregating 150 trees, Random Forest smoothed out boundary transitions, achieving an $R^2$ score of **89.61%**.

#### Q13: Why did Support Vector Regressor (SVR) perform poorly on this task?
**Answer:** SVR tries to fit a smooth continuous hyperplane within a narrow epsilon tube. Since AQI step functions contain sharp threshold boundaries, forcing a smooth mathematical surface results in a poor fit ($R^2 = 38.60\%$) and high computation times due to quadratic kernel complexity.

#### Q14: Explain the difference between GridSearchCV and RandomizedSearchCV.
**Answer:** `GridSearchCV` performs an exhaustive search across every possible combination of specified parameters, which can be computationally slow. `RandomizedSearchCV` samples a fixed number of parameter settings randomly from specified distributions, which keeps computational overhead low while finding near-optimal parameters.

#### Q15: Why did you regularize your tuned Random Forest if it slightly lowered the R² score?
**Answer:** The tuned Random Forest (which used `min_samples_leaf=2` and `min_samples_split=5`) scored 89.61% compared to the default model's 89.92%. We preferred the regularized model because restricting leaf size prevents trees from memorizing noise, resulting in better generalization on unseen user inputs.

---

### Section 3: Model Evaluation & Metrics

#### Q16: What do MAE, RMSE, and R² score measure?
**Answer:** 
- **MAE (Mean Absolute Error):** Measures the average magnitude of absolute errors (easy to interpret; our model deviates by ~3.7 AQI units on average).
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily (useful for catching large prediction mistakes; our model scored ~16.4).
- **$R^2$ (R-squared/Coefficient of Determination):** Measures the proportion of target variance explained by features (our model explains 89.61% of AQI variation).

#### Q17: How did you compute classification metrics (like Accuracy and F1-score) for a regression model?
**Answer:** We did not train a classifier. Instead, we predicted continuous AQI values on the test set, mapped both true and predicted values to CPCB risk categories, and calculated standard classification metrics (accuracy, precision, recall) on the mapped categories.

#### Q18: What did your confusion matrix reveal about Ridge Regression?
**Answer:** Ridge Regression had difficulty predicting extreme conditions: on true "Good" days (AQI 0-50), it predicted "Satisfactory" or "Moderate" because linear regressors predict near the mean of the training data. This highlighted the necessity of non-linear tree split boundaries.

---

### Section 4: Feature Importance & Interpretability

#### Q19: What does feature importance represent in tree-based models?
**Answer:** It measures how frequently a feature is selected for splits and how much it reduces impurity (variance) across all trees. It indicates *predictive association*, not a physical causal relationship.

#### Q20: Which features were most important, and does this align with environmental science?
**Answer:** Carbon Monoxide (`co_median`) had the highest importance (68.53%), followed by Nitrogen Dioxide (`no2_median`, 14.38%) and PM2.5 (6.70%). This is highly consistent, as vehicles and combustion processes emit CO and NO2 alongside fine particles, making them highly reliable tracers of urban smog.

#### Q21: What engineered features did you create and how did they perform?
**Answer:** We created:
- `PM25_PM10_ratio` to differentiate combustion-derived fine particles from mechanical dust.
- `PM25_Wind_Ratio` (PM2.5 / Wind Speed) to represent atmospheric stagnation.
Both ranked higher in predictive importance than raw weather variables, validating that modeling physical interactions improves model performance.

---

### Section 5: Model Deployment & Streamlit

#### Q22: What file format did you use to save the model, and what does it contain?
**Answer:** We used **joblib** to save a single serialized file: `artifacts/model.joblib`. It contains the entire Scikit-Learn `Pipeline` (the numeric imputer, scaler, categorical encoder, and the final fitted Random Forest estimator).

#### Q23: Why did you save the preprocessing steps together with the model estimator?
**Answer:** Saving them together guarantees that raw input entered by a user in the Streamlit app is cleaned, scaled, and encoded in the *exact same way* as the training data, preventing runtime exceptions and feature mismatch bugs.

#### Q24: How does the Streamlit app handle input validation?
**Answer:** The app uses numeric input forms with specified boundary limits. If a user inputs negative values or impossible humidity ranges (e.g. >100%), the application blocks predictions and displays a red warning banner.

#### Q25: How are calendar context features handled in Streamlit?
**Answer:** The app features a date picker. When a user selects a date, the app automatically extracts the `Month`, `DayOfWeek`, and `Season` fields, passing them to the pipeline in the background.

#### Q26: What are the main limitations of your deployed system?
**Answer:** 
1. **Extrapolation Limit:** Being an ensemble of trees, Random Forest cannot extrapolate predictions outside the range of targets seen in training (capped at 500).
2. **Tabular Only:** The model predicts based on current parameters and does not capture chronological lag or spatial shifts.
