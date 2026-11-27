# Air Quality Prediction System — Predicting AQI & Environmental Conditions Using Machine Learning

An academic AI/ML project designed to predict continuous **Air Quality Index (AQI)** values and map environmental conditions to health risk categories using classical Machine Learning regression algorithms.

## 1. Project Problem Statement
Air quality fluctuates dynamically based on multiple interacting environmental factors (pollutants like $PM_{2.5}$, $PM_{10}$, $NO_2$, $SO_2$, $CO$, and weather conditions like temperature, humidity, and wind speed). Static statistical estimations fail to capture the non-linear relationships across these features. This project develops a machine learning regression system that:
1. Accepts environmental and pollutant concentrations.
2. Validates inputs against physical boundaries.
3. Predicts a continuous numerical AQI value.
4. Categorizes the AQI into CPCB health bands (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
5. Updates results immediately in an interactive Streamlit web application; it does not fetch live sensor readings.

---

## 2. Project Objectives
- **AQI Model Development:** Train a regressor to predict continuous AQI from tabular daily data.
- **Environmental Feature Analysis:** Understand correlations and feature importance of environmental factors.
- **Comparative Evaluation:** Compare classical ML regression models and evaluate the deployed Random Forest on later dates.
- **Interactive UI Integration:** Explore scenario presets and live pollutant/weather inputs, compare the model estimate with the CPCB calculation, view charts and health advisories, and export a result as JSON.

---

## 3. Dataset & Feature Mappings
The current processed dataset and committed model use `data/raw/synthetic_air_quality_data.csv`, a locally generated dataset of 51,000 simulated records. `src/data_preprocessing.py` prioritizes this file whenever it exists. `src/generate_dataset.py` creates the pollutant and weather values and derives each AQI target from CPCB pollutant sub-indices. Each row is anchored by one pollutant sampled inside the target category, while the remaining pollutants vary from their Good-range minimum up to the same category ceiling; this produces realistic mixed-pollutant rows instead of moving every pollutant up and down together. These are **not observed sensor readings**, and they do not represent actual cities.

The repository also contains `data/raw/city_pollution_data.csv`, a public DEAP-derived dataset downloaded by `src/fetch_data.py`. It is only selected by preprocessing when the synthetic file is absent. The fallback path has not been used for the committed processed data, and its source-specific pollutant unit conversions should be validated before using it for training.

For the synthetic dataset, pollutant fields use CPCB concentration units:
- **PM2.5 (`pm25_median`)** and **PM10 (`pm10_median`):** particulate concentrations in $\mu g/m^3$
- **NO2 (`no2_median`), SO2 (`so2_median`), and O3 (`o3_median`):** gas concentrations in $\mu g/m^3$
- **CO (`co_median`):** carbon monoxide in $mg/m^3$
- **Temperature (`temperature_median`):** $^\circ C$; **humidity (`humidity_median`):** $\%$; **wind speed (`wind-speed_median`):** $m/s$

---

## 4. Methodology Pipeline
```
Dataset Selection -> Data Collection -> Preprocessing & Cleaning -> Feature Engineering 
-> Exploratory Random-Split Comparisons -> Date-Grouped Holdout Evaluation
-> Model Serialization -> Streamlit App Deployment
```

1. **Data Preprocessing:** Handled missing values (imputed using median values in pipeline), duplicates (removed), invalid values (negative values replaced with NaN), and verified target calculations.
2. **AQI Calculation standard:** Mapped target values using official **Indian CPCB standards** based on sub-index breakpoints:
   - Good (0–50) | Satisfactory (51–100) | Moderate (101–200) | Poor (201–300) | Very Poor (301–400) | Severe (401–500)
   - The breakpoint tables are **contiguous** (no 0.1-wide holes between bands), so a concentration cannot fall through to the 500 cap and be mis-scored. The same `src/aqi_utils.py` module is used by the pipeline and the Streamlit app.
   - Every `Category` label is derived from its numeric `AQI` value, so the two can never disagree.
3. **Feature Engineering:** Calendar profiles (`Month`, `DayOfWeek`, `Season`) and physical interactions (`PM25_PM10_ratio`, `PM25_Wind_Ratio`, `Stagnation_Index`, `Gaseous_Index`, `Thermal_Humidity_Index`) were engineered. `src/feature_builder.py` constructs the same 21-feature row for app inference and tests.
4. **Evaluation:** The original model comparison scripts use random row splits and are exploratory. `python src/evaluate_temporal.py` instead holds out complete dates: it trains on records through 2024-03-13 and evaluates records from 2024-03-14 onward for the current generated dataset.

---

## 5. Chronological Model Evaluation
A matching Random Forest pipeline was fitted on the earlier dates and evaluated on a later-date holdout. All records from a date stay in the same split. The test target is same-day AQI derived from pollutant values, so this measures later-date **AQI estimation**, not forecasting future conditions.

| Model | MAE (AQI points) | RMSE | $R^2$ | CPCB category accuracy |
| :--- | ---: | ---: | ---: | ---: |
| **Random Forest (chronological holdout)** | **0.9376** | **3.2872** | **0.9995** | **99.22%** |
| Mean baseline | 134.9799 | 151.6378 | approximately 0.0000 | 16.28% |

These metrics are measured on the generated synthetic dataset and do not establish real-world performance. Earlier multi-model scores in `reports/raw_model_comparison.csv`, `reports/model_comparison.csv`, and `reports/tuned_model_comparison.csv` come from random row splits; treat them as exploratory, not as evidence of generalization to future dates or cities. The standalone `Air_Quality_Prediction_Presentation_Guide.pdf` predates the latest dataset regeneration, so its metric figures are superseded by the table above and `reports/temporal_evaluation.csv`.

---

## 6. Champion Model & Serialization
The deployed **Random Forest Regressor** (`n_estimators=150`, `min_samples_split=2`, `min_samples_leaf=1`) is serialized with its preprocessing pipeline in `artifacts/model.joblib`. The committed artifact is trained on the generated synthetic dataset. The chronological evaluation fits a separate copy using only its training-date partition; it does not evaluate the all-data production artifact. The app builds its 21-feature inference row with `src/feature_builder.py`, the same contract used by the pipeline and tests.

---

## 7. Project Structure
```
air-quality-prediction/
│
├── data/
│   ├── raw/                 # Unmodified raw CSV dataset
│   └── processed/           # Processed & engineered datasets
│
├── notebooks/
│   └── eda_modeling.ipynb  # Interactive analysis notebook
│
├── src/
│   ├── fetch_data.py        # Dataset downloader
│   ├── dataset_audit.py     # Dataset profile & integrity checks
│   ├── aqi_utils.py         # Shared CPCB breakpoints, sub-indices & categories
│   ├── feature_builder.py   # Shared 21-feature inference row builder
│   ├── data_preprocessing.py # CPCB calculations & data cleaning
│   ├── feature_engineering.py# calendar & ratio transformations
│   ├── train.py             # Baseline model training
│   ├── train_all.py         # Six raw model pipeline
│   ├── tune.py              # Hyperparameter optimizer
│   ├── evaluate.py          # Continuous & category-level verification
│   ├── evaluate_temporal.py # Date-grouped holdout evaluation
│   ├── test_temporal_evaluation.py # Date split regression tests
│   ├── test_aqi_utils.py    # CPCB maths regression tests
│   ├── select_model.py      # Feature importance rankings
│   ├── save_model.py        # Final retrain and serialization script
│   └── test_inference.py    # Edge-cases test scenarios
│
├── artifacts/
│   └── model.joblib         # Serialized final pipeline
│
├── app/
│   └── app.py               # Streamlit application
│
├── reports/
│   ├── figures/             # Saved visual plots (PNG format)
│   ├── dataset_audit_report.md
│   ├── evaluation_report.md
│   ├── temporal_evaluation_report.md
│   ├── temporal_evaluation.csv
│   ├── model_interpretability.md
│   └── model_comparison.csv
│
├── requirements.txt         # Dependencies list
└── README.md
```

---

## 8. Installation & Execution

### Prerequisites
- Python 3.10 to 3.12 installed on your system.
- Git

### 🚀 Quick Start (Run in 2 Steps)

```bash
# 1. Clone the repository & install dependencies
git clone https://github.com/AnshGour16/aqi-prediction-system.git
cd aqi-prediction-system
pip install -r requirements.txt

# 2. Launch the Streamlit Web Application
python -m streamlit run app/app.py
```
*(The app automatically builds and verifies the cryptographic champion model on first launch if not already present, and opens in your browser at `http://localhost:8501`)*

In the app, choose a scenario or adjust the sliders to update the results. The main AQI can show either the machine-learning estimate or the CPCB calculation, and the current result can be downloaded as JSON. This is an input-driven estimate, not a live sensor dashboard or a substitute for official local air-quality guidance.

---

### 🔬 Full Pipeline Execution (Optional — Re-train from Scratch)

If you wish to re-download the dataset, re-run exploratory data analysis, evaluate all six model families, and re-tune hyperparameters, note that preprocessing still prioritizes the generated synthetic file if it is present. The DEAP-derived fallback is not unit-normalized by the current preprocessing code; validate its units before using it.

```bash
# Step 1: Download & Preprocess Data
python src/fetch_data.py
python src/data_preprocessing.py
python src/feature_engineering.py

# Step 2: Train, Tune & Evaluate All Six Model Families
python src/train_all.py
python src/tune.py
python src/evaluate.py
python src/select_model.py
python src/save_model.py
python src/evaluate_temporal.py

# Step 3: Run Verification Tests
python src/test_inference.py
python -m unittest src.test_temporal_evaluation src.test_aqi_utils
```

## 9. Model Limitations & Future Enhancements
- **Limitations:** The current model learns to estimate a same-day AQI target derived from the same pollutant inputs; it is not a future forecast. The committed data is synthetic, so the results do not establish real-world accuracy. Tree models cannot extrapolate outside training limits (capped at 500 AQI). Global feature importance is not a causal or per-scenario explanation. The app does not display live sensor readings.
- **Future Scope:** Integrating real-time IoT air pollution sensor networks, and leveraging LSTM/RNN models for sequential forecasting.

---

## 10. Team Responsibilities
- **Ansh Gour (2410990264):** Lead ML System Architect. Implemented preprocessing pipelines, CPCB equations, and pipeline serialization.
- **Neeti Sharma (2410990149):** Data Engineer. Setup data downloading and cleaning logic.
- **Zeeya Singh (2410990404):** Frontend Developer. Created the Streamlit interface, custom color mappings, and public health advisories.
- **Kanshika (2410990372):** QA & Evaluation Specialist. Implemented detailed model evaluation scripts, cross-validation tuning, and edge-case testing.
