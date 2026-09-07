# Air Quality Prediction System — Predicting AQI & Environmental Conditions Using Machine Learning

An academic AI/ML project designed to predict continuous **Air Quality Index (AQI)** values and map environmental conditions to health risk categories using classical Machine Learning regression algorithms.

## 1. Project Problem Statement
Air quality fluctuates dynamically based on multiple interacting environmental factors (pollutants like $PM_{2.5}$, $PM_{10}$, $NO_2$, $SO_2$, $CO$, and weather conditions like temperature, humidity, and wind speed). Static statistical estimations fail to capture the non-linear relationships across these features. This project develops a machine learning regression system that:
1. Accepts environmental and pollutant concentrations.
2. Validates inputs against physical boundaries.
3. Predicts a continuous numerical AQI value.
4. Categorizes the AQI into CPCB health bands (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
5. Displays results in real-time through an interactive Streamlit web application.

---

## 2. Project Objectives
- **AQI Model Development:** Train a regressor to predict continuous AQI from tabular daily data.
- **Environmental Feature Analysis:** Understand correlations and feature importance of environmental factors.
- **Comparative Evaluation:** Compare 5 classical ML regression models (Ridge, Decision Tree, Random Forest, KNN, SVR).
- **Interactive UI Integration:** Deploy the champion model in a clean Streamlit interface with color-coded risk alerts and public advisories.

---

## 3. Dataset & Feature Mappings
We utilize the **DEAP** (Deciphering Environmental Air Pollution) public dataset, which contains daily measurements across major cities:
- **PM2.5 (`pm25_median`):** Fine Particulate Matter ($\mu g/m^3$)
- **PM10 (`pm10_median`):** Coarse Particulate Matter ($\mu g/m^3$)
- **NO2 (`no2_median`):** Nitrogen Dioxide (converted from ppb to $\mu g/m^3$ via $\times 1.88$)
- **SO2 (`so2_median`):** Sulfur Dioxide (converted from ppb to $\mu g/m^3$ via $\times 2.62$)
- **CO (`co_median`):** Carbon Monoxide (converted from ppm to $mg/m^3$ via $\times 1.15$)
- **O3 (`o3_median`):** Ozone (converted from ppb to $\mu g/m^3$ via $\times 1.96$)
- **Temperature (`temperature_median`):** Relative temperature in $^\circ C$
- **Humidity (`humidity_median`):** Relative humidity in $\%$
- **Wind Speed (`wind-speed_median`):** Average wind velocity in $m/s$

---

## 4. Methodology Pipeline
```
Dataset Selection -> Data Collection -> Preprocessing & Cleaning -> Feature Engineering 
-> Train-Test Split -> Preprocessing Pipeline -> Model Training -> Hyperparameter Tuning 
-> Model Evaluation & Comparison -> Model Serialization -> Streamlit App Deployment
```

1. **Data Preprocessing:** Handled missing values (imputed using median values in pipeline), duplicates (removed), invalid values (negative values replaced with NaN), and verified target calculations.
2. **AQI Calculation standard:** Mapped target values using official **Indian CPCB standards** based on sub-index breakpoints:
   - Good (0–50) | Satisfactory (51–100) | Moderate (101–200) | Poor (201–300) | Very Poor (301–400) | Severe (>400)
3. **Feature Engineering:** Calendar profiles (`Month`, `DayOfWeek`, `Season`) and physical interactions (`PM25_PM10_ratio`, `PM25_Wind_Ratio`) were engineered.
4. **Train-Test Split:** 80% training set (22,350 rows), 20% test set (5,588 rows) with `random_state=42`.

---

## 5. Model Evaluation & Comparison
The tuned models were evaluated on regression metrics (MAE, RMSE, $R^2$) and derived classification accuracy (after mapping continuous AQI to CPCB bands):

| Model | Regression MAE | Regression RMSE | Regression $R^2$ | Categorical Accuracy | Weighted F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **0.7202** | **8.5120** | **0.9971 (99.71%)** | **98.88%** | **0.9888** | 🏆 Champion |
| **Decision Tree (Tuned)** | 0.7674 | 9.5833 | 0.9963 (99.63%) | 99.83% | 0.9983 | High Precision |
| **Gradient Boosting (Tuned)**| 2.5993 | 8.0106 | 0.9974 (99.74%) | 96.31% | 0.9633 | High Generalization |
| **Support Vector Regressor (Tuned)** | 8.3841 | 15.3334 | 0.9906 (99.06%) | 82.07% | 0.8205 | Strong Non-linear |
| **K-Nearest Neighbors (Tuned)** | 13.3141 | 21.0425 | 0.9823 (98.23%) | 75.93% | 0.7579 | Distance-weighted |
| **Ridge Regression (Tuned)** | 18.4646 | 25.7412 | 0.9735 (97.35%) | 68.82% | 0.6810 | Baseline |

---

## 6. Champion Model & Serialization
The **Tuned Random Forest Regressor** (`n_estimators=150`, `min_samples_split=5`, `min_samples_leaf=2`) was selected as our champion model. The complete preprocessing + estimator pipeline was trained on 100% of the clean dataset and serialized to `artifacts/model.joblib`.

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
│   ├── dataset_audit.py     # Initial profile & correlation checks
│   ├── data_preprocessing.py # CPCB calculations & data cleaning
│   ├── feature_engineering.py# calendar & ratio transformations
│   ├── train.py             # Baseline model training
│   ├── train_all.py         # 5 raw models pipeline
│   ├── tune.py              # Hyperparameter optimizer
│   ├── evaluate.py          # Continuous & category-level verification
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
│   ├── model_interpretability.md
│   └── model_comparison.csv
│
├── requirements.txt         # Dependencies list
└── README.md
```

---

## 8. Installation & Execution

### Prerequisites
- Python 3.8 to 3.12 installed on your system.
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

---

### 🔬 Full Pipeline Execution (Optional — Re-train from Scratch)

If you wish to re-download the dataset, re-run exploratory data analysis, evaluate all 5 algorithms, and re-tune hyperparameters:

```bash
# Step 1: Download & Preprocess Data
python src/fetch_data.py
python src/data_preprocessing.py
python src/feature_engineering.py

# Step 2: Train, Tune & Evaluate All 5 Models
python src/train_all.py
python src/tune.py
python src/evaluate.py
python src/select_model.py
python src/save_model.py

# Step 3: Run Verification Tests
python src/test_inference.py
```

## 9. Model Limitations & Future Enhancements
- **Limitations:** Tree models cannot extrapolate outside training limits (capped at 500 AQI). Feature importance denotes mathematical association rather than physical causality.
- **Future Scope:** Integrating real-time IoT air pollution sensor networks, and leveraging LSTM/RNN models for sequential forecasting.

---

## 10. Team Responsibilities
- **Ansh Gour (2410990264):** Lead ML System Architect. Implemented preprocessing pipelines, CPCB equations, and pipeline serialization.
- **Neeti Sharma (2410990149):** Data Engineer. Setup data downloading, cleaning logic, and unit conversions.
- **Zeeya Singh (2410990404):** Frontend Developer. Created the Streamlit interface, custom color mappings, and public health advisories.
- **Kanshika (2410990372):** QA & Evaluation Specialist. Implemented detailed model evaluation scripts, cross-validation tuning, and edge-case testing.
