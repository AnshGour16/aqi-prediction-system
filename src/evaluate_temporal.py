import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

if __package__:
    from .save_model import get_model_features, get_preprocessing_pipeline
else:
    from save_model import get_model_features, get_preprocessing_pipeline


def split_by_date(df, test_fraction=0.2):
    if "Date" not in df.columns:
        raise ValueError("Dataset must include a Date column for chronological evaluation.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("Date column contains missing or invalid dates.")

    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    if len(unique_dates) < 2:
        raise ValueError("At least two unique dates are required for chronological evaluation.")

    split_index = int(len(unique_dates) * (1.0 - test_fraction))
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    cutoff = unique_dates[split_index]
    train_mask = dates < cutoff

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty training or test set.")
    return train_df, test_df, cutoff


def map_aqi_to_category(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def evaluate_predictions(model_name, y_true, y_pred):
    bounded_predictions = np.clip(y_pred, 0.0, 500.0)
    true_categories = pd.Series(y_true).apply(map_aqi_to_category)
    predicted_categories = pd.Series(bounded_predictions).apply(map_aqi_to_category)
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, bounded_predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, bounded_predictions)),
        "R2": r2_score(y_true, bounded_predictions),
        "Category Accuracy": accuracy_score(true_categories, predicted_categories)
    }


def run_temporal_evaluation():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "data", "processed", "engineered_air_quality.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if "AQI" not in df.columns:
        raise ValueError("Engineered dataset must include the AQI target column.")
    train_df, test_df, cutoff = split_by_date(df)
    numeric_features, categorical_features = get_model_features(df)
    feature_columns = numeric_features + categorical_features

    y_train = pd.to_numeric(train_df["AQI"], errors="coerce")
    y_test = pd.to_numeric(test_df["AQI"], errors="coerce")
    if not np.isfinite(y_train.to_numpy()).all() or not np.isfinite(y_test.to_numpy()).all():
        raise ValueError("AQI target contains missing or non-finite values.")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    models = [
        (
            "Mean baseline",
            DummyRegressor(strategy="mean")
        ),
        (
            "Random Forest (chronological holdout)",
            Pipeline(steps=[
                ("preprocessor", get_preprocessing_pipeline(
                    numeric_features, categorical_features
                )),
                ("regressor", RandomForestRegressor(
                    n_estimators=150,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        )
    ]

    results = []
    for model_name, estimator in models:
        print(f"[TEMPORAL EVALUATION] Fitting {model_name}...")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        results.append(evaluate_predictions(model_name, y_test, predictions))

    results_df = pd.DataFrame(results)
    results_df.insert(1, "Training Rows", len(train_df))
    results_df.insert(2, "Test Rows", len(test_df))
    results_df.insert(3, "Training Through", pd.to_datetime(train_df["Date"]).max().date())
    results_df.insert(4, "Test From", cutoff.date())

    source_path = os.path.join(project_dir, "data", "raw", "synthetic_air_quality_data.csv")
    if os.path.exists(source_path):
        source_description = (
            "`synthetic_air_quality_data.csv` (the preprocessing pipeline prioritizes this "
            "generated dataset when it is present)"
        )
    else:
        source_description = (
            "`city_pollution_data.csv` (the public DEAP-derived fallback used when the synthetic "
            "dataset is absent)"
        )

    report_lines = [
        "# Chronological AQI Evaluation",
        "",
        "## Evaluation design",
        f"- **Configured source:** {source_description}.",
        f"- **Rows evaluated:** {len(df):,}.",
        f"- **Training rows:** {len(train_df):,} through "
        f"{pd.to_datetime(train_df['Date']).max().date()}.",
        f"- **Test rows:** {len(test_df):,} from {cutoff.date()} onward.",
        f"- **Date groups:** {pd.to_datetime(train_df['Date']).nunique():,} training dates and "
        f"{pd.to_datetime(test_df['Date']).nunique():,} test dates; no date appears in both sets.",
        "- **Target meaning:** same-day AQI derived from pollutant sub-indices. This evaluates "
        "AQI estimation on later dates; it is not a forecast of future air quality.",
        "- **Synthetic-data caveat:** the default generated dataset is simulated, not measured "
        "sensor data. Strong performance on it does not establish real-world accuracy.",
        "",
        "## Results",
        results_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "MAE and RMSE are measured in AQI points. Category accuracy compares the derived CPCB "
        "bands after clipping estimates to the 0–500 range.",
        ""
    ]

    reports_dir = os.path.join(project_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "temporal_evaluation_report.md")
    csv_path = os.path.join(reports_dir, "temporal_evaluation.csv")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))
    results_df.to_csv(csv_path, index=False)

    print(f"[TEMPORAL EVALUATION] Report: {report_path}")
    print(f"[TEMPORAL EVALUATION] Results: {csv_path}")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run_temporal_evaluation()
