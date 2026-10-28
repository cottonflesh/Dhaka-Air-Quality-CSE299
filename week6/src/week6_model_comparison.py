"""
Week 6: Next-hour PM2.5 model comparison and error analysis
Project: Dhaka Air Quality (CSE299)

This Week 6 script continues the Week 5 PM2.5 work, but changes the task into a
true time-series forecasting problem:

    data available at hour t  ->  predict PM2.5 at hour t+1

Faculty-aligned design
----------------------
1. Keep the split chronological; never randomly split future rows into training.
2. Use current/recent PM2.5 history as forecasting features.
3. Use an appropriate persistence baseline: next hour PM2.5 = current PM2.5.
4. Compare several ML models on a future validation period.
5. Select the best ML model using validation RMSE, then test on the newest period.
6. Save detailed test-set error analysis and a reusable fitted model.

Run from the repository root:
    python week6/src/week6_model_comparison.py

Optional:
    python week6/src/week6_model_comparison.py --max-train-rows 0
"""

from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "Dhaka_CAMS_Cleaned_2012_2024.csv"
OUTPUT_DIR = ROOT / "week6" / "outputs"
MODEL_DIR = ROOT / "week6" / "models"

TARGET_CANDIDATES = ["PM2.5", "PM2_5", "PM25", "PM2.5 (ug/m3)", "PM2.5(µg/m³)"]
LOCATION_CANDIDATES = ["Location", "Station", "Site"]

# Canonical feature name -> possible names in project CSVs.
CURRENT_FEATURE_ALIASES = {
    "PM10": ["PM10"],
    "Temperature": ["Temperature", "Temp"],
    "Humidity": ["Humidity", "RH", "RelativeHumidity"],
    "WindSpeed": ["WindSpeed", "Wind Speed"],
    "WindDirection": ["WindDirection", "WindDir", "Wind Dir", "Wind Direction"],
    "CO": ["CO"],
    "NO": ["NO"],
    "NO2": ["NO2", "NO₂"],
    "NOX": ["NOX"],
    "SO2": ["SO2", "SO₂"],
    "O3": ["O3", "O₃"],
    "Pressure": ["Pressure", "BP"],
    "SolarRadiation": ["SolarRadiation", "SolarRad", "Solar Rad"],
    "Rain": ["Rain", "Rainfall"],
}

PM25_LAG_HOURS = [1, 3, 6, 24]


def find_column(columns, candidates):
    """Find a column using case-insensitive exact matching."""
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def parse_datetime(frame):
    """Parse timestamp formats used by the cleaned project datasets."""
    dt_col = find_column(frame.columns, ["DateTime", "Datetime", "Timestamp", "Date Time"])
    if dt_col is not None:
        return pd.to_datetime(frame[dt_col], errors="coerce")

    date_col = find_column(frame.columns, ["Date", "date"])
    time_col = find_column(frame.columns, ["Time", "time"])
    if date_col is not None and time_col is not None:
        return pd.to_datetime(
            frame[date_col].astype(str).str.strip()
            + " "
            + frame[time_col].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        )
    if date_col is not None:
        return pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)

    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def prepare_forecasting_frame(raw):
    """
    Build one row per station-hour using only information available at hour t.

    Target: PM2.5 exactly one hour later at the same station.
    Lag features are joined by exact timestamp, so a missing hour is not silently
    treated as a valid lag.
    """
    target_col = find_column(raw.columns, TARGET_CANDIDATES)
    if target_col is None:
        raise ValueError(f"Could not find a PM2.5 column. Columns found: {list(raw.columns)}")

    location_col = find_column(raw.columns, LOCATION_CANDIDATES)
    if location_col is None:
        raise ValueError("Could not find a Location/Station/Site column.")

    dt = parse_datetime(raw)

    frame = pd.DataFrame({
        "Location": raw[location_col].astype("string").fillna("Unknown").astype(str).str.strip(),
        "DateTime": dt,
        "PM25_Current": pd.to_numeric(raw[target_col], errors="coerce"),
    })

    for canonical, aliases in CURRENT_FEATURE_ALIASES.items():
        col = find_column(raw.columns, aliases)
        if col is not None:
            frame[canonical] = pd.to_numeric(raw[col], errors="coerce")

    # Remove unusable timestamp/location rows and duplicate station-hours.
    frame = frame.dropna(subset=["DateTime"])
    frame = frame[frame["Location"].ne("")]
    frame = frame.sort_values(["Location", "DateTime"]).drop_duplicates(
        subset=["Location", "DateTime"], keep="last"
    )

    # Negative PM2.5 values are not meaningful concentrations.
    frame.loc[frame["PM25_Current"] < 0, "PM25_Current"] = np.nan

    # Exact historical PM2.5 lags. Shift each source timestamp forward so it joins
    # only when an observation really exists h hours earlier at the same station.
    pm_lookup = frame[["Location", "DateTime", "PM25_Current"]].copy()
    for hours in PM25_LAG_HOURS:
        lag = pm_lookup.copy()
        lag["DateTime"] = lag["DateTime"] + pd.Timedelta(hours=hours)
        lag = lag.rename(columns={"PM25_Current": f"PM25_Lag_{hours}h"})
        frame = frame.merge(lag, on=["Location", "DateTime"], how="left", validate="one_to_one")

    # Exact next-hour PM2.5 target. Move the future observation back one hour so
    # it joins the row representing the information available at time t.
    future = pm_lookup.copy()
    future["DateTime"] = future["DateTime"] - pd.Timedelta(hours=1)
    future = future.rename(columns={"PM25_Current": "Target_PM25_NextHour"})
    frame = frame.merge(future, on=["Location", "DateTime"], how="left", validate="one_to_one")

    # Calendar information known at prediction time.
    frame["Month"] = frame["DateTime"].dt.month
    frame["Day"] = frame["DateTime"].dt.day
    frame["Hour"] = frame["DateTime"].dt.hour
    frame["DayOfWeek"] = frame["DateTime"].dt.dayofweek

    # Cyclical encodings help tree-independent interpretation and preserve the
    # fact that hour 23 is close to hour 0 and December is close to January.
    frame["Hour_Sin"] = np.sin(2 * np.pi * frame["Hour"] / 24)
    frame["Hour_Cos"] = np.cos(2 * np.pi * frame["Hour"] / 24)
    frame["Month_Sin"] = np.sin(2 * np.pi * frame["Month"] / 12)
    frame["Month_Cos"] = np.cos(2 * np.pi * frame["Month"] / 12)

    # The persistence baseline needs current PM2.5, so require both current and
    # next-hour target. Other predictors/lags may remain missing and are imputed.
    frame = frame.dropna(subset=["PM25_Current", "Target_PM25_NextHour"])
    frame = frame[
        np.isfinite(frame["PM25_Current"]) & np.isfinite(frame["Target_PM25_NextHour"])
    ].copy()
    frame = frame.sort_values(["DateTime", "Location"]).reset_index(drop=True)

    if len(frame) < 100:
        raise ValueError(f"Only {len(frame)} usable next-hour forecasting rows were created.")

    return frame


def chronological_train_validation_test(frame):
    """Use older years for training and the newest years for validation/test."""
    dt = frame["DateTime"]

    # Preferred project split: train through 2022, validate on 2023, test on 2024.
    train = frame[dt.dt.year <= 2022].copy()
    validation = frame[dt.dt.year == 2023].copy()
    test = frame[dt.dt.year >= 2024].copy()

    if len(train) >= 500 and len(validation) >= 100 and len(test) >= 100:
        description = "Train: through 2022 | Validation: 2023 | Test: 2024+"
        return train, validation, test, description

    # Fallback: chronological 70/15/15 by unique timestamps.
    ordered_times = np.array(sorted(frame["DateTime"].dropna().unique()))
    if len(ordered_times) < 20:
        raise ValueError("Not enough unique timestamps for a chronological split.")

    train_cut = ordered_times[int(len(ordered_times) * 0.70)]
    validation_cut = ordered_times[int(len(ordered_times) * 0.85)]

    train = frame[frame["DateTime"] < train_cut].copy()
    validation = frame[(frame["DateTime"] >= train_cut) & (frame["DateTime"] < validation_cut)].copy()
    test = frame[frame["DateTime"] >= validation_cut].copy()

    if min(len(train), len(validation), len(test)) < 50:
        raise ValueError(
            f"Chronological fallback split too small: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}"
        )

    description = "Chronological 70/15/15 fallback"
    return train, validation, test, description


def regression_metrics(name, y_true, y_pred):
    return {
        "Model": name,
        "Rows": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_pipeline(estimator, X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def choose_recent_rows(frame, max_rows):
    """For faster fitting, use the most recent training rows instead of random sampling."""
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    return frame.sort_values("DateTime").tail(max_rows).copy()


def main():
    parser = argparse.ArgumentParser(description="Week 6 next-hour PM2.5 time-series model comparison")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to cleaned Dhaka CSV")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=10000,
        help="Use the most recent N training rows for tree models; 0 means all rows",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset: {data_path}\n"
            "Run from the repository root or pass --data with the cleaned CSV path."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {data_path}")
    raw = pd.read_csv(data_path, low_memory=False)
    frame = prepare_forecasting_frame(raw)
    print(f"Usable next-hour forecasting rows: {len(frame):,}")

    train_full, validation, test, split_description = chronological_train_validation_test(frame)
    train = choose_recent_rows(train_full, args.max_train_rows)

    target = "Target_PM25_NextHour"
    excluded = {target, "DateTime"}
    features = [c for c in frame.columns if c not in excluded]

    X_train, y_train = train[features], train[target]
    X_val, y_val = validation[features], validation[target]
    X_test, y_test = test[features], test[target]

    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18,
            min_samples_leaf=5,
            random_state=args.random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.random_state,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.random_state,
        ),
    }

    # ------------------------------------------------------------------
    # 1) Validation model comparison
    # ------------------------------------------------------------------
    validation_results = []

    # Faculty-appropriate time-series baseline: next hour equals current hour.
    val_persistence = validation["PM25_Current"].to_numpy(dtype=float)
    validation_results.append(regression_metrics(
        "Persistence Baseline (next hour = current PM2.5)", y_val, val_persistence
    ))

    fitted_on_train = {}
    for name, estimator in models.items():
        print(f"Training {name} on historical training period...")
        pipeline = build_pipeline(deepcopy(estimator), X_train)
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        validation_results.append(regression_metrics(name, y_val, val_pred))
        fitted_on_train[name] = pipeline

    validation_comparison = pd.DataFrame(validation_results).sort_values("RMSE").reset_index(drop=True)
    validation_comparison.to_csv(OUTPUT_DIR / "validation_model_comparison.csv", index=False)

    nonbaseline = validation_comparison[
        validation_comparison["Model"] != "Persistence Baseline (next hour = current PM2.5)"
    ]
    best_model_name = str(nonbaseline.iloc[0]["Model"])
    print(f"Best ML model by validation RMSE: {best_model_name}")

    # ------------------------------------------------------------------
    # 2) Refit candidates on train + validation, then evaluate newest test
    # ------------------------------------------------------------------
    train_plus_val = pd.concat([train_full, validation], ignore_index=True)
    train_plus_val = choose_recent_rows(train_plus_val, args.max_train_rows)
    X_train_val = train_plus_val[features]
    y_train_val = train_plus_val[target]

    test_results = []
    test_prediction_frames = []

    test_persistence = test["PM25_Current"].to_numpy(dtype=float)
    test_results.append(regression_metrics(
        "Persistence Baseline (next hour = current PM2.5)", y_test, test_persistence
    ))
    test_prediction_frames.append(pd.DataFrame({
        "DateTime": test["DateTime"].astype(str).to_numpy(),
        "Location": test["Location"].astype(str).to_numpy(),
        "Model": "Persistence Baseline (next hour = current PM2.5)",
        "Current_PM2.5": test["PM25_Current"].to_numpy(dtype=float),
        "Actual_NextHour_PM2.5": y_test.to_numpy(dtype=float),
        "Predicted_NextHour_PM2.5": test_persistence,
        "Error": test_persistence - y_test.to_numpy(dtype=float),
        "Absolute_Error": np.abs(test_persistence - y_test.to_numpy(dtype=float)),
    }))

    fitted_final = {}
    for name, estimator in models.items():
        print(f"Refitting {name} on training + validation periods...")
        pipeline = build_pipeline(deepcopy(estimator), X_train_val)
        pipeline.fit(X_train_val, y_train_val)
        pred = pipeline.predict(X_test)
        test_results.append(regression_metrics(name, y_test, pred))
        test_prediction_frames.append(pd.DataFrame({
            "DateTime": test["DateTime"].astype(str).to_numpy(),
            "Location": test["Location"].astype(str).to_numpy(),
            "Model": name,
            "Current_PM2.5": test["PM25_Current"].to_numpy(dtype=float),
            "Actual_NextHour_PM2.5": y_test.to_numpy(dtype=float),
            "Predicted_NextHour_PM2.5": pred,
            "Error": pred - y_test.to_numpy(dtype=float),
            "Absolute_Error": np.abs(pred - y_test.to_numpy(dtype=float)),
        }))
        fitted_final[name] = pipeline

    test_comparison = pd.DataFrame(test_results).sort_values("RMSE").reset_index(drop=True)
    test_comparison.to_csv(OUTPUT_DIR / "test_model_comparison.csv", index=False)

    predictions = pd.concat(test_prediction_frames, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Save split summary so the report can prove there was no random train/test split.
    split_summary = pd.DataFrame([
        {
            "Split": "Train",
            "Rows": len(train_full),
            "Start": train_full["DateTime"].min(),
            "End": train_full["DateTime"].max(),
        },
        {
            "Split": "Validation",
            "Rows": len(validation),
            "Start": validation["DateTime"].min(),
            "End": validation["DateTime"].max(),
        },
        {
            "Split": "Test",
            "Rows": len(test),
            "Start": test["DateTime"].min(),
            "End": test["DateTime"].max(),
        },
    ])
    split_summary["Description"] = split_description
    split_summary.to_csv(OUTPUT_DIR / "time_split_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Detailed error analysis for the validation-selected ML model
    # ------------------------------------------------------------------
    best_pipeline = fitted_final[best_model_name]
    best_predictions = predictions[predictions["Model"] == best_model_name].copy()

    best_predictions.sort_values("Absolute_Error", ascending=False).head(100).to_csv(
        OUTPUT_DIR / "largest_100_prediction_errors.csv", index=False
    )

    error_summary = pd.DataFrame([{
        "Selected_Model": best_model_name,
        "Selection_Basis": "Lowest validation RMSE among ML models",
        "Test_Rows": len(best_predictions),
        "Mean_Error_Bias": float(best_predictions["Error"].mean()),
        "Median_Absolute_Error": float(best_predictions["Absolute_Error"].median()),
        "90th_Percentile_Absolute_Error": float(best_predictions["Absolute_Error"].quantile(0.90)),
        "Max_Absolute_Error": float(best_predictions["Absolute_Error"].max()),
        "Split": split_description,
    }])
    error_summary.to_csv(OUTPUT_DIR / "error_summary.csv", index=False)

    monthly = best_predictions.copy()
    monthly["DateTime"] = pd.to_datetime(monthly["DateTime"], errors="coerce")
    monthly["Month"] = monthly["DateTime"].dt.to_period("M").astype(str)
    monthly_error = monthly.groupby("Month", as_index=False).agg(
        Rows=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        Mean_Error_Bias=("Error", "mean"),
    )
    monthly_error.to_csv(OUTPUT_DIR / "monthly_test_error.csv", index=False)

    # Save the final model and metadata needed by the later web app.
    joblib.dump({
        "model": best_pipeline,
        "target": "PM2.5 one hour ahead",
        "feature_columns": features,
        "lag_hours": PM25_LAG_HOURS,
        "best_model_name": best_model_name,
        "selection_basis": "Lowest validation RMSE among ML models",
        "split": split_description,
    }, MODEL_DIR / "Week6_Best_NextHour_PM25_Model.joblib")

    # Feature importance for the selected tree model.
    try:
        prep = best_pipeline.named_steps["preprocessor"]
        model = best_pipeline.named_steps["model"]
        names = prep.get_feature_names_out()
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)
        importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

        top = importance_df.head(15).sort_values("Importance")
        plt.figure(figsize=(9, 6))
        plt.barh(top["Feature"], top["Importance"])
        plt.title(f"Top 15 Features — {best_model_name}")
        plt.xlabel("Feature importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"Feature importance export skipped: {exc}")

    # Validation comparison plot.
    plt.figure(figsize=(9, 5))
    plt.bar(validation_comparison["Model"], validation_comparison["RMSE"])
    plt.title("Week 6 Validation Model Comparison — Next-hour PM2.5")
    plt.ylabel("RMSE (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_model_comparison_rmse.png", dpi=160)
    plt.close()

    # Test actual vs predicted for the selected ML model.
    plt.figure(figsize=(6, 6))
    plt.scatter(
        best_predictions["Actual_NextHour_PM2.5"],
        best_predictions["Predicted_NextHour_PM2.5"],
        alpha=0.30,
        s=12,
    )
    low = min(
        best_predictions["Actual_NextHour_PM2.5"].min(),
        best_predictions["Predicted_NextHour_PM2.5"].min(),
    )
    high = max(
        best_predictions["Actual_NextHour_PM2.5"].max(),
        best_predictions["Predicted_NextHour_PM2.5"].max(),
    )
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title(f"Actual vs Predicted Next-hour PM2.5 — {best_model_name}")
    plt.xlabel("Actual next-hour PM2.5")
    plt.ylabel("Predicted next-hour PM2.5")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted_next_hour.png", dpi=160)
    plt.close()

    # Residual distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(best_predictions["Error"], bins=40)
    plt.title(f"Next-hour PM2.5 Prediction Residuals — {best_model_name}")
    plt.xlabel("Predicted − actual PM2.5")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=160)
    plt.close()

    # A compact machine-readable summary for README/report generation.
    summary = pd.DataFrame([{
        "Forecast_Target": "PM2.5 at t+1 hour",
        "Persistence_Baseline": "Predicted PM2.5(t+1) = PM2.5(t)",
        "Lag_Features": ", ".join(f"{h}h" for h in PM25_LAG_HOURS),
        "Selected_Model": best_model_name,
        "Train_Rows_Full": len(train_full),
        "Train_Rows_Fitted": len(train),
        "Validation_Rows": len(validation),
        "Test_Rows": len(test),
        "Split": split_description,
    }])
    summary.to_csv(OUTPUT_DIR / "week6_run_summary.csv", index=False)

    print("\nValidation comparison (model selection):")
    print(validation_comparison.to_string(index=False))
    print("\nFinal test comparison:")
    print(test_comparison.to_string(index=False))
    print(f"\nSplit: {split_description}")
    print(f"Selected ML model: {best_model_name}")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Model saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
