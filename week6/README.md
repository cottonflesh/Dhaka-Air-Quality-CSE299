# Week 6 — Next-hour PM2.5 Model Comparison & Error Analysis

This folder continues the Week 5 work. Week 5 remains unchanged.

## What changed in Week 6

The task is now treated as a time-series forecasting problem:

**information available at hour t → predict PM2.5 at hour t+1**

To avoid temporal leakage, the split is chronological rather than random:

- Train: through 2022
- Validation: 2023
- Test: 2024+

The forecasting features include current PM2.5, exact PM2.5 lags at 1h, 3h, 6h and 24h, current weather/pollutant variables, location, and calendar/time variables.

## Baseline

The baseline is a persistence forecast:

**Predicted PM2.5(t+1) = PM2.5(t)**

This is more appropriate for an hourly time-series problem than predicting the training-set mean.

## Models compared

- Decision Tree
- Random Forest
- Extra Trees
- Persistence baseline

The best ML model is selected using validation RMSE, then evaluated on the newest test period.

## Current run results

### Validation — 2023

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 19.82 | **33.63** | **0.828** |
| Persistence baseline | **15.91** | 34.56 | 0.818 |
| Extra Trees | 19.67 | 35.73 | 0.805 |
| Decision Tree | 24.07 | 40.88 | 0.745 |

Random Forest is selected because it has the lowest validation RMSE among the ML models.

### Final test — 2024+

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 16.18 | **32.36** | **0.788** |
| Extra Trees | 15.31 | 33.02 | 0.779 |
| Persistence baseline | **15.19** | 36.70 | 0.727 |
| Decision Tree | 22.48 | 40.84 | 0.662 |

The persistence baseline still has the smallest MAE, while Random Forest has the best RMSE and R². This means Random Forest reduces larger prediction errors more effectively, although the simple persistence forecast remains very competitive for typical hourly changes.

## Run

From the repository root:

```bash
python week6/src/week6_model_comparison.py
```

The cleaned dataset is expected at:

`data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv`

## Main outputs

- `validation_model_comparison.csv`
- `test_model_comparison.csv`
- `test_predictions.csv`
- `time_split_summary.csv`
- `error_summary.csv`
- `monthly_test_error.csv`
- `feature_importance.csv`
- comparison/error-analysis graphs
