# Dhaka Air Quality Prediction System

CSE299 | Section 14 | Group 8

## Team

| Member | Student ID |
|---|---|
| Nizam Uddin Nihon | 2031139342 |
| Ikramun Masud Lubaba | 2232451042 |
| Md Huzayfa | 2222948642 |

## Project

PM2.5 forecasting for Dhaka using hourly CAMS-BD data from BARC, Darussalam and DoE (2012–2024).

- Week 4: cleaned the Dhaka monitoring data.
- Week 5: initial PM2.5 model and baseline.
- Week 6: next-hour forecasting and model comparison.
- Week 7: FastAPI dashboard with the next-hour Random Forest model.
- Week 8: 24-hour and 168-hour forecasting using the previous 168 hours of PM2.5 measurements.

## Run the dashboard

Open Command Prompt in the `week8` folder and run:

```cmd
run_windows.bat
```

Then open http://127.0.0.1:8000.

For the training commands, results and model details, see [week8/README.md](week8/README.md).

The cleaned dataset is at `data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv`.

## 2024 Week 8 test results

| Forecast | Gradient boosting MAE | Gradient boosting RMSE |
|---|---:|---:|
| 24 hours | 26.10 | 44.10 |
| 168 hours | 31.77 | 49.69 |

Full results: [week8/outputs/model_comparison.csv](week8/outputs/model_comparison.csv).

## Data source

CAMS-BD: https://data.mendeley.com/datasets/hwmzyh5w4w/1
