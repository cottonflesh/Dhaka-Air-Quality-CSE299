# Week 5 — Baseline + First ML Model

Week 5 implements the first machine-learning experiment for the Dhaka Air project.

## Target
The Week 5 experiment predicts **PM2.5 concentration**. The supplied dataset did not contain an AQI target column, so these results must not be presented as AQI-prediction results.

## Models
- Baseline: constant prediction using the training-set mean PM2.5.
- First ML model: Random Forest Regressor.

## Initial held-out results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline (training mean) | 54.2878 | 70.9029 | -0.0063 |
| Random Forest Regressor | 28.0010 | 45.5150 | 0.5853 |

## Files
- `src/Week5_Training_Code.py` — training code used for the Week 5 experiment.
- `results/Week5_Model_Evaluation.csv` — baseline vs Random Forest metrics.
- `results/Week5_PM25_Test_Predictions.csv` — actual and predicted test values.
- `results/Week5_Feature_Importance.csv` — Random Forest feature importance.
- `results/Week5_Model_Results.xlsx` — formatted results workbook.
- `models/Week5_Random_Forest_PM25_Model.joblib` — fitted model pipeline.
- `reports/Week5_Model_Training_Report.*` — technical Week 5 notes.

## Reproducibility note
The Week 5 artifacts were produced from the prepared Week 5 dataset snapshot used at that milestone. The repository's `data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv` is the later refreshed cleaned dataset. Re-running the model on the refreshed file may produce slightly different metrics.
