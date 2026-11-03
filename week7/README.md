# Dhaka Air Quality Predictor — Week 7 Update

This folder contains the **Week 7 mid-project update** for CSE299 Section 14, Group 8.

Week 7 focuses on the **first working web prototype and ML-model integration**. It reuses the selected Week 6 next-hour PM2.5 Random Forest model and connects it to a FastAPI backend plus an interactive browser dashboard.

## Week 7 work completed
- Built the initial web-app/dashboard prototype.
- Connected the trained Week 6 Random Forest model to a Python FastAPI backend.
- Added a real `/api/predict` endpoint for next-hour PM2.5 forecasting.
- Added current PM2.5 and 1h / 3h / 6h / 24h lag inputs.
- Added station, weather, pollutant and time inputs.
- Added model status, prediction output, model metrics, feature importance and 2024 actual-vs-predicted visualization.
- Removed the old hand-written JavaScript prediction formula and placeholder XGBoost results.
- Kept official AQI separate from the one-hour PM2.5 forecast; the page shows only an indicative PM2.5 sub-index/category.

## Important continuity
**Week 5:** baseline + first Random Forest model.

**Week 6:** time-series refinement, lag features, model comparison and error analysis.

**Week 7:** initial web prototype + integration of the selected Week 6 model.

This does **not** mean the final Week 9 dashboard is complete. Week 7 is an integration prototype for the mid-project update.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Then open `http://127.0.0.1:8000`.

On Windows, you can also run `run_windows.bat`.

## Main files
- `app.py` — FastAPI backend and real model inference.
- `templates/index.html` — Week 7 dashboard prototype.
- `models/Week6_Best_NextHour_PM25_Model.joblib` — selected Week 6 forecasting model reused in Week 7.
- `week6_model_reference/` — Week 6 metrics and outputs used as model-reference evidence.
- `WEEK7_UPDATE.md` — concise update for faculty/GitHub.
