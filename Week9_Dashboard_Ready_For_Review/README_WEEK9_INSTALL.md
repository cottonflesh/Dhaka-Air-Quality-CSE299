# Week 9 Dashboard — Ready to Install

This upgrade adds a historical analytics dashboard to the existing Week 8 FastAPI application. It leaves the Week 8 predictor and forecast logic intact.

## Install

1. Extract this ZIP into the root of `Dhaka-Air-Quality-CSE299` (the folder containing `week8/` and `data/`).
2. Run:
   ```bash
   python install_week9_dashboard.py
   ```
3. Start the Week 8 app using your usual command.
4. Open `http://127.0.0.1:8000/week9`.

## Dashboard features

- Historical PM2.5 and PM10 trend (monthly or daily).
- Station and year-range filters.
- Station coverage chart and station summary table.
- Week 8 model comparison using validation/test MAE.
- Gradient Boosting lead-hour error chart for 24-hour and 168-hour forecasts.

## Data and dependencies

Uses the existing cleaned dataset at `data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv` and existing Week 8 CSV outputs in `week8/outputs/`. It uses the libraries already imported by Week 8; no new Python packages are required. The HTML uses CDN-hosted Tailwind CSS, Chart.js, and Inter font, so those visual assets require an internet connection in the browser.

All observations are historical. The dashboard does not claim to provide live sensor readings or official AQI.

The installer is idempotent: it detects the Week 9 marker before modifying `week8/app.py` again.
