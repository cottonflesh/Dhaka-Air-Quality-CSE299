# Week 8 — PM2.5 Forecasting

Forecast the next **24 hours** or **7 days (168 hours)** using the previous **168 hourly readings**.

## Run the dashboard (Windows)

1. Open this `week8` folder in Command Prompt.
2. Run `run_windows.bat`.
3. Open http://127.0.0.1:8000.
4. Scroll to **24-hour and 7-day PM2.5 forecast**. Click **Load measured example**, select a forecast length, and click **Forecast PM2.5**.

You can also paste 168 readings, oldest first. Use `null` for a missing hour. The last reading must be available.

## Train and test again

From the main repository folder:

```cmd
python -m pip install -r week8/requirements.txt
python week8/src/train_evaluate.py
```

The code saves models in `week8/models/` and results in `week8/outputs/`.

## Chronos comparison (optional)

This needs an internet connection the first time to download the model.

```cmd
python -m pip install -r week8/requirements_chronos.txt
python week8/src/test_chronos.py
```

It checks Chronos, Gradient Boosting, and both baselines on the same sampled dates. Results are saved as `week8/outputs/chronos_comparison.csv`.

## Files

- `app.py` — dashboard and forecast API
- `src/forecast_core.py` — hourly history and forecast features
- `src/train_evaluate.py` — training, validation and testing
- `src/test_chronos.py` — pretrained Chronos experiment
- `outputs/model_comparison.csv` — 2023 and 2024 forecast errors
- `models/` — saved forecasting models

Notes: [WEEK8_NOTES.md](WEEK8_NOTES.md)

## Check the setup

From the project folder run `python -m unittest discover -s week8/tests -v`.
