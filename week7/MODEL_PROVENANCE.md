# Model provenance

The Week 7 web prototype integrates `models/Week6_Best_NextHour_PM25_Model.joblib`.

That model was selected during the **Week 6 time-series model-comparison workflow**. Week 7 does not claim that a new model was trained; the work is the **integration of the already-selected model into the first working web prototype**.

Forecast target: PM2.5 one hour ahead.
Lag features: current PM2.5 plus 1h, 3h, 6h and 24h PM2.5 history.
Chronological evaluation: training through 2022, validation on 2023, testing on 2024+.
