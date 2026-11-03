# Model provenance

The Week 7 web prototype integrates `models/Week6_Best_NextHour_PM25_Model.joblib`.


Forecast target: PM2.5 one hour ahead.
Lag features: current PM2.5 plus 1h, 3h, 6h and 24h PM2.5 history.
Chronological evaluation: training through 2022, validation on 2023, testing on 2024+.
