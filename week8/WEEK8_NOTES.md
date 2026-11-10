# Week 8 Notes

## Goal

Use the last 7 days (168 hours) to forecast hourly PM2.5 for:
- the next day (24 values)
- the next 7 days (168 values)

## What was done

- Prepared forecast windows separately for each monitoring station.
- Kept missing hours in the timeline. Future missing readings are not used when calculating errors.
- Trained one Gradient Boosting model for each forecast length. For each horizon, training used up to 250 forecast starting points per station (500 total in this run). Validation and testing used all eligible starting points.
- Compared it with Persistence and Daily repeat.
- Used data through 2022 for training, 2023 for validation, and 2024 for testing.
- Added the forecast graph to the dashboard.
- Added a separate script to test Chronos-Bolt on the same dates.

## Test results — 2024

| Forecast length | Method | MAE | RMSE |
|---|---|---:|---:|
| 24 hours | Gradient boosting | 26.10 | 44.10 |
| 24 hours | Daily repeat | 32.65 | 55.47 |
| 24 hours | Persistence | 32.61 | 55.98 |
| 168 hours | Gradient boosting | 31.77 | 49.69 |
| 168 hours | Daily repeat | 39.26 | 63.66 |
| 168 hours | Persistence | 40.18 | 64.02 |

Lower MAE and RMSE are better. The Gradient Boosting model has the lowest errors in both tests.

For 24-hour forecasts, 815 daily starting points and 19,042 observed future hours were scored. For 168-hour forecasts, 742 starting points and 117,854 observed hours were scored. The same future observations were used to compare methods within each forecast length.

## How the methods work

- **Persistence:** every future hour is the latest PM2.5 value.
- **Daily repeat:** the last 24 hours are repeated into the future.
- **Gradient boosting:** learns patterns from past readings, forecast hour, date and station.
- **Chronos-Bolt:** can predict without training on our dataset. The comparison script is provided, but a verified Chronos result has not been recorded yet.

## Important points

- 24-hour and 168-hour results are separate tasks. They should not be compared with the older one-hour model scores.
- The forecast is for PM2.5 concentration in µg/m³, not official AQI.
- Time-LLM and fine-tuned pretrained model testing have not been completed yet.
- 7-day forecasts start each day, so forecast windows can overlap. Reported observed-hour totals are scoring points, not independent hours.
