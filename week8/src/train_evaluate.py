"""Week 8: 24h and 168h PM2.5 forecasts from 168h historical context.

Run from the repository root:
    python week8/src/train_evaluate.py

2023 validation and 2024 test are held out from training. The same future
observations are used for every method within a horizon and split.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast_core import (HORIZONS, Window, read_data, make_windows, sample_windows,
                           baseline, training_matrix, forecast_ml)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / 'data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv'
DEFAULT_OUTPUTS = ROOT / 'week8/outputs'
DEFAULT_MODELS = ROOT / 'week8/models'


def metric_rows(data: pd.DataFrame) -> list[dict]:
    results = []
    for (horizon, split, model), group in data.groupby(['HorizonHours','Split','Model'], sort=True):
        diff = group.predicted.to_numpy() - group.actual.to_numpy()
        results.append({'HorizonHours':int(horizon), 'Split':split, 'Model':model,
                        'DailyOrigins':int(group[['Station','Origin']].drop_duplicates().shape[0]),
                        'ObservedHours':len(group),
                        'MAE':round(float(np.mean(np.abs(diff))), 3),
                        'RMSE':round(float(np.sqrt(np.mean(diff**2))), 3)})
    return results


def append_predictions(rows: list[dict], window: Window, method: str,
                       forecast: np.ndarray, horizon: int) -> None:
    keep = np.flatnonzero(np.isfinite(window.actual))
    for h in keep:
        rows.append({'HorizonHours':horizon, 'Split':window.split, 'Model':method,
                     'Station':window.station, 'Origin':str(window.origin),
                     'ForecastTime':str(window.origin + pd.Timedelta(hours=int(h)+1)),
                     'LeadHour':int(h)+1, 'actual':float(window.actual[h]),
                     'predicted':float(forecast[h])})


def evaluate_all(windows: list[Window], model, horizon: int,
                 output: Path, save_points: bool = False) -> pd.DataFrame:
    rows = []
    for w in windows:
        forecasts = {
            'Persistence':baseline(w.history, horizon, 'Persistence'),
            'Daily repeat':baseline(w.history, horizon, 'Daily repeat'),
            'Gradient boosting':forecast_ml(model, w.history, w.station, w.origin, horizon),
        }
        for method, prediction in forecasts.items():
            append_predictions(rows, w, method, prediction, horizon)
    points = pd.DataFrame(rows)
    if points.empty:
        raise RuntimeError('No valid forecast points available for evaluation')
    if save_points:
        points.to_csv(output / f'forecast_points_{horizon}h.csv', index=False)
    # Forecast lead errors make short vs longer horizons easier to compare.
    lead = points.groupby(['Split','Model','LeadHour'], sort=True).apply(
        lambda grp:pd.Series({'MAE':float(np.abs(grp.predicted-grp.actual).mean()),
                              'RMSE':float(np.sqrt(np.mean((grp.predicted-grp.actual)**2))),
                              'Hours':len(grp)}), include_groups=False).reset_index()
    lead.insert(0, 'HorizonHours', horizon)
    lead.round(3).to_csv(output / f'lead_hour_errors_{horizon}h.csv', index=False)
    ex = points[(points.Split == 'test')].sort_values(['Station','Origin','LeadHour'])
    if not ex.empty:
        first_station = ex.Station.iloc[0]
        first_origin = ex[ex.Station == first_station].Origin.iloc[0]
        ex = ex[(ex.Station == first_station) & (ex.Origin == first_origin)]
        ex.to_csv(output / f'example_forecast_{horizon}h.csv', index=False)
    return points


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    data = read_data(args.data)
    results, counts = [], []
    for horizon in HORIZONS:
        print(f'\nPreparing {horizon}h forecasting windows...', flush=True)
        windows, coverage = make_windows(data, horizon, args.context_coverage, args.target_coverage)
        coverage.to_csv(args.output / 'station_coverage.csv',index=False)
        train_windows = sample_windows(windows, 'train', args.train_origins_per_station)
        validation_windows = sample_windows(windows, 'validation', args.eval_origins_per_station)
        test_windows = sample_windows(windows, 'test', args.eval_origins_per_station)
        if not train_windows or not validation_windows or not test_windows:
            raise RuntimeError(f'Missing train/validation/test windows for {horizon}h')
        counts.append({'HorizonHours':horizon, 'TrainOrigins':len(train_windows),
                       'ValidationOrigins':len(validation_windows), 'TestOrigins':len(test_windows)})
        print(f'  train={len(train_windows)}, validation={len(validation_windows)}, test={len(test_windows)}', flush=True)
        X, y = training_matrix(train_windows, horizon)
        print(f'  training on {len(y):,} observed target hours...', flush=True)
        model = HistGradientBoostingRegressor(
            max_iter=args.iterations, learning_rate=0.07, max_leaf_nodes=25,
            l2_regularization=2., min_samples_leaf=35, random_state=42,
        )
        model.fit(X, y)
        path = args.models / f'pm25_{horizon}h_gradient_boosting.joblib'
        joblib.dump({'model':model, 'horizon':horizon,
                     'context_hours':168, 'model_name':'Gradient boosting',
                     'train_end':'2022-12-31', 'validation_year':2023,
                     'test_year':2024, 'features':list(X.columns)}, path)
        print(f'  saved model: {path.name}',flush=True)
        points = evaluate_all(validation_windows + test_windows, model, horizon,
                              args.output, args.save_points)
        results.extend(metric_rows(points))
        del points, X, y, model

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(['HorizonHours','Split','RMSE']).reset_index(drop=True)
    comparison.to_csv(args.output / 'model_comparison.csv',index=False)
    pd.DataFrame(counts).to_csv(args.output / 'forecast_window_counts.csv',index=False)
    metadata = {'data_file':str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data), 'train_until':'2022-12-31',
                'validation':'2023', 'test':'2024', 'context_hours':168,
                'daily_forecast_origin':'23:00', 'horizons_hours':[24,168],
                'train_origins_per_station_limit':args.train_origins_per_station,
                'eval_origins_per_station_limit':args.eval_origins_per_station,
                'iterations':args.iterations,
                'context_coverage_threshold':args.context_coverage,
                'target_coverage_threshold':args.target_coverage,
                'missing_future_values':'excluded from scoring',
                'context_imputation':'within preceding 168 hours only'}
    (args.output/'run_settings.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('\n'+comparison.to_string(index=False),flush=True)
    return comparison


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=DEFAULT_CSV)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUTS)
    ap.add_argument('--models',type=Path,default=DEFAULT_MODELS)
    ap.add_argument('--train-origins-per-station',type=int,default=250,
                    help='Evenly sampled past-day origins per station; 0 uses every valid day')
    ap.add_argument('--eval-origins-per-station',type=int,default=0,
                    help='0 evaluates every eligible validation/test origin')
    ap.add_argument('--iterations',type=int,default=100)
    ap.add_argument('--context-coverage',type=float,default=.7)
    ap.add_argument('--target-coverage',type=float,default=.7)
    ap.add_argument('--save-points',action='store_true',
                    help='Save all hourly predictions (creates larger CSV files)')
    args=ap.parse_args()
    if not args.data.exists():ap.error(f'Data not found: {args.data}')
    if args.train_origins_per_station < 0 or args.eval_origins_per_station < 0:
        ap.error('Origin limits must be 0 or greater')
    train(args)

if __name__=='__main__':main()
