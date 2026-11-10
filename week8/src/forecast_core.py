"""Shared hourly-window code for the Week 8 PM2.5 forecast experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

CONTEXT_HOURS = 168
HORIZONS = (24, 168)
STATION_NAMES = {"BARC": 0, "Darussalam": 1, "DoE": 2}


@dataclass
class Window:
    station: str
    origin: pd.Timestamp
    split: str
    history: np.ndarray  # always 168 past hours, already filled from past only
    actual: np.ndarray  # 24/168 future hours, missing values retained as nan


def read_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Location", "DateTime", "PM2.5"], low_memory=False)
    df.columns = ["station", "time", "pm25"]
    df["station"] = df["station"].astype(str).str.strip().replace({
        "BARC (Farmgate)": "BARC", "Darussalam (Mirpur)": "Darussalam",
        "DoE (Agargaon)": "DoE", "DOE": "DoE",
    })
    df = df[df.station.isin(STATION_NAMES)]
    df["time"] = pd.to_datetime(df.time, errors="coerce").dt.floor("h")
    df["pm25"] = pd.to_numeric(df.pm25, errors="coerce")
    df.loc[~df.pm25.between(0, 1000), "pm25"] = np.nan
    df = df.dropna(subset=["time"]).sort_values(["station", "time", "pm25"], na_position="first")
    df = df.drop_duplicates(["station", "time"], keep="last")
    return df


def make_windows(data: pd.DataFrame, horizon: int, min_history: float = .70,
                 min_target: float = .70) -> tuple[list[Window], pd.DataFrame]:
    """Daily origins (23:00); targets cannot cross a split-year boundary.

    We never fill target values; only observed future hours contribute to MAE/RMSE.
    """
    if horizon not in HORIZONS:
        raise ValueError("Horizon must be 24 or 168 hours")
    windows, coverage = [], []
    for station, group in data.groupby('station', sort=True):
        series = group.set_index('time').pm25.sort_index().asfreq('h')
        coverage.append({'Station': station, 'Hours': len(series),
                         'ObservedPM25': int(series.notna().sum()),
                         'Coverage': round(float(series.notna().mean()), 4)})
        vals = series.to_numpy(dtype=float)
        dates = series.index
        for idx in np.flatnonzero(dates.hour == 23):
            if idx + horizon >= len(vals) or idx < CONTEXT_HOURS - 1:
                continue
            future_start = dates[idx + 1]
            future_end = dates[idx + horizon]
            if future_start.year != future_end.year:
                continue
            year = future_start.year
            split = 'train' if year <= 2022 else 'validation' if year == 2023 else 'test' if year == 2024 else None
            if split is None:
                continue
            raw_history = vals[idx - CONTEXT_HOURS + 1:idx + 1].copy()
            actual = vals[idx + 1:idx + horizon + 1].copy()
            if (np.isfinite(raw_history).mean() < min_history or
                np.isfinite(actual).mean() < min_target or
                not np.isfinite(raw_history[-1])):
                continue
            # Only the 168 known past hours can supply fill values.
            history = pd.Series(raw_history).ffill().bfill().to_numpy(dtype=float)
            if np.isfinite(history).all():
                windows.append(Window(station, dates[idx], split, history, actual))
    return windows, pd.DataFrame(coverage)


def sample_windows(windows: list[Window], split: str, per_station: int = 0) -> list[Window]:
    chosen = [w for w in windows if w.split == split]
    if per_station <= 0:
        return chosen
    sampled = []
    for station in STATION_NAMES:
        group = [w for w in chosen if w.station == station]
        if not group:
            continue
        # Spread sampled forecast origins through all available years/months.
        ids = np.linspace(0, len(group) - 1, min(per_station, len(group)), dtype=int)
        sampled.extend(group[int(i)] for i in ids)
    return sorted(sampled, key=lambda w: (w.station, w.origin))


def baseline(history: np.ndarray, horizon: int, method: str) -> np.ndarray:
    if method == 'Persistence':
        return np.full(horizon, float(history[-1]))
    if method == 'Daily repeat':
        return np.tile(history[-24:], int(np.ceil(horizon / 24)))[:horizon]
    raise ValueError('Unknown baseline')


def build_features(history: np.ndarray, station: str, origin: pd.Timestamp,
                   horizon: int) -> pd.DataFrame:
    """Lead-aware supervised forecasting: each future hour is a separate input row.

    No future measurement, actual weather or other unavailable input is used.
    """
    hist = np.asarray(history, dtype=float)
    if hist.shape != (CONTEXT_HOURS,) or not np.isfinite(hist).all():
        raise ValueError('Need exactly 168 valid historical PM2.5 values')
    leads = np.arange(1, horizon + 1)
    future = origin + pd.to_timedelta(leads, unit='h')
    daypos = (leads - 1) % 24
    summary = {}
    for hours in (6, 24, 72, 168):
        recent = hist[-hours:]
        summary[f'mean_{hours}h'] = float(np.mean(recent))
        summary[f'std_{hours}h'] = float(np.std(recent))
        summary[f'max_{hours}h'] = float(np.max(recent))
        summary[f'min_{hours}h'] = float(np.min(recent))
    feats = pd.DataFrame({
        'station_code': np.full(horizon, STATION_NAMES[station]),
        'lead_hour': leads,
        'lead_day': (leads - 1) // 24,
        'hour_sin': np.sin(2 * np.pi * future.hour.to_numpy() / 24),
        'hour_cos': np.cos(2 * np.pi * future.hour.to_numpy() / 24),
        'month_sin': np.sin(2 * np.pi * future.month.to_numpy() / 12),
        'month_cos': np.cos(2 * np.pi * future.month.to_numpy() / 12),
        'dow_sin': np.sin(2 * np.pi * future.dayofweek.to_numpy() / 7),
        'dow_cos': np.cos(2 * np.pi * future.dayofweek.to_numpy() / 7),
        'recent_pm25': np.full(horizon, hist[-1]),
        'lag_1h': np.full(horizon, hist[-2]),
        'lag_3h': np.full(horizon, hist[-4]),
        'lag_6h': np.full(horizon, hist[-7]),
        'lag_24h': np.full(horizon, hist[-25]),
        'lag_48h': np.full(horizon, hist[-49]),
        'last_day_same_hour': hist[-24:][daypos],
        'last_week_same_hour': hist[-168:-144][daypos],
        'trend_24h': np.full(horizon, np.mean(hist[-24:]) - np.mean(hist[-48:-24])),
    })
    for key, value in summary.items():
        feats[key] = value
    return feats


def training_matrix(windows: list[Window], horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    frames, targets = [], []
    for w in windows:
        observed = np.isfinite(w.actual)
        if observed.any():
            frames.append(build_features(w.history, w.station, w.origin, horizon).loc[observed])
            targets.append(w.actual[observed])
    if not frames:
        raise ValueError('Not enough training observations')
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


def forecast_ml(model, history: np.ndarray, station: str, origin: pd.Timestamp,
                horizon: int) -> np.ndarray:
    features = build_features(history, station, origin, horizon)
    return np.maximum(0, np.asarray(model.predict(features), dtype=float).reshape(-1))
