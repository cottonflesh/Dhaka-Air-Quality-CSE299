from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT.parent / "week7" / "models" / "Week6_Best_NextHour_PM25_Model.joblib"
INDEX_PATH = ROOT / "templates" / "index.html"

app = FastAPI(title="Dhaka Air Quality Predictor", version="week8")

MODEL_BUNDLE: dict[str, Any] | None = None
MODEL_LOAD_ERROR: str | None = None
try:
    MODEL_BUNDLE = joblib.load(MODEL_PATH)
except Exception as exc:  # Keep the page usable even if model file is missing.
    MODEL_LOAD_ERROR = str(exc)


class PredictionInput(BaseModel):
    location: str = Field(default="BARC")
    datetime: str
    pm25_current: float = Field(ge=0)
    pm25_lag_1h: float | None = Field(default=None, ge=0)
    pm25_lag_3h: float | None = Field(default=None, ge=0)
    pm25_lag_6h: float | None = Field(default=None, ge=0)
    pm25_lag_24h: float | None = Field(default=None, ge=0)
    pm10: float | None = Field(default=None, ge=0)
    temperature: float | None = None
    humidity: float | None = Field(default=None, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0)
    wind_direction: float | None = Field(default=None, ge=0, le=360)
    co: float | None = None
    no: float | None = None
    no2: float | None = None
    nox: float | None = None
    so2: float | None = None
    o3: float | None = None
    pressure: float | None = None
    solar_radiation: float | None = None
    rain: float | None = Field(default=None, ge=0)


def _pm25_subindex(pm25: float) -> tuple[int, str]:
    """Indicative PM2.5 sub-index using 2024 U.S. EPA breakpoints.

    This is shown only as an interpretation aid. Official PM2.5 AQI is based on
    the applicable 24-hour concentration, while this project predicts one hour ahead.
    """
    c = max(0.0, math.floor(float(pm25) * 10.0) / 10.0)
    bands = [
        (0.0, 9.0, 0, 50, "Good"),
        (9.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 125.4, 151, 200, "Unhealthy"),
        (125.5, 225.4, 201, 300, "Very Unhealthy"),
        (225.5, 325.4, 301, 500, "Hazardous"),
    ]
    for c_lo, c_hi, i_lo, i_hi, label in bands:
        if c <= c_hi:
            idx = ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo
            return int(round(idx)), label
    return 500, "Hazardous"


def _build_feature_row(payload: PredictionInput, feature_columns: list[str]) -> pd.DataFrame:
    try:
        dt = pd.Timestamp(payload.datetime)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid datetime.") from exc

    if pd.isna(dt):
        raise HTTPException(status_code=400, detail="Invalid datetime.")

    row = {
        "Location": payload.location.strip() or "BARC",
        "PM25_Current": payload.pm25_current,
        "PM10": payload.pm10,
        "Temperature": payload.temperature,
        "Humidity": payload.humidity,
        "WindSpeed": payload.wind_speed,
        "WindDirection": payload.wind_direction,
        "CO": payload.co,
        "NO": payload.no,
        "NO2": payload.no2,
        "NOX": payload.nox,
        "SO2": payload.so2,
        "O3": payload.o3,
        "Pressure": payload.pressure,
        "SolarRadiation": payload.solar_radiation,
        "Rain": payload.rain,
        "PM25_Lag_1h": payload.pm25_lag_1h,
        "PM25_Lag_3h": payload.pm25_lag_3h,
        "PM25_Lag_6h": payload.pm25_lag_6h,
        "PM25_Lag_24h": payload.pm25_lag_24h,
        "Month": dt.month,
        "Day": dt.day,
        "Hour": dt.hour,
        "DayOfWeek": dt.dayofweek,
        "Hour_Sin": np.sin(2 * np.pi * dt.hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * dt.hour / 24),
        "Month_Sin": np.sin(2 * np.pi * dt.month / 12),
        "Month_Cos": np.cos(2 * np.pi * dt.month / 12),
    }

    # Missing optional values stay as NaN and are handled by the trained pipeline's imputers.
    frame = pd.DataFrame([row])
    for col in feature_columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[feature_columns]


@app.get("/")
def home():
    return FileResponse(INDEX_PATH)


@app.get("/api/status")
def status():
    if MODEL_BUNDLE is None:
        return {
            "model_loaded": False,
            "error": MODEL_LOAD_ERROR,
            "expected_model_path": str(MODEL_PATH),
        }
    return {
        "model_loaded": True,
        "model": MODEL_BUNDLE.get("best_model_name", "Random Forest"),
        "target": MODEL_BUNDLE.get("target", "PM2.5 one hour ahead"),
        "lag_hours": MODEL_BUNDLE.get("lag_hours", [1, 3, 6, 24]),
        "split": MODEL_BUNDLE.get("split"),
        "selection_basis": MODEL_BUNDLE.get("selection_basis"),
    }


@app.post("/api/predict")
def predict(payload: PredictionInput):
    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="The Week 6 selected model is not loaded. Add models/Week6_Best_NextHour_PM25_Model.joblib.",
        )

    model = MODEL_BUNDLE["model"]
    feature_columns = MODEL_BUNDLE["feature_columns"]
    X = _build_feature_row(payload, feature_columns)

    prediction = float(model.predict(X)[0])
    prediction = max(0.0, prediction)
    subindex, category = _pm25_subindex(prediction)

    return {
        "predicted_pm25_next_hour": round(prediction, 2),
        "indicative_pm25_subindex": subindex,
        "category": category,
        "official_aqi_note": (
            "This is an indicative PM2.5 sub-index/category only. Official PM2.5 AQI "
            "uses the applicable 24-hour concentration; this project forecasts one hour ahead."
        ),
        "model": MODEL_BUNDLE.get("best_model_name", "Random Forest"),
        "target": MODEL_BUNDLE.get("target", "PM2.5 one hour ahead"),
        "split": MODEL_BUNDLE.get("split"),
    }


# Week 8: next-day and next-week hourly PM2.5 forecasts
from functools import lru_cache
from fastapi.responses import JSONResponse
import json

WEEK8_MODELS = ROOT / "models"
SAMPLE_PATH = ROOT / "samples" / "sample_history.json"

class SequenceInput(BaseModel):
    history: list[float | None] = Field(min_length=168, max_length=168)
    station: str = Field(default="BARC")
    origin: str
    horizon: int = Field(default=24)
    method: str = Field(default="Gradient boosting")


@lru_cache(maxsize=2)
def load_forecast_model(horizon: int):
    path = WEEK8_MODELS / f"pm25_{horizon}h_gradient_boosting.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if bundle.get('horizon') != horizon or bundle.get('context_hours') != 168:
        raise ValueError('Saved forecast model has incompatible horizon or history length')
    return bundle['model']


@app.get("/api/forecast-status")
def forecast_status():
    return {"available_methods": ["Persistence", "Daily repeat"] +
            (["Gradient boosting"] if all((WEEK8_MODELS / f"pm25_{h}h_gradient_boosting.joblib").exists() for h in (24, 168)) else []),
            "horizons": [24, 168], "history_hours": 168,
            "model_24h_loaded": (WEEK8_MODELS / "pm25_24h_gradient_boosting.joblib").exists(),
            "model_168h_loaded": (WEEK8_MODELS / "pm25_168h_gradient_boosting.joblib").exists()}


@app.get("/api/sample-history")
def sample_history():
    if not SAMPLE_PATH.exists():
        raise HTTPException(status_code=404, detail='Example history not available')
    return json.loads(SAMPLE_PATH.read_text(encoding='utf8'))


@app.post("/api/forecast-sequence")
def forecast_sequence(payload: SequenceInput):
    if payload.horizon not in (24, 168):
        raise HTTPException(status_code=422, detail="Forecast must be 24 or 168 hours")
    station = payload.station.strip()
    if station not in ("BARC", "Darussalam", "DoE"):
        raise HTTPException(status_code=422, detail="Select BARC, Darussalam or DoE")
    try:
        origin = pd.Timestamp(payload.origin)
        if pd.isna(origin) or origin.tzinfo is not None:
            raise ValueError('Use local time without a timezone suffix')
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Enter a valid local observation timestamp") from exc
    if origin.hour != 23 or origin.minute != 0 or origin.second != 0:
        raise HTTPException(status_code=422, detail="Forecast origin must be at 23:00 (last hour of the day)")
    raw = np.array([np.nan if x is None else x for x in payload.history], dtype=float)
    if np.any(np.isfinite(raw) & ((raw < 0) | (raw > 1000))):
        raise HTTPException(status_code=422, detail="PM2.5 readings must be between 0 and 1000")
    if np.mean(np.isfinite(raw)) < .7 or not np.isfinite(raw[-1]):
        raise HTTPException(status_code=422, detail="At least 70% of the last 168 hours and the latest hour must have a reading")
    history = pd.Series(raw).ffill().bfill().to_numpy(dtype=float)
    if payload.method == 'Persistence':
        forecast = np.repeat(history[-1], payload.horizon)
    elif payload.method == 'Daily repeat':
        forecast = np.tile(history[-24:], int(np.ceil(payload.horizon / 24)))[:payload.horizon]
    elif payload.method == 'Gradient boosting':
        try:
            model = load_forecast_model(payload.horizon)
            if model is None:
                raise HTTPException(status_code=503, detail='Run train_evaluate.py to train the Week 8 model')
            from src.forecast_core import forecast_ml
            forecast = forecast_ml(model, history, station, origin, payload.horizon)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail='Forecast model could not be loaded') from exc
    else:
        raise HTTPException(status_code=422, detail='Unknown forecasting method')
    return {"station":station,"origin":str(origin),"horizon_hours":payload.horizon,
            "method":payload.method, "unit":"µg/m³", "forecast_pm25":[round(float(v),2) for v in np.maximum(0,forecast)],
            "timestamps": [(origin+pd.Timedelta(hours=i)).isoformat() for i in range(1,payload.horizon+1)],
            "note":"Hourly PM2.5 forecasts. Not official AQI."}
