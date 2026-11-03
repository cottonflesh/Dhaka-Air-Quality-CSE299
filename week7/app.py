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
MODEL_PATH = ROOT / "models" / "Week6_Best_NextHour_PM25_Model.joblib"
INDEX_PATH = ROOT / "templates" / "index.html"

app = FastAPI(title="Dhaka Air Quality Predictor", version="week7")

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
