# WEEK9_DASHBOARD_API
# Week 9 historical analytics endpoints. Uses existing Week 8 imports and paths.
DASHBOARD_DATA_PATH = ROOT.parent / "data" / "processed" / "Dhaka_CAMS_Cleaned_2012_2024.csv"
DASHBOARD_OUTPUTS = ROOT / "outputs"
DASHBOARD_STATIONS = ("BARC", "Darussalam", "DoE")


@lru_cache(maxsize=1)
def load_dashboard_frame() -> pd.DataFrame:
    if not DASHBOARD_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DASHBOARD_DATA_PATH}")
    columns = ["Location", "DateTime", "PM2.5", "PM10", "Temperature", "Humidity"]
    frame = pd.read_csv(DASHBOARD_DATA_PATH, usecols=columns, low_memory=False)
    aliases = {
        "BARC (Farmgate)": "BARC",
        "Darussalam (Mirpur)": "Darussalam",
        "DoE (Agargaon)": "DoE",
        "DOE": "DoE",
    }
    frame["Location"] = frame["Location"].astype(str).str.strip().replace(aliases)
    frame = frame[frame["Location"].isin(DASHBOARD_STATIONS)].copy()
    frame["DateTime"] = pd.to_datetime(frame["DateTime"], errors="coerce")
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["DateTime"]).sort_values("DateTime")


def _dashboard_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 2)
    return value


@app.get("/week9")
def week9_dashboard():
    return FileResponse(ROOT / "templates" / "week9_dashboard.html")


@app.get("/api/dashboard/summary")
def dashboard_summary(station: str = "All", start_year: int = 2012, end_year: int = 2024):
    if station != "All" and station not in DASHBOARD_STATIONS:
        raise HTTPException(status_code=422, detail="Select All, BARC, Darussalam or DoE")
    if start_year < 2012 or end_year > 2024 or start_year > end_year:
        raise HTTPException(status_code=422, detail="Choose a year range within 2012–2024")
    try:
        frame = load_dashboard_frame()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard dataset unavailable: {exc}") from exc

    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year + 1}-01-01")
    frame = frame[(frame["DateTime"] >= start) & (frame["DateTime"] < end)]
    selected = frame if station == "All" else frame[frame["Location"] == station]
    measured = selected.dropna(subset=["PM2.5"])
    latest = measured.tail(1)
    latest_record = None
    if not latest.empty:
        row = latest.iloc[0]
        latest_record = {
            "station": str(row["Location"]),
            "datetime": row["DateTime"].isoformat(),
            "pm25": _dashboard_value(row["PM2.5"]),
            "pm10": _dashboard_value(row["PM10"]),
            "temperature": _dashboard_value(row["Temperature"]),
            "humidity": _dashboard_value(row["Humidity"]),
        }

    station_rows = []
    for name in DASHBOARD_STATIONS:
        group = frame[frame["Location"] == name]
        pm = group.dropna(subset=["PM2.5"])
        last = pm.tail(1)
        station_rows.append({
            "station": name,
            "records": int(len(group)),
            "pm25_observed": int(len(pm)),
            "pm25_coverage_pct": round(float(group["PM2.5"].notna().mean() * 100), 2) if len(group) else 0,
            "latest_pm25": _dashboard_value(last.iloc[0]["PM2.5"]) if not last.empty else None,
            "latest_datetime": last.iloc[0]["DateTime"].isoformat() if not last.empty else None,
        })
    return {
        "station_filter": station,
        "record_count": int(len(selected)),
        "pm25_observed": int(len(measured)),
        "station_count": int(selected["Location"].nunique()),
        "period_start": selected["DateTime"].min().date().isoformat() if not selected.empty else None,
        "period_end": selected["DateTime"].max().date().isoformat() if not selected.empty else None,
        "latest_observation": latest_record,
        "stations": station_rows,
        "data_note": "Historical CAMS-BD measurements in the cleaned dataset; not live monitoring.",
    }


@app.get("/api/dashboard/history")
def dashboard_history(station: str = "All", start_year: int = 2012,
                      end_year: int = 2024, frequency: str = "month"):
    if station != "All" and station not in DASHBOARD_STATIONS:
        raise HTTPException(status_code=422, detail="Select All, BARC, Darussalam or DoE")
    if start_year < 2012 or end_year > 2024 or start_year > end_year:
        raise HTTPException(status_code=422, detail="Choose a year range within 2012–2024")
    if frequency not in ("month", "day"):
        raise HTTPException(status_code=422, detail="Frequency must be month or day")
    try:
        frame = load_dashboard_frame()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard dataset unavailable: {exc}") from exc

    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year + 1}-01-01")
    selected = frame[(frame["DateTime"] >= start) & (frame["DateTime"] < end)]
    if station != "All":
        selected = selected[selected["Location"] == station]
    if selected.empty:
        return {"labels": [], "pm25": [], "pm10": [], "frequency": frequency, "station": station}

    rule = "MS" if frequency == "month" else "D"
    grouped = selected.set_index("DateTime").resample(rule).agg({
        "PM2.5": "mean", "PM10": "mean"
    }).dropna(how="all")
    return {
        "labels": [stamp.strftime("%Y-%m" if frequency == "month" else "%Y-%m-%d") for stamp in grouped.index],
        "pm25": [_dashboard_value(v) for v in grouped["PM2.5"]],
        "pm10": [_dashboard_value(v) for v in grouped["PM10"]],
        "frequency": frequency,
        "station": station,
    }


@app.get("/api/dashboard/model-metrics")
def dashboard_model_metrics():
    path = DASHBOARD_OUTPUTS / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Week 8 model_comparison.csv is missing")
    metrics = pd.read_csv(path)
    return {
        "records": metrics.to_dict(orient="records"),
        "source": "week8/outputs/model_comparison.csv",
        "note": "Validation is 2023; test is 2024. Errors are PM2.5 concentration errors in µg/m³.",
    }


@app.get("/api/dashboard/error-analysis")
def dashboard_error_analysis(horizon: int = 24):
    if horizon not in (24, 168):
        raise HTTPException(status_code=422, detail="Horizon must be 24 or 168 hours")
    path = DASHBOARD_OUTPUTS / f"lead_hour_errors_{horizon}h.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Week 8 {horizon}-hour error analysis is missing")
    errors = pd.read_csv(path)
    errors = errors[(errors["Split"] == "test") & (errors["Model"] == "Gradient boosting")]
    return {
        "horizon": horizon,
        "records": errors.to_dict(orient="records"),
        "source": f"week8/outputs/lead_hour_errors_{horizon}h.csv",
        "note": "Per-lead-hour test errors for the Gradient Boosting forecast model.",
    }
