#!/usr/bin/env python3
"""Clean the CAMS-BD raw files and build one Dhaka-only CSV.

Usage:
    python src/clean_dhaka_data.py \
        --input data/raw/CAMS-BD.zip \
        --output data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv \
        --audit-dir data/audit

The input may be the outer Mendeley ZIP, the inner CAMS-BD ZIP, or a directory
containing the four source XLSX files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from lxml import etree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
BASE_DATE = datetime(1899, 12, 30)

# Current project scope: Dhaka stations with usable timestamps.
DHAKA_STATIONS = {"BARC", "Darussalam", "DoE"}

STD_FIELDS = [
    "SO2", "NO", "NO2", "NOX", "CO", "O3", "PM2.5", "PM10",
    "WindSpeed", "WindDirection", "Temperature", "Humidity", "Pressure",
    "SolarRadiation", "Rain", "VerticalWindSpeed",
]

ALIASES = {
    "SO2": "SO2", "NO": "NO", "NO2": "NO2", "NOX": "NOX",
    "CO": "CO", "O3": "O3", "PM2.5": "PM2.5", "PM10": "PM10",
    "Wind Speed": "WindSpeed", "WS": "WindSpeed",
    "Wind Dir": "WindDirection", "WD": "WindDirection",
    "Temperature": "Temperature", "Temp": "Temperature",
    "RH": "Humidity", "BP": "Pressure",
    "Solar Rad": "SolarRadiation", "SR": "SolarRadiation",
    "Rain": "Rain", "V Wind Speed": "VerticalWindSpeed", "VWS": "VerticalWindSpeed",
}

BOUNDS = {
    "SO2": (0, 1000), "NO": (0, 1500), "NO2": (0, 1000), "NOX": (0, 2000),
    "CO": (0, 100), "O3": (0, 1000), "PM2.5": (0, 1500), "PM10": (0, 2000),
    "WindSpeed": (0, 60), "WindDirection": (0, 360), "Temperature": (-20, 60),
    "Humidity": (0, 100), "Pressure": (850, 1100), "SolarRadiation": (0, 1600),
    "Rain": (0, 300), "VerticalWindSpeed": (-50, 50),
}

OUTPUT_COLUMNS = ["Location", "DateTime"] + STD_FIELDS + ["SourceYear"]


def safe_float(value):
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in {"na", "nan", "n/a", "none", "null", "-", "--"}:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def excel_datetime(value):
    value = safe_float(value)
    if value is None:
        return None
    try:
        return BASE_DATE + timedelta(days=value)
    except (OverflowError, ValueError):
        return None


def parse_old_timestamp(date_raw, time_raw):
    """Handle the mixed 2012-2021 timestamp format used in the source file."""
    value = safe_float(date_raw)
    if value is None:
        return None

    # Some later rows in the old workbook encode the hour directly in the serial.
    if abs(value - math.floor(value)) > 1e-5:
        dt = excel_datetime(value)
        return dt.replace(second=0, microsecond=0) if dt else None

    base = excel_datetime(math.floor(value))
    if base is None:
        return None

    label = str(time_raw or "").strip()
    match = re.match(r"^(\d{1,2}):(\d{2})", label)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    # Source labels are commonly hour-ending labels (01:00 means interval starting 00:00).
    if hour == 24 and minute == 0:
        hour = 23
    elif 1 <= hour <= 23:
        hour -= 1
    elif hour != 0:
        return None

    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def clean_numeric(field, value, quality_counter):
    value = safe_float(value)
    if value is None:
        return None
    if field in BOUNDS:
        lo, hi = BOUNDS[field]
        if not (lo <= value <= hi):
            quality_counter[f"invalid_{field}"] += 1
            return None
    return value


def col_number(cell_ref: str) -> int:
    n = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def shared_strings(zf: zipfile.ZipFile):
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join((t.text or "") for t in si.iter(NS + "t")) for si in root.iter(NS + "si")]


def cell_value(cell, shared):
    typ = cell.get("t")
    if typ == "inlineStr":
        t = cell.find(".//" + NS + "t")
        return "" if t is None else (t.text or "")
    v = cell.find(NS + "v")
    if v is None:
        return None
    raw = v.text
    if typ == "s":
        try:
            return shared[int(raw)]
        except Exception:
            return raw
    return raw


def extract_nested_zip_or_dir(input_path: Path, work_dir: Path) -> list[Path]:
    """Return all CAMS-BD xlsx files from a directory or nested ZIP download."""
    if input_path.is_dir():
        return sorted(input_path.rglob("*.xlsx"))

    if input_path.suffix.lower() != ".zip":
        raise ValueError("Input must be a directory or ZIP file.")

    first = work_dir / "outer"
    first.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path) as zf:
        zf.extractall(first)

    xlsx = sorted(first.rglob("*.xlsx"))
    if xlsx:
        return xlsx

    nested = list(first.rglob("*.zip"))
    if not nested:
        raise FileNotFoundError("No XLSX files or nested ZIP found in the input archive.")

    second = work_dir / "inner"
    second.mkdir(parents=True, exist_ok=True)
    for zpath in nested:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(second)

    xlsx = sorted(second.rglob("*.xlsx"))
    if not xlsx:
        raise FileNotFoundError("No CAMS-BD XLSX files found after extracting the nested ZIP.")
    return xlsx


def extract_workbook(path: Path):
    file_name = path.name
    old_format = "2012-2021" in file_name
    records = []
    stats = Counter()
    station_stats = defaultdict(Counter)
    quality = Counter()

    with zipfile.ZipFile(path) as zf:
        shared = shared_strings(zf)
        with zf.open("xl/worksheets/sheet1.xml") as stream:
            header_map = None
            for _, row in ET.iterparse(stream, events=("end",), tag=NS + "row", huge_tree=True):
                row_number = int(row.get("r", "0"))
                cells = [c for c in row if c.tag == NS + "c"]

                if row_number == 1:
                    raw_headers = {}
                    for c in cells:
                        raw_headers[col_number(c.get("r", "A1"))] = (cell_value(c, shared) or "").strip()
                    header_map = {name: index for index, name in raw_headers.items() if name}
                    row.clear()
                    continue

                stats["raw_rows"] += 1
                if not cells or col_number(cells[0].get("r", "A1")) != 0:
                    row.clear()
                    continue

                location = (cell_value(cells[0], shared) or "").strip()
                if not location:
                    row.clear()
                    continue

                station_stats[location]["raw_rows"] += 1
                if location not in DHAKA_STATIONS:
                    row.clear()
                    continue

                stats["selected_dhaka_rows"] += 1
                values = {0: location}
                for c in cells[1:]:
                    values[col_number(c.get("r", "A1"))] = cell_value(c, shared)

                if old_format:
                    timestamp = parse_old_timestamp(
                        values.get(header_map.get("Date", -999)),
                        values.get(header_map.get("Time", -999)),
                    )
                else:
                    raw_dt = values.get(header_map.get("Date & Time", -999))
                    timestamp = excel_datetime(raw_dt)
                    if timestamp:
                        timestamp = timestamp.replace(second=0, microsecond=0)

                if timestamp is None or not (2012 <= timestamp.year <= 2024):
                    stats["invalid_or_out_of_scope_timestamp"] += 1
                    station_stats[location]["invalid_timestamp"] += 1
                    row.clear()
                    continue

                rec = {
                    "Location": location,
                    "DateTime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "SourceYear": "2012-2021" if old_format else str(timestamp.year),
                }

                for source_header, index in header_map.items():
                    field = ALIASES.get(source_header)
                    if field:
                        rec[field] = clean_numeric(field, values.get(index), quality)
                for field in STD_FIELDS:
                    rec.setdefault(field, None)

                records.append(rec)
                station_stats[location]["valid_timestamp_rows"] += 1
                if rec["PM2.5"] is not None:
                    station_stats[location]["pm25_present"] += 1
                row.clear()

    return records, stats, station_stats, quality


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Outer/inner CAMS-BD ZIP or extracted directory")
    parser.add_argument("--output", required=True, help="Output cleaned CSV")
    parser.add_argument("--audit-dir", required=True, help="Directory for audit CSV/JSON files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cams_bd_") as tmp:
        work_dir = Path(tmp)
        xlsx_files = extract_nested_zip_or_dir(input_path, work_dir)

        all_records = []
        inventory = []
        station_rows = []
        quality_total = Counter()

        for path in xlsx_files:
            records, stats, station_stats, quality = extract_workbook(path)
            all_records.extend(records)
            quality_total.update(quality)
            inventory.append({
                "File": path.name,
                "Raw_Rows": stats["raw_rows"],
                "Selected_Dhaka_Rows": stats["selected_dhaka_rows"],
                "Valid_2012_2024_Rows": len(records),
                "Invalid_or_Out_of_Scope_Timestamp_Rows": stats["invalid_or_out_of_scope_timestamp"],
            })
            for location, counter in station_stats.items():
                if location in DHAKA_STATIONS:
                    station_rows.append({"File": path.name, "Location": location, **dict(counter)})

        # Deduplicate by station and timestamp, keeping the row with the most measurements.
        by_key = {}
        duplicates_removed = 0
        for rec in all_records:
            key = (rec["Location"], rec["DateTime"])
            score = sum(rec.get(field) is not None for field in STD_FIELDS)
            if key not in by_key:
                by_key[key] = (score, rec)
            else:
                duplicates_removed += 1
                if score > by_key[key][0]:
                    by_key[key] = (score, rec)

        cleaned = [item[1] for item in by_key.values()]
        cleaned.sort(key=lambda r: (r["Location"], r["DateTime"]))
        write_csv(output_path, cleaned, OUTPUT_COLUMNS)

        write_csv(audit_dir / "Raw_File_Inventory.csv", inventory, list(inventory[0].keys()))
        if station_rows:
            fields = ["File", "Location", "raw_rows", "valid_timestamp_rows", "invalid_timestamp", "pm25_present"]
            write_csv(audit_dir / "Raw_Dhaka_Station_Audit.csv", station_rows, fields)

        station_summary = []
        grouped = defaultdict(list)
        for rec in cleaned:
            grouped[rec["Location"]].append(rec)
        for location, rows in sorted(grouped.items()):
            station_summary.append({
                "Location": location,
                "Rows": len(rows),
                "First_Timestamp": rows[0]["DateTime"],
                "Last_Timestamp": rows[-1]["DateTime"],
                "PM25_Available": sum(r["PM2.5"] is not None for r in rows),
                "PM10_Available": sum(r["PM10"] is not None for r in rows),
            })
        write_csv(
            audit_dir / "Cleaned_Station_Summary.csv",
            station_summary,
            ["Location", "Rows", "First_Timestamp", "Last_Timestamp", "PM25_Available", "PM10_Available"],
        )

        summary = {
            "cleaned_rows": len(cleaned),
            "stations": sorted(grouped),
            "duplicates_removed": duplicates_removed,
            "invalid_values_set_to_missing": dict(quality_total),
            "note": "Sangsad is not included in the current cleaned dataset because the older source block lacks usable timestamps for this workflow.",
        }
        with (audit_dir / "Cleaning_Summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Cleaned dataset written to: {output_path}")
    print(f"Rows: {len(cleaned):,}")
    print(f"Audit files written to: {audit_dir}")


if __name__ == "__main__":
    main()
