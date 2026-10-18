#!/usr/bin/env python3
"""Print a quick status summary of the cleaned Dhaka dataset."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    print(f"Rows: {len(df):,}")
    print(f"Stations: {df['Location'].nunique()}")
    print(f"Start: {df['DateTime'].min()}")
    print(f"End: {df['DateTime'].max()}")
    print("\nRows by station:")
    print(df["Location"].value_counts().to_string())

    fields = ["PM2.5", "PM10", "Temperature", "Humidity", "WindSpeed", "WindDirection"]
    print("\nAvailability:")
    for field in fields:
        available = int(df[field].notna().sum())
        pct = 100 * available / len(df)
        print(f"{field:15s}: {available:>7,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
