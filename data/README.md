# Data

## Source
The project uses the **CAMS-BD: Hourly Air Quality and Meteorological Measurements Dataset** from Mendeley Data:

https://data.mendeley.com/datasets/hwmzyh5w4w/1

The original download contains four Excel workbooks covering 2012-2021, 2022, 2023, and 2024.

## Why the raw dataset is not in this repository
The original downloaded ZIP is larger than GitHub's normal 100 MB per-file limit. Download it from Mendeley and place it in:

`data/raw/`

The cleaning script can accept either the outer Mendeley ZIP or an extracted folder containing the four `.xlsx` files.

## Current processed dataset
`processed/Dhaka_CAMS_Cleaned_2012_2024.csv`

Current scope:
- Dhaka stations: BARC, Darussalam, DoE
- Period: 2012-11-01 to 2024-12-31
- Rows after current cleaning: 206,271
- Main variables: PM2.5, PM10, temperature, humidity, wind speed/direction, gaseous pollutants, pressure, solar radiation, rainfall, and timestamp

`Sangsad` is not included in the current cleaned dataset because the older source block does not provide usable timestamps for the time-based analysis.
