# CSE299 Project Progress Update

## Project: Dhaka Air Quality Prediction System

**Course:** CSE299  
**Section:** 14  
**Team Members:**  
- Nizam Uddin Nihon — 2031139342  
- Ikramun Masud Lubaba — 2232451042  
- Md Huzayfa — 2222948642

### 1. Current objective
The current development stage focuses on converting the original CAMS-BD raw dataset into a reliable Dhaka-specific dataset that can later be used for machine-learning model development and the web application.

### 2. Dataset source
The project uses the CAMS-BD hourly air-quality and meteorological dataset from Mendeley Data. The raw package contains yearly workbooks covering 2012-2021, 2022, 2023, and 2024.

### 3. Work completed
- Opened and inspected all four original workbooks.
- Identified the usable Dhaka stations: BARC, Darussalam, and DoE.
- Selected the main pollution and meteorological fields required by the project.
- Standardized inconsistent yearly headers and changing column order.
- Parsed and standardized date/time values.
- Converted measurement fields to numeric form.
- Treated invalid/out-of-range sensor readings as missing.
- Removed duplicate station-timestamp records.
- Sorted the cleaned data chronologically.
- Generated audit tables and charts as evidence of the cleaning process.

### 4. Current cleaned dataset
- Cleaned rows: **206,271**
- Stations: **3**
- Time range: **2012-11-01 to 2024-12-31**
- PM2.5 available rows: **162,685**
- PM10 available rows: **147,432**
- Temperature available rows: **132,013**
- Humidity available rows: **146,928**
- Wind-speed available rows: **125,821**

The cleaned dataset contains PM2.5, PM10, temperature, humidity, wind speed/direction, CO, NO, NO2, NOX, SO2, O3, pressure, solar radiation, rainfall, timestamp, and location fields.

### 5. Data-quality observations
The raw dataset is not uniform across all years. Column names and column positions change between files, some values are missing, and some station blocks have incomplete time information. The cleaning pipeline therefore standardizes fields by their names instead of assuming a fixed Excel column position.

The Sangsad station is not included in the current cleaned dataset because its older source records do not provide usable timestamps for the time-based project workflow.

### 6. Evidence produced
The repository includes:
- raw file inventory;
- Dhaka station audit;
- cleaned station summary;
- cleaned-data preview;
- raw-file audit chart; and
- station data-availability chart.

### 7. Current status
The raw-to-clean data preparation stage is complete enough to begin model development. No final model-performance claim is being made in this update.

### 8. Next work
The next stage will prepare the ML input features and prediction target, create the first baseline/model experiments, compare results, and then begin integrating the prediction workflow into the web application.
