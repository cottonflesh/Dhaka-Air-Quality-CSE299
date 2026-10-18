# Dhaka Air Quality Prediction System

CSE299 project focused on preparing Dhaka air-quality and meteorological data for a machine-learning based prediction web application.

## Team

**Course:** CSE299  
**Section:** 14

| Member | Student ID |
|---|---|
| Nizam Uddin Nihon | 2031139342 |
| Ikramun Masud Lubaba | 2232451042 |
| Md Huzayfa | 2222948642 |

## Current project status

The repository now includes work through **Week 5**: raw-data inspection, Dhaka extraction, cleaning/standardization, a baseline model, and the first Random Forest model for PM2.5 prediction. Model comparison and deeper error analysis are planned for Week 6.

### Work completed
- Collected the original CAMS-BD Mendeley dataset.
- Inspected all four raw workbooks: 2012-2021, 2022, 2023, and 2024.
- Identified usable Dhaka monitoring stations: BARC, Darussalam, and DoE.
- Standardized changing column names and column order across yearly files.
- Parsed and standardized timestamps.
- Converted pollutant and meteorological columns to numeric values.
- Treated invalid/out-of-range sensor values as missing instead of silently keeping them.
- Removed duplicate station-timestamp records.
- Produced one chronological cleaned Dhaka dataset for 2012-2024.
- Generated audit files and visual evidence for the progress update.
- Built a Week 5 baseline using the training-set mean PM2.5.
- Trained the first Random Forest regression model and recorded held-out metrics.
- Saved Week 5 code, predictions, feature importance, evaluation tables, and the fitted model artifact.

## Dataset summary

| Item | Current value |
|---|---:|
| Cleaned Dhaka rows | 206,271 |
| Stations | 3 |
| Start | 2012-11-01 00:00:00 |
| End | 2024-12-31 23:00:00 |
| PM2.5 available rows | 162,685 |
| PM10 available rows | 147,432 |
| Temperature available rows | 132,013 |
| Humidity available rows | 146,928 |
| Wind-speed available rows | 125,821 |

## Main variables

`PM2.5`, `PM10`, `Temperature`, `Humidity`, `WindSpeed`, `WindDirection`, `CO`, `NO`, `NO2`, `NOX`, `SO2`, `O3`, `Pressure`, `SolarRadiation`, `Rain`, `DateTime`, and `Location`.

## Repository structure

```text
Dhaka-Air-Quality-CSE299/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/                    # raw files are downloaded separately
│   ├── processed/              # cleaned Dhaka dataset
│   └── audit/                  # data-audit summaries
├── src/
│   ├── clean_dhaka_data.py
│   └── summarize_cleaned_data.py
├── reports/
│   ├── Progress_Update_Report.md
│   ├── Progress_Update_Report.docx
│   └── Dhaka_Air_CSE299_Project_Proposal.pdf
├── results/
│   ├── cleaned_dataset_preview.csv
│   ├── raw_file_audit.png
│   └── station_data_availability.png
├── week5/
│   ├── README.md
│   ├── src/
│   ├── results/
│   ├── models/
│   └── reports/
└── presentation/
    ├── Dhaka_Air_Track3_Project_Proposal.pptx
    └── Dhaka_Air_Project_Proposal_Final.pptx
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Recreate the cleaned dataset

1. Download CAMS-BD from Mendeley.
2. Put the downloaded ZIP in `data/raw/`.
3. Run:

```bash
python src/clean_dhaka_data.py \
  --input "data/raw/CAMS-BD.zip" \
  --output "data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv" \
  --audit-dir "data/audit"
```

The script can also take an extracted directory instead of a ZIP.

## Generate a quick cleaned-data summary

```bash
python src/summarize_cleaned_data.py \
  --input "data/processed/Dhaka_CAMS_Cleaned_2012_2024.csv"
```

## Next development stage

The next milestone is **Week 6 model comparison and error analysis**. Additional models can then be compared before choosing the model to integrate into the web application.

## Data credit

CAMS-BD dataset, Mendeley Data: https://data.mendeley.com/datasets/hwmzyh5w4w/1
