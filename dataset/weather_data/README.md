# 🌦️ Weather Data Preprocessing & Merging Pipeline

This repository contains a **Python-based data preprocessing pipeline** for transforming raw meteorological CSV files (downloade on http://storico.meteotrentino.it/web.htm ) into a clean, hourly-aligned, unified dataset.

The pipeline performs:
1. **Cleaning & normalization** of raw station data.
2. **Hourly aggregation** (one value per hour).
3. **Detection & removal** of corrupted or missing data.
4. **Merging** of all preprocessed variables into a single dataset, keeping only valid common hours.

---

## 📁 Project Structure

```bash
weather_pipeline/
│
├── raw/ # 📥 Original raw CSV files
├── processed/ # ⚙️ Cleaned, hourly-aligned outputs
│
├── weather_data.csv # ✅ Unified dataset (one row per hour)
│
├── dataPreprocessing.py # Script 1: Clean & preprocess raw files
├── dataMerge.py # Script 2: Merge all preprocessed files
└── README.md # 📘 Documentation (this file)
```


## ⚙️ 1️⃣ Preprocessing Raw Data

Script: dataPreprocessing.py

This script processes all .csv files inside raw_weather_csv/ and:
- Detects where the actual data section starts (skipping metadata headers).
- Parses timestamps (format: HH:MM:SS DD/MM/YYYY).
- Aggregates intra-hour readings → one mean value per hour.
- Detects and removes missing or invalid hours.
- Rounds numeric values to 2 decimal places.

Saves a clean file named <original>_preprocessed.csv inside preprocessed_weather_csv/.

Usage:
```bash
python dataPreprocessing.py
```


## 📊 2️⃣ Merging All Cleaned Data

Script: dataMerge.py

This script loads all hourly-aligned CSVs from preprocessed_weather_csv/ and:
- Merges them on the hour column.
- Keeps only hours where all variables have valid (non-missing) values.
- Outputs a single dataset final_weather_dataset.csv.

Usage:
```bash
python dataMerge.py
```

## 🧾 Summary of the Pipeline
| Step | Script                       | Description                            | Output Folder               |
| ---- | ---------------------------- | -------------------------------------- | --------------------------- |
| 1️⃣  | `dataPreprocessing.py`       | Cleans and aligns each raw sensor file | `processed/` |
| 2️⃣  | `merge_preprocessed_data.py` | Merges all preprocessed files by hour  | `weather_data.csv` |


## 📈 Final Dataset Structure
| hour                | humidity | irradiance | pressure | rain | temperature | wind_speed | wind_direction |
| ------------------- | -------- | ---------- | -------- | ---- | ----------- | ---------- | -------------- |
| 2003-01-01 00:00:00 | 100.0    |            |          |      | 12.45       | 2.31       | 180.0          |
| 2003-01-01 01:00:00 | 98.5     |            |          |      | 12.12       | 2.10       | 190.0          |
| ...                 | ...      | ...        | ...      | ...  | ...         | ...        | ...            |
