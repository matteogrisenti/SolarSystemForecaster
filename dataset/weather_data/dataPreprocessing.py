import pandas as pd
import numpy as np
import re
from io import StringIO
from pathlib import Path

def preprocess_hourly_data(input_csv, output_csv):
    """
    Preprocess a single weather CSV file:
    - Detect data start
    - Parse datetime and main value column
    - Convert values to numeric
    - Aggregate intra-hour duplicates
    - Remove missing/problematic hours
    - Round values
    - Save cleaned data with dynamic column name
    """
    try:
        # --- Step 1: Safe file reading with encoding fallback ---
        try:
            with open(input_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(input_csv, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        # Find first line that looks like "hh:mm:ss dd/mm/yyyy"
        pattern = re.compile(r'^\d{2}:\d{2}:\d{2}\s+\d{2}/\d{2}/\d{4}')
        start_idx = None
        for i, line in enumerate(lines):
            if pattern.match(line.strip()):
                start_idx = i
                break

        if start_idx is None:
            print(f"⚠️  Skipping {input_csv.name} — No valid data start found.")
            return

        data_lines = lines[start_idx:]
        csv_data = StringIO(''.join(data_lines))
        
        # --- Step 2: Load using pandas ---
        df = pd.read_csv(
            csv_data,
            header=None,
            names=["datetime", "value", "quality", "note"],
            on_bad_lines='skip',
            dtype={"note": str},
            low_memory=False
        )

        # --- Step 3: Parse datetime and clean ---
        df["datetime"] = pd.to_datetime(df["datetime"], format="%H:%M:%S %d/%m/%Y", errors='coerce')
        df = df.dropna(subset=["datetime"])

        # --- Step 4: Convert value to numeric safely ---
        df["value"] = (
            df["value"]
            .astype(str)             # ensure string for cleaning
            .str.replace(",", ".", regex=False)  # convert commas to dots
            .str.extract(r"([-+]?\d*\.?\d+)")[0] # extract only numeric pattern
        )
        df["value"] = pd.to_numeric(df["value"], errors='coerce')

        # Remove invalid values
        df = df.dropna(subset=["value"])

        # --- Step 5: Merge intra-hour duplicates ---
        df["hour"] = df["datetime"].dt.floor("h")
        df_hourly = df.groupby("hour", as_index=False)["value"].mean()

        # --- Step 6: Detect missing or invalid hours ---
        start, end = df_hourly["hour"].min(), df_hourly["hour"].max()
        theoretical_range = pd.date_range(start=start, end=end, freq="h")
        theoretical_total = len(theoretical_range)
        
        df_hourly = df_hourly.set_index("hour").reindex(theoretical_range)
        missing_hours = df_hourly[df_hourly.isna().any(axis=1)]
        removed_count = len(missing_hours)
        df_clean = df_hourly.dropna().reset_index().rename(columns={"index": "hour", "value": "data"})

        # --- Step 7: Round numeric values ---
        df_clean["data"] = df_clean["data"].round(2)

        # --- Step 8: Rename column dynamically ---
        column_name = input_csv.stem  # file name without .csv
        df_clean = df_clean.rename(columns={"data": column_name})

        # --- Step 9: Summary ---
        initial_count = len(df_hourly)
        remaining_count = len(df_clean)

        print(f"\n📄 File: {input_csv.name}")
        print("==== Preprocessing Summary ====")
        print(f"Theoretical total hours: {theoretical_total}")
        print(f"Initial detected hours:  {initial_count}")
        print(f"Removed problematic:     {removed_count}")
        print(f"Remaining valid samples: {remaining_count}")
        print("================================")

        # --- Step 10: Save result ---
        df_clean.to_csv(output_csv, index=False)
        print(f"✅ Saved cleaned file: {output_csv}")

    except Exception as e:
        print(f"❌ Error processing {input_csv.name}: {e}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    raw_data_directory = Path("./raw")
    preprocessed_data_directory = Path("./processed")

    preprocessed_data_directory.mkdir(parents=True, exist_ok=True)

    print("🚀 START WEATHER DATASET PREPROCESSING")
    print("=" * 70)
    print(f"📁 Input Directory:  {raw_data_directory.resolve()}")
    print(f"💾 Output Directory: {preprocessed_data_directory.resolve()}")
    print("=" * 70)

    csv_files = list(raw_data_directory.glob("*.csv"))

    if not csv_files:
        print("⚠️  No CSV files found in input directory.")
    else:
        for file in csv_files:
            output_path = preprocessed_data_directory  / f"{file.stem}_cleaned.csv"
            preprocess_hourly_data(file, output_path)

    print("\n🎉 All files processed successfully!")
