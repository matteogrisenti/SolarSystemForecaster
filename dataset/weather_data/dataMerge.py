import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

def merge_preprocessed_data(input_dir, output_file):
    """
    Merge all preprocessed hourly CSVs into a single dataset.
    Keeps only hours where all sensors have valid data.
    Keeps hour, hour_number, day, and month only once.
    """
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print("⚠️  No preprocessed CSV files found.")
        return

    print("🚀 START MERGING PREPROCESSED DATASETS")
    print("=" * 70)
    print(f"📁 Input Directory:  {input_dir.resolve()}")
    print(f"💾 Output File:      {Path(output_file).resolve()}")
    print("=" * 70)

    merged_df = None
    all_columns = []

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)

        if "datetime" not in df.columns:
            print(f"⚠️  Skipping {file.name} — missing 'datetime' column.")
            continue

        # Ensure proper datetime
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Identify the main sensor column (the data column)
        data_cols = [c for c in df.columns if c not in ["datetime", "hour", "day", "month"]]
        if not data_cols:
            print(f"⚠️  Skipping {file.name} — no sensor data column found.")
            continue

        data_col = data_cols[0]
        all_columns.append(data_col)
        print(f"📄 Loaded {file.name} with variable: {data_col} ({len(df)} rows)")

        # For the first file, keep everything
        if merged_df is None:
            merged_df = df
        else:
            # For subsequent files, drop the repeated date component columns before merging
            df = df.drop(columns=["hour", "day", "month"], errors="ignore")
            merged_df = pd.merge(merged_df, df, on="datetime", how="inner")

    if merged_df is None or merged_df.empty:
        print("❌ No valid data after merging.")
        return

    # Drop any incomplete rows
    theoretical_total = merged_df.shape[0]
    merged_df = merged_df.dropna()
    final_count = merged_df.shape[0]
    removed = theoretical_total - final_count

    # --- Print summary ---
    print("\n==== Final Merge Summary ====")
    print(f"Total variables merged:   {len(all_columns)}")
    print(f"Variables:                {', '.join(all_columns)}")
    print(f"Theoretical total hours:  {theoretical_total}")
    print(f"Removed incomplete hours: {removed}")
    print(f"Remaining complete rows:  {final_count}")
    print("=============================")

    # Sort & save
    merged_df = merged_df.sort_values("datetime").reset_index(drop=True)
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Final merged dataset saved to: {output_file}")



# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_directory = Path("./processed")
    output_dataset = Path("./weather_data.csv")

    merge_preprocessed_data(input_directory, output_dataset)
