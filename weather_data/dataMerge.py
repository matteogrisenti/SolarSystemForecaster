import pandas as pd
from pathlib import Path

def merge_preprocessed_data(input_dir, output_file):
    """
    Merge all preprocessed hourly CSVs into a single dataset.
    Keeps only hours where all sensors have valid data.
    Prints a detailed summary.
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

    # --- Load and merge each file ---
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        
        if "hour" not in df.columns:
            print(f"⚠️  Skipping {file.name} — missing 'hour' column.")
            continue
        
        # Ensure proper datetime parsing
        df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
        df = df.dropna(subset=["hour"])
        
        # Report
        colname = [c for c in df.columns if c != "hour"][0]
        all_columns.append(colname)
        print(f"📄 Loaded {file.name} with column: {colname} ({len(df)} rows)")

        # Merge step
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="hour", how="inner")  # keep only hours in both

    if merged_df is None or merged_df.empty:
        print("❌ No valid data after merging.")
        return

    # --- Check for missing data ---
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

    # --- Save result ---
    merged_df = merged_df.sort_values("hour").reset_index(drop=True)
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Final merged dataset saved to: {output_file}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_directory = Path("./preprocessed_weather_csv")
    output_dataset = Path("./final_weather_dataset.csv")

    merge_preprocessed_data(input_directory, output_dataset)
