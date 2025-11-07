import pandas as pd
from pathlib import Path

def preprocess_solar_data(input_dir, output_csv):
    """
    Preprocess yearly solar production Excel files:
    - Reads all .xlsx files from the input directory
    - Cleans and validates the data
    - Merges all years into a single hourly CSV file
    - Removes duplicates and invalid values
    """

    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.xlsx"))

    if not files:
        print("⚠️ No Excel files found in input directory.")
        return

    print("🔆 START SOLAR DATA PREPROCESSING")
    print("=" * 70)
    print(f"📁 Input Directory:  {input_dir.resolve()}")
    print(f"💾 Output File:      {Path(output_csv).resolve()}")
    print("=" * 70)

    all_data = []

    for file in files:
        print(f"\n📄 Processing file: {file.name}")

        # Load data (skip empty rows or extra headers)
        df = pd.read_excel(file, engine="openpyxl")

        # Expect columns: "Data", "Potenza Attiva FV Trento"
        if df.shape[1] < 2:
            print(f"⚠️  Skipping {file.name} — unexpected column count ({df.shape[1]})")
            continue

        # Rename columns to standard names
        df.columns = ["datetime", "power"]

        # Drop completely empty rows
        df = df.dropna(subset=["datetime", "power"])

        # Parse datetime with day-first logic
        # This prevents month/day inversion (like 12/01/2020 being read as Dec 1 instead of Jan 12)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["datetime"])


        # --- REMOTION OF OUTLIERS OR DUPLICATED DATETIME ---
        initial_len = len(df)

        # Remove invalid or impossible power values (e.g., negative or too high)
        invalid_mask = (df["power"] < 0) | (df["power"] >= 2000)
        print(f"🗑️  Removed invalid/outliers: {invalid_mask.sum()}")
        df = df[~invalid_mask]

        # Remove duplicates
        duplicate_removed = df.duplicated(subset=["datetime"]).sum()
        print(f"🗑️  Removed duplicates: {duplicate_removed}")
        df = df.drop_duplicates(subset=["datetime"], keep="first")

    

        # --- RECOVER THE MISSING DATETIME --
        # Recover missing date point during the night 22:00-5:00 and add to the data with power 0
        full_range = pd.date_range(start=df["datetime"].min(), end=df["datetime"].max(), freq="h")
        missing_times = full_range.difference(df["datetime"])          # Identify missing timestamps

        # Prepare nighttime filler rows
        fill_rows = []
        for t in missing_times:
            hour = t.hour
            # Nighttime range: 22:00–05:00
            if hour >= 22 or hour <= 5:
                fill_rows.append({"datetime": t, "power": 0})

        # Add missing nighttime rows
        if fill_rows:
            df = pd.concat([df, pd.DataFrame(fill_rows)], ignore_index=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            # Show which hours were filled
            night_hours = sorted(set([t.hour for t in missing_times if t.hour >= 22 or t.hour <= 5]))
            print(f"🌙 Added {len(fill_rows)} nighttime (22:00-5:00) missing rows (set to 0).")


        # Sort chronologically
        df = df.sort_values("datetime").reset_index(drop=True)

        # Report summary
        removed = initial_len - len(df)
        print(f"✅ Clean data points: {len(df)}  |  Removed: {removed}")

        all_data.append(df)

    if not all_data:
        print("❌ No valid data to merge.")
        return

    # --- Merge all dataframes ---
    merged_df = pd.concat(all_data, ignore_index=True)

    # Drop duplicates across years
    merged_df = merged_df.drop_duplicates(subset=["datetime"], keep="first")

    # Sort by time
    merged_df = merged_df.sort_values("datetime").reset_index(drop=True)

    # --- Save final dataset ---
    merged_df.to_csv(output_csv, index=False)

    print("\n==== Final Summary ====")
    print(f"Total yearly files processed: {len(all_data)}")
    print(f"Final merged dataset size:   {len(merged_df)} rows")
    print("=============================")
    print(f"✅ Merged solar dataset saved to: {output_csv}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_directory = Path("./raw")  # folder with yearly Excel files
    output_file = Path("./solar_power_data.csv")

    preprocess_solar_data(input_directory, output_file)
