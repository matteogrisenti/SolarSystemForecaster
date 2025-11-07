import pandas as pd

# Load CSV file
file_path = "solar_power_data.csv"  
df = pd.read_csv(file_path, parse_dates=['datetime'])

# Output file
output_file = "power_data_inspection.txt"

with open(output_file, "w") as f:

    # ----------------------
    # 1. Basic Info
    # ----------------------
    f.write("----- BASIC INFO -----\n")
    df_shape = df.shape
    f.write(f"Shape (rows, columns): {df_shape}\n")
    f.write(f"Number of data points: {len(df)}\n\n")

    f.write("----- HEAD -----\n")
    f.write(df.head().to_string() + "\n\n")

    # ----------------------
    # 2. Missing Values
    # ----------------------
    f.write("----- MISSING VALUES -----\n")
    missing_values = df.isnull().sum()
    f.write(missing_values.to_string() + "\n\n")

    # ----------------------
    # 3. Duplicates
    # ----------------------
    f.write("----- DUPLICATES -----\n")
    duplicates = df.duplicated().sum()
    f.write(f"Number of duplicate rows: {duplicates}\n\n")

    # ----------------------
    # 4. Missing Datetime Periods
    # ----------------------
    f.write("----- MISSING DATETIME PERIODS -----\n")
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    df_sorted['diff'] = df_sorted['datetime'].diff().dt.total_seconds() / 3600

    # Assuming the data should be hourly
    missing_periods = df_sorted[df_sorted['diff'] > 1]

    if missing_periods.empty:
        f.write("No missing datetime periods detected.\n\n")
    else:
        total_missing_points = int((missing_periods['diff'] - 1).sum())
        f.write(f"Missing periods detected: {len(missing_periods)}\n")
        f.write(f"Total missing datetimes: {total_missing_points}\n")

        for idx, row in missing_periods.iterrows():
            start_missing = df_sorted.loc[idx - 1, 'datetime'] + pd.Timedelta(hours=1)
            end_missing = row['datetime'] - pd.Timedelta(hours=1)
            gap_hours = int(row['diff'] - 1)
            f.write(f"  {start_missing} to {end_missing}  ({gap_hours} missing hours)\n")

        f.write("\n")


    # ----------------------
    # 5. Basic Statistics
    # ----------------------
    f.write("----- STATISTICS -----\n")
    f.write(df.describe().to_string() + "\n\n")

    # ----------------------
    # 6. Outlier Check
    # ----------------------
    f.write("----- OUTLIER CHECK -----\n")
    col = 'power'
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    f.write(f"{col}: {len(outliers)} potential outliers\n")

    f.write("\nData inspection complete!\n")

print(f"Inspection results saved to {output_file}")
