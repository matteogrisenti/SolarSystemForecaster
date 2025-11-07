import pandas as pd

# === USER INPUTS ===
input_file = "../FFN/dataset/dataset.csv"     # Path to your source CSV
output_file = "groundtruth.csv"           # Path to save filtered CSV

start_datetime = "2024-11-04 00:00:00"
end_datetime   = "2025-10-28 00:00:00"

# === SCRIPT ===
# Read the CSV file and parse datetime column
df = pd.read_csv(input_file, parse_dates=["datetime"])

# Filter rows between start and end datetime
mask = (df["datetime"] >= start_datetime) & (df["datetime"] <= end_datetime)
filtered_df = df.loc[mask]

# Save filtered data to a new CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to '{output_file}' with {len(filtered_df)} rows.")
