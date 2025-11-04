import pandas as pd

# --- Configuration ---
input_file = "../../dataset/weather_data/weather_data.csv"  # your input CSV file
output_file = "historical_actual.csv"        # filtered output CSV file
start_datetime = "2024-11-04 00:00:00"  # start of the datetime range
end_datetime = "2025-10-28 00:00:00"    # end of the datetime range

# --- Load CSV file ---
df = pd.read_csv(input_file, parse_dates=['datetime'])

# --- Filter by datetime range ---
mask = (df['datetime'] >= start_datetime) & (df['datetime'] <= end_datetime)
filtered_df = df.loc[mask]

# --- Save filtered CSV ---
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
