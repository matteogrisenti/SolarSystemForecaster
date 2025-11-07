import pandas as pd

# === CONFIGURATION ===
input_csv = "../dataset/dataset.csv"       # Path to your source CSV
output_csv = "filtered.csv"   # Path to save filtered CSV
datetime_column = "datetime"  # Name of the datetime column in your CSV

# === SCRIPT ===
def filter_by_date(input_csv, output_csv, datetime_column):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Convert the datetime column to proper datetime type
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    # Drop any rows where datetime could not be parsed
    df = df.dropna(subset=[datetime_column])

    # Define the cutoff date
    cutoff = pd.Timestamp("2022-01-01 00:00:00")

    # Filter rows from the cutoff date onward
    filtered_df = df[df[datetime_column] >= cutoff]

    # Save the filtered data to a new CSV
    filtered_df.to_csv(output_csv, index=False)

    print(f"Filtered data saved to: {output_csv}")
    print(f"Rows kept: {len(filtered_df)} / {len(df)}")

# === RUN ===
if __name__ == "__main__":
    filter_by_date(input_csv, output_csv, datetime_column)
