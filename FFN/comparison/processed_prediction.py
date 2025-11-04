import pandas as pd

# === CONFIGURATION ===
input_csv = "predictions.csv"         # Input file name
output_excel = "processed.xlsx" # Output Excel file name

# === SCRIPT ===
def process_csv_to_excel(input_csv, output_excel):
    # Read the CSV
    df = pd.read_csv(input_csv)

    # Keep only datetime and predicted_power columns
    df = df[["datetime", "predicted_power"]]

    # Round predicted_power to nearest integer
    df["predicted_power"] = df["predicted_power"].round(0).astype(int)

    # Save to Excel (no index column)
    df.to_excel(output_excel, index=False)

    print(f"Processed Excel file saved as: {output_excel}")
    print(df.head())  # Show preview

# === RUN ===
if __name__ == "__main__":
    process_csv_to_excel(input_csv, output_excel)
