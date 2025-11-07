import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, max_error

# Paths to CSV files
pred_csv = "test_predictions.csv"  # CSV with predicted_power
truth_csv = "../dataset/train_test_split/y_test.csv"  # CSV with actual_power
output_dir = ""  # save report in same folder

# Load CSVs
df_pred = pd.read_csv(pred_csv)
df_truth = pd.read_csv(truth_csv)

# Check columns
if 'predicted_power' not in df_pred.columns:
    raise ValueError("Predictions CSV must contain 'predicted_power' column.")
if 'power' not in df_truth.columns:
    raise ValueError("Ground truth CSV must contain 'power' column.")

# Ensure same number of rows
if len(df_pred) != len(df_truth):
    raise ValueError(f"Predictions and ground truth have different number of rows: "
                     f"{len(df_pred)} vs {len(df_truth)}")

# Extract arrays
y_pred = df_pred['predicted_power'].values
y_true = df_truth['power'].values

# Compute standard metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)
max_err = max_error(y_true, y_pred)

# Compute percentage accuracy within tolerance levels
def accuracy_within(y_true, y_pred, pct):
    """Returns fraction of predictions within ±pct% of true value."""
    tol = pct / 100
    within = np.abs(y_pred - y_true) <= tol * np.abs(y_true)
    return np.mean(within) * 100  # percentage

acc_5 = accuracy_within(y_true, y_pred, 5)
acc_10 = accuracy_within(y_true, y_pred, 10)
acc_15 = accuracy_within(y_true, y_pred, 15)
acc_20 = accuracy_within(y_true, y_pred, 20)

# Generate report
report = []
report.append("="*60)
report.append("FEEDFORWARD NEURAL NETWORK - PREDICTION EVALUATION")
report.append("="*60)
report.append(f"Number of samples: {len(df_pred)}")
report.append(f"MSE: {mse:.4f}")
report.append(f"RMSE: {rmse:.4f}")
report.append(f"MAE: {mae:.4f}")
report.append(f"Median AE: {medae:.4f}")
report.append(f"Max Error: {max_err:.4f}")
report.append(f"R²: {r2:.4f}")
report.append("-"*60)
report.append("Accuracy within tolerance:")
report.append(f" ±5%:  {acc_5:.2f}%")
report.append(f" ±10%: {acc_10:.2f}%")
report.append(f" ±15%: {acc_15:.2f}%")
report.append(f" ±20%: {acc_20:.2f}%")
report.append("="*60)

report_text = "\n".join(report)
print("\n" + report_text)

# Save report to file
report_path = os.path.join(output_dir, "fnn_prediction_evaluation.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\n✅ Evaluation report saved to: {report_path}")
