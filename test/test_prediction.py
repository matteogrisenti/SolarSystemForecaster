import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths to CSV files
pred_csv = "test_predictions.csv"  # CSV with 'datetime' and 'predicted_power'
truth_csv = "groundtruth.csv"      # CSV with 'datetime' and 'power'
output_dir = ""                    # save report in same folder

# Load CSVs
df_pred = pd.read_csv(pred_csv)
df_truth = pd.read_csv(truth_csv)

# Check required columns
for col in ['datetime', 'predicted_power']:
    if col not in df_pred.columns:
        raise ValueError(f"Predictions CSV must contain '{col}' column.")
for col in ['datetime', 'power']:
    if col not in df_truth.columns:
        raise ValueError(f"Ground truth CSV must contain '{col}' column.")

# Convert datetime columns to datetime objects
df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
df_truth['datetime'] = pd.to_datetime(df_truth['datetime'])

# Merge dataframes on datetime to align predictions with ground truth
df_merged = pd.merge(df_pred, df_truth, on='datetime', how='inner')

if df_merged.empty:
    raise ValueError("No matching datetime values found between predictions and ground truth.")

# -------------------------
# Part 1: Hourly Evaluation (all hours)
# -------------------------
y_pred = df_merged['predicted_power'].values
y_true = df_merged['power'].values

# Metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Additional statistics
errors = np.abs(y_true - y_pred)
max_error = np.max(errors)
min_error = np.min(errors)
mean_percentage_error = np.mean(errors / (np.abs(y_true) + 1e-8)) * 100

# Accuracy thresholds
accuracy_thresholds = [0.05, 0.10, 0.20]  # 5%, 10%, 20%
accuracy_results = {}
for threshold in accuracy_thresholds:
    within_threshold = np.mean(errors <= threshold * np.abs(y_true))
    accuracy_results[threshold] = within_threshold * 100

# -------------------------
# Part 1b: Hourly Evaluation (exclude ground truth = 0)
# -------------------------
df_nonzero = df_merged[df_merged['power'] != 0].copy()

y_pred_nz = df_nonzero['predicted_power'].values
y_true_nz = df_nonzero['power'].values

mse_nz = mean_squared_error(y_true_nz, y_pred_nz)
rmse_nz = np.sqrt(mse_nz)
mae_nz = mean_absolute_error(y_true_nz, y_pred_nz)
r2_nz = r2_score(y_true_nz, y_pred_nz)

errors_nz = np.abs(y_true_nz - y_pred_nz)
max_error_nz = np.max(errors_nz)
min_error_nz = np.min(errors_nz)
mean_percentage_error_nz = np.mean(errors_nz / (np.abs(y_true_nz) + 1e-8)) * 100

accuracy_results_nz = {}
for threshold in accuracy_thresholds:
    within_threshold = np.mean(errors_nz <= threshold * np.abs(y_true_nz))
    accuracy_results_nz[threshold] = within_threshold * 100

# -------------------------
# Part 1c: Per-hour metrics (exclude zero production)
# -------------------------
df_nonzero['hour'] = df_nonzero['datetime'].dt.hour

hourly_metrics = {}
for hour, group in df_nonzero.groupby('hour'):
    y_true_h = group['power'].values
    y_pred_h = group['predicted_power'].values
    errors_h = np.abs(y_true_h - y_pred_h)

    metrics = {
        'MSE': mean_squared_error(y_true_h, y_pred_h),
        'RMSE': np.sqrt(mean_squared_error(y_true_h, y_pred_h)),
        'MAE': mean_absolute_error(y_true_h, y_pred_h),
        'R2': r2_score(y_true_h, y_pred_h),
        'Max Error': np.max(errors_h),
        'Min Error': np.min(errors_h),
        'Mean Percentage Error': np.mean(errors_h / (np.abs(y_true_h) + 1e-8)) * 100,
        'Accuracy': {threshold: np.mean(errors_h <= threshold * np.abs(y_true_h)) * 100 for threshold in accuracy_thresholds}
    }
    hourly_metrics[hour] = metrics

# -------------------------
# Part 2: Daily Evaluation
# -------------------------
df_merged['date'] = df_merged['datetime'].dt.date
daily_df = df_merged.groupby('date').agg({
    'predicted_power': 'sum',
    'power': 'sum'
}).reset_index()

y_pred_daily = daily_df['predicted_power'].values
y_true_daily = daily_df['power'].values

mse_daily = mean_squared_error(y_true_daily, y_pred_daily)
rmse_daily = np.sqrt(mse_daily)
mae_daily = mean_absolute_error(y_true_daily, y_pred_daily)
r2_daily = r2_score(y_true_daily, y_pred_daily)

errors_daily = np.abs(y_true_daily - y_pred_daily)
max_error_daily = np.max(errors_daily)
min_error_daily = np.min(errors_daily)
mean_percentage_error_daily = np.mean(errors_daily / (np.abs(y_true_daily) + 1e-8)) * 100

accuracy_results_daily = {}
for threshold in accuracy_thresholds:
    within_threshold = np.mean(errors_daily <= threshold * np.abs(y_true_daily))
    accuracy_results_daily[threshold] = within_threshold * 100

# -------------------------
# Generate Report
# -------------------------
report = []

# Hourly Evaluation (all hours)
report.append("="*70)
report.append("HOURLY PREDICTION EVALUATION - ALL HOURS")
report.append("="*70)
report.append(f"Number of aligned samples: {len(df_merged)}")
report.append(f"MSE:  {mse:.4f}")
report.append(f"RMSE: {rmse:.4f}")
report.append(f"MAE:  {mae:.4f}")
report.append(f"R²:   {r2:.4f}")
report.append(f"Max Error: {max_error:.4f}")
report.append(f"Min Error: {min_error:.4f}")
report.append(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
report.append("-"*70)
report.append("Prediction Accuracy within Thresholds:")
for threshold, acc in accuracy_results.items():
    report.append(f"  Within {int(threshold*100)}%: {acc:.2f}%")
report.append("="*70 + '\n\n')

# Hourly Evaluation (exclude zero production)
report.append("HOURLY PREDICTION EVALUATION - NONZERO GROUND TRUTH")
report.append("="*70)
report.append(f"Number of aligned nonzero samples: {len(df_nonzero)}")
report.append(f"MSE:  {mse_nz:.4f}")
report.append(f"RMSE: {rmse_nz:.4f}")
report.append(f"MAE:  {mae_nz:.4f}")
report.append(f"R²:   {r2_nz:.4f}")
report.append(f"Max Error: {max_error_nz:.4f}")
report.append(f"Min Error: {min_error_nz:.4f}")
report.append(f"Mean Percentage Error: {mean_percentage_error_nz:.2f}%")
report.append("-"*70)
report.append("Prediction Accuracy within Thresholds:")
for threshold, acc in accuracy_results_nz.items():
    report.append(f"  Within {int(threshold*100)}%: {acc:.2f}%")
report.append("="*70+ '\n\n')

# Hourly per-hour metrics
report.append("PER-HOUR METRICS (EXCLUDING ZERO PRODUCTION HOURS)")
report.append("="*70)
for hour in sorted(hourly_metrics.keys()):
    m = hourly_metrics[hour]
    report.append(f"Hour {hour:02d}:")
    report.append(f"  MSE:  {m['MSE']:.4f}")
    report.append(f"  RMSE: {m['RMSE']:.4f}")
    report.append(f"  MAE:  {m['MAE']:.4f}")
    report.append(f"  R²:   {m['R2']:.4f}")
    report.append(f"  Max Error: {m['Max Error']:.4f}")
    report.append(f"  Min Error: {m['Min Error']:.4f}")
    report.append(f"  Mean Percentage Error: {m['Mean Percentage Error']:.2f}%")
    report.append("  Accuracy within thresholds:")
    for threshold, acc in m['Accuracy'].items():
        report.append(f"    Within {int(threshold*100)}%: {acc:.2f}%")
    report.append("-"*40+ '\n\n')

# Daily Evaluation
report.append("DAILY PREDICTION EVALUATION (SUMS)")
report.append("="*70)
report.append(f"Number of days evaluated: {len(daily_df)}")
report.append(f"MSE:  {mse_daily:.4f}")
report.append(f"RMSE: {rmse_daily:.4f}")
report.append(f"MAE:  {mae_daily:.4f}")
report.append(f"R²:   {r2_daily:.4f}")
report.append(f"Max Error: {max_error_daily:.4f}")
report.append(f"Min Error: {min_error_daily:.4f}")
report.append(f"Mean Percentage Error: {mean_percentage_error_daily:.2f}%")
report.append("-"*70+ '\n\n')
report.append("Prediction Accuracy within Thresholds (Daily Sums):")
for threshold, acc in accuracy_results_daily.items():
    report.append(f"  Within {int(threshold*100)}%: {acc:.2f}%")
report.append("="*70+ '\n\n')

# Save report
report_text = "\n".join(report)
print("\n" + report_text)

report_path = os.path.join(output_dir, "fnn_prediction_evaluation.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\n✅ Evaluation report saved to: {report_path}")
