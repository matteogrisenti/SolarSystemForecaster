import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FFN_estimator import SolarPowerEstimator

# Initialize the estimator
estimator = SolarPowerEstimator()

# Example 4: Predict from CSV file
print("\n" + "="*60)
print(" Predict from Filtered CSV File")
print("="*60)

results = estimator.predict_from_csv(
        csv_path='filtered.csv',
        output_path='predictions.csv'
)
print(results.head())