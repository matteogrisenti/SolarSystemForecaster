"""
===============================================================================
Weather Forecast Calibration Script
===============================================================================
This script applies a pre-trained Forecast Calibrator to historical forecast data
to improve the accuracy of weather predictions through statistical correction.

Usage:
    python calibrate_historical_forecast.py

Functionality:
    - Loads a trained ForecastCalibrator from file (calibrator.pkl)
    - Loads historical forecast data from CSV
    - Applies calibration using the ForecastCalibrator
    - Rounds the calibrated values to two decimal places
    - Saves the calibrated forecast to CSV
    - Prints a calibration summary

Requirements:
    - calibrator.pkl file (trained calibrator model)
    - dataset/historical_forecast.csv (forecast data to be calibrated)

Author: <Your Name or Team Name>
Project: SolarSystemForecaster
===============================================================================
"""

import json
import os
import sys
import pandas as pd

# Ensure parent directory is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calibrator import ForecastCalibrator

def post_process(df):
    """
    Apply post-processing corrections to ensure physical constraints.
    
    Post-processing rules:
    1. Irradiance: Set to 0 if negative (no negative solar radiation)
    2. Rain: Set to 0 if negative (no negative precipitation)
    3. Wind velocity: Set to 0 if negative (no negative speed)
    4. Round all numeric columns (except datetime, hour, day, month) to 2 decimals
    
    Args:
        df: DataFrame with forecast data
        verbose: verbose level
    
    Returns:
        DataFrame: Post-processed forecast data
    """
    
    df_processed = df.copy()
    
    # 1. Irradiance: set negative values to 0
    if 'irradiance' in df_processed.columns:
        mask = df_processed['irradiance'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'irradiance'] = 0
    
    # 2. Rain: set negative values to 0
    if 'rain' in df_processed.columns:
        mask = df_processed['rain'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'rain'] = 0
    
    # 3. Wind velocity: set negative values to 0
    if 'wind_velocity' in df_processed.columns:
        mask = df_processed['wind_velocity'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'wind_velocity'] = 0
    
    # 4. Round numeric columns (except datetime, hour, day, month)
    exclude_cols = ['datetime', 'hour', 'day', 'month']
    numeric_cols = df_processed.select_dtypes(include='number').columns
    round_cols = [col for col in numeric_cols if col not in exclude_cols]

    if round_cols:
        df_processed[round_cols] = df_processed[round_cols].round(2)
    
    return df_processed


def main():
    print("=" * 70)
    print("Weather Forecast Calibration System")
    print("Distribution Matching Method")
    print("=" * 70)
        
    print("\n\n### Testing Calibration ###\n")
    
    try:
        # Load the trained calibrator
        calibrator = ForecastCalibrator()
        calibrator.load('../calibrator.pkl')
        
        # Load new forecast data
        new_forecast = pd.read_csv('../dataset/historical_forecast.csv')
        
        # Apply calibration
        calibrated_forecast = calibrator.calibrate(new_forecast, verbose=2)
        
        # ✅ Round to 2 decimal places
        calibrated_forecast = post_process(calibrated_forecast)
        
        # Save calibrated forecast
        os.makedirs('test', exist_ok=True)
        output_path = 'calibrated_historical_forecast.csv'
        calibrated_forecast.to_csv(output_path, index=False)
        print(f"\n✓ Calibrated forecast saved to '{output_path}'")
        
        # Show summary
        print("\n" + "=" * 70)
        print("Calibration Summary")
        print("=" * 70)
        summary = calibrator.get_summary()
        print(json.dumps(summary, indent=2))
        
    except FileNotFoundError as e:
        print(f"\n⚠ Could not load calibrator or forecast: {e}")
        print("\nMake sure you have:")
        print("  1. Trained the calibrator first (calibrator.pkl)")
        print("  2. A forecast file to calibrate (dataset/historical_forecast.csv)")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
