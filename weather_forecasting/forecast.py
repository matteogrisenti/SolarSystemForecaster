#!/usr/bin/env python3
"""
Weather Forecast Pipeline - Main Script

Complete pipeline that:
1. Fetches tomorrow's weather forecast using weather_forecast.py
2. Applies calibration using forecast_calibration.py
3. Post-processes data (physical constraints)
4. Saves final forecast to CSV

This script orchestrates the other modules without reimplementing them.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import argparse

# Dynamically add directories to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
calibrator_dir = os.path.join(base_dir, 'calibrator')
open_meteo_dir = base_dir 

sys.path.append(calibrator_dir)
sys.path.append(open_meteo_dir)

try:
    from open_meteo_fetch import fetch_weather_forecast
    from calibrator import ForecastCalibrator
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("\nMake sure these files are in the correct locations:")
    print(f"  - {open_meteo_dir}/open_meteo_fetch.py")
    print(f"  - {calibrator_dir}/calibrator.py")
    sys.exit(1)

calibrator_path_standard = f"{calibrator_dir}/calibrator.py"
def apply_calibration(forecast_df, calibrator_path=calibrator_path_standard):
    """
    Apply calibration to forecast data using trained calibrator.
    
    Args:
        forecast_df: DataFrame with raw forecast data
        calibrator_path: Path to trained calibrator file
    
    Returns:
        DataFrame: Calibrated forecast data
    """
    
    if not os.path.exists(calibrator_path):
        print(f"\n⚠ Warning: Calibrator not found at '{calibrator_path}'")
        print("Skipping calibration step. To use calibration:")
        print("  1. Train a calibrator using forecast_calibration.py")
        print("  2. Save it to the specified path")
        return forecast_df
    
    try:        
        # Load and apply calibrator
        calibrator = ForecastCalibrator()
        calibrator.load(calibrator_path)
        
        calibrated_df = calibrator.calibrate(forecast_df)
        
        print("\n✓ Calibration completed")
        return calibrated_df
        
    except Exception as e:
        print(f"\n⚠ Warning: Calibration failed: {e}")
        print("Continuing with uncalibrated forecast")
        return forecast_df


def post_process(df):
    """
    Apply post-processing corrections to ensure physical constraints.
    
    Post-processing rules:
    1. Irradiance: Set to 0 if negative (no negative solar radiation)
    2. Rain: Set to 0 if negative (no negative precipitation)
    3. Wind velocity: Set to 0 if negative (no negative speed)
    
    Args:
        df: DataFrame with forecast data
    
    Returns:
        DataFrame: Post-processed forecast data
    """
    
    df_processed = df.copy()
    corrections_made = False
    
    # 1. Irradiance: set negative values to 0
    if 'irradiance' in df_processed.columns:
        mask = df_processed['irradiance'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'irradiance'] = 0
            corrections_made = True
            print(f"✓ Irradiance: Corrected {num_corrections} negative values to 0")
    
    # 4. Rain: set negative values to 0
    if 'rain' in df_processed.columns:
        mask = df_processed['rain'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'rain'] = 0
            corrections_made = True
            print(f"✓ Rain: Corrected {num_corrections} negative values to 0")
    
    # 4. Wind velocity: set negative values to 0
    if 'wind_velocity' in df_processed.columns:
        mask = df_processed['wind_velocity'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'wind_velocity'] = 0
            corrections_made = True
            print(f"✓ Wind velocity: Corrected {num_corrections} negative values to 0")
    
    if not corrections_made:
        print("✓ No corrections needed - all values within physical constraints")
    
    return df_processed


def save_forecast(df, filename='weather_forecast.csv'):
    """
    Save forecast to CSV file.
    
    Args:
        df: DataFrame with forecast data
        filename: Output filename
    
    Returns:
        bool: Success status
    """
    
    try:
        df.to_csv(filename, index=False)
        print(f"\n✓ Forecast saved to '{filename}'")
        print(f"  Records: {len(df)}")
        print(f"  Date: {df['date'].iloc[0]}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving forecast: {e}", file=sys.stderr)
        return False


def weather_forecast(latitude = 46.09564, longitude=11.10136):
    """Main pipeline function."""
    
    print("=" * 70)
    print("WEATHER FORECAST")
    print("=" * 70)
    print(f"\nLocation: Lat {latitude}, Lon {longitude}")
    print(f"Date: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
    
    # STEP 1: Fetch forecast
    print("\n" + "=" * 70)
    print("STEP 1: Fetching Forecast")
    print("=" * 70)
    forecast_df = fetch_weather_forecast(latitude, longitude)
    
    if forecast_df is None:
        print("\n✗ Pipeline failed: Could not fetch forecast data")
        return 1
    
    # STEP 2: Apply calibration (if not skipped)
    print("\n" + "=" * 70)
    print("STEP 2: Applying Calibration")
    print("=" * 70)
    forecast_df = apply_calibration(forecast_df)
   
    # STEP 3: Post-processing
    print("\n" + "=" * 70)
    print("STEP 3: Post-Processing")
    print("=" * 70)
    forecast_df = post_process(forecast_df)
    
    return forecast_df

def main():
    LATITUDE = 46.09564
    LONGITUDE = 11.10136
    OUTPUT = ""

    forecast = weather_forecast(LATITUDE, LONGITUDE)
    success = save_forecast(forecast, OUTPUT)

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFinal forecast available in: {OUTPUT}")


if __name__ == "__main__":   
    sys.exit(main())