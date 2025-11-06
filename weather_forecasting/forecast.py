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
import logging

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


# Configure logging
logger = logging.getLogger("forecast")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("WEATHER FORECAST - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add handler if it hasn't been added yet
if not logger.hasHandlers():
    logger.addHandler(ch)


calibrator_path_standard = f"{calibrator_dir}/calibrator.pkl"
def apply_calibration(forecast_df, verbose=1, calibrator_path=calibrator_path_standard):
    """
    Apply calibration to forecast data using trained calibrator.
    
    Args:
        forecast_df: DataFrame with raw forecast data
        calibrator_path: Path to trained calibrator file
    
    Returns:
        DataFrame: Calibrated forecast data
    """
    
    if not os.path.exists(calibrator_path):
        logger.warning(f"\n⚠ Warning: Calibrator not found at '{calibrator_path}'")
        logger.warning("Skipping calibration step. To use calibration:")
        logger.warning("  1. Train a calibrator using forecast_calibration.py")
        logger.warning("  2. Save it to the specified path")
        return forecast_df
    
    try:        
        # Load and apply calibrator
        # 1) Load Calibrator
        try:
            calibrator = ForecastCalibrator()
            calibrator.load(calibrator_path, verbose = verbose)
        except Exception as e:
            raise RuntimeError(f"Error loading calibrator ... \n\t Calibrator Path: {calibrator_path} \n\t Error: {e}")
        # 2) Apply Calibrator
        try:
            calibrated_df = calibrator.calibrate(forecast_df, verbose=0)
        except Exception as e:
            raise RuntimeError(f"Error appling calibrator: {e}")
        
        if verbose >=1: 
            logger.info("✓ Calibration completed")
        return calibrated_df
        
    except Exception as e:
        logger.warning(f"\n⚠ Warning: Calibration failed: {e}")
        logger.warning("Continuing with uncalibrated forecast")
        return forecast_df


def post_process(df, verbose=1):
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
            if verbose >=1: 
                logger.info(f"✓ Irradiance: Corrected {num_corrections} negative values to 0")
    
    # 2. Rain: set negative values to 0
    if 'rain' in df_processed.columns:
        mask = df_processed['rain'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'rain'] = 0
            if verbose >=1: 
                logger.info(f"✓ Rain: Corrected {num_corrections} negative values to 0")
    
    # 3. Wind velocity: set negative values to 0
    if 'wind_velocity' in df_processed.columns:
        mask = df_processed['wind_velocity'] < 0
        num_corrections = mask.sum()
        if num_corrections > 0:
            df_processed.loc[mask, 'wind_velocity'] = 0
            if verbose >=1: 
                logger.info(f"✓ Wind velocity: Corrected {num_corrections} negative values to 0")
    
    # 4. Round numeric columns (except datetime, hour, day, month)
    exclude_cols = ['datetime', 'hour', 'day', 'month']
    numeric_cols = df_processed.select_dtypes(include='number').columns
    round_cols = [col for col in numeric_cols if col not in exclude_cols]

    if round_cols:
        df_processed[round_cols] = df_processed[round_cols].round(2)
    
    return df_processed


def save_csv(df, filename='tomorrow_forecast.csv'):
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
        logger.info(f"✓ Forecast saved to '{filename}'")
        logger.info(f"  Records: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"✗ Error saving forecast: {e}", file=sys.stderr)
        return False


def weather_forecast(latitude = 46.09564, longitude=11.10136, verbose=1):
    """
    All forecast pipeline
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        verbose (int): Verbosity level:
            0 - No output
            1 - Summary only
            2 - Detailed output for each step
            3 - Save csv file for each step
    
    Returns:
        pd.DataFrame: calibrated & post-processed forecast data.
    """
    
    if verbose >=1:
        logger.info("=" * 70)
        logger.info("WEATHER FORECAST")
        logger.info("=" * 70)
        logger.info(f"Location: Lat {latitude}, Lon {longitude}")
        logger.info(f"Date: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}")
    
    # STEP 1: Fetch forecast
    if verbose >= 2:
        print('')
        logger.info("=" * 70)
        logger.info("STEP 1: Fetching Forecast")
        logger.info("=" * 70)
    forecast_df = fetch_weather_forecast(latitude, longitude, verbose)
    
    if forecast_df is not None:
        if verbose >= 2:
            print('')
            logger.info("Forecast DataFrame preview:")
            print(forecast_df.head())
    else:
        logger.error("✗ No forecast data returned")
        return 1
    
    # STEP 2: Apply calibration (if not skipped)
    if verbose >= 2:
        print('\n')
        logger.info("=" * 70)
        logger.info("STEP 2: Applying Calibration")
        logger.info("=" * 70)
    calibrated_forecast_df = apply_calibration(forecast_df, verbose = verbose)
   
    # STEP 3: Post-processing
    if verbose >= 2:
        print('\n')
        logger.info("=" * 70)
        logger.info("STEP 3: Post-Processing")
        logger.info("=" * 70)
    fine_forecast_df = post_process(calibrated_forecast_df, verbose=verbose)

    if verbose == 3:
        print('\n')
        logger.info("=" * 70)
        logger.info("Save All Intermiadiate Steps")
        save_csv(forecast_df, 'raw_tomorrow_forecast.csv')
        save_csv(calibrated_forecast_df, 'calibrated_tomorrow_forecast.csv')
        save_csv(fine_forecast_df, 'fine_tomorrow_forecast.csv')
        logger.info("=" * 70)
    
    return fine_forecast_df





#=====================================================================
# Suplementary Code
#=====================================================================

def main():
    LATITUDE = 46.09564
    LONGITUDE = 11.10136

    forecast = weather_forecast(LATITUDE, LONGITUDE, verbose=0)

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":   
    sys.exit(main())