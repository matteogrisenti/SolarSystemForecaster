#!/usr/bin/env python3
"""
Weather Forecast Calibration System - Distribution Matching

Calibrates weather forecasts by matching the statistical distribution
(mean and variance) of forecasts to actual measurements.

Calibration formula:
    calibrated = (forecast - μ_forecast) × (σ_actual / σ_forecast) + μ_actual

This transforms the forecast distribution to match the actual data distribution.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle
import logging


# Configure logging
logger = logging.getLogger("calibration")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("WEATHER FORECAST - CLIBRATION - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add handler if it hasn't been added yet
if not logger.hasHandlers():
    logger.addHandler(ch)


class ForecastCalibrator:
    """
    Calibrates weather forecasts using distribution matching.
    
    Computes mean and standard deviation of both forecast and actual data,
    then transforms forecasts to match the actual distribution.
    """
    
    PARAMETERS = [
        'air_temperature',
        'humidity', 
        'irradiance',
        'pressure',
        'rain',
        'wind_direction',
        'wind_velocity'
    ]
    
    def __init__(self):
        """Initialize calibrator."""
        self.calibration_params = {}
        self.statistics = {}
        self.is_trained = False
        self.training_date = None
        self.training_samples = 0
    

    def train(self, forecast_df, actual_df, matching_columns=['date', 'hour']):
        """
        Train calibration by comparing forecast vs actual data distributions.
        
        Args:
            forecast_df: DataFrame with forecasted data
            actual_df: DataFrame with actual measured data
            matching_columns: Columns to use for matching records (default: date and hour)
        
        Returns:
            dict: Training statistics for each parameter
        """
        
        print("=" * 70)
        print("Training Calibration Models - Distribution Matching")
        print("=" * 70)
        
        # Merge forecast and actual data
        merged = pd.merge(
            forecast_df,
            actual_df,
            on=matching_columns,
            suffixes=('_forecast', '_actual')
        )
        
        if len(merged) == 0:
            raise ValueError("No matching records found between forecast and actual data!")
        
        print(f"\nMatched {len(merged)} records")
        print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
        
        self.training_samples = len(merged)
        self.training_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Train for each parameter
        for param in self.PARAMETERS:
            forecast_col = f"{param}_forecast"
            actual_col = f"{param}_actual"
            
            if forecast_col not in merged.columns or actual_col not in merged.columns:
                print(f"\n⚠ Skipping {param} - column not found")
                continue
            
            # Remove NaN values
            mask = merged[forecast_col].notna() & merged[actual_col].notna()
            forecast_vals = merged.loc[mask, forecast_col].values
            actual_vals = merged.loc[mask, actual_col].values
            
            if len(forecast_vals) < 10:
                print(f"\n⚠ Skipping {param} - insufficient data ({len(forecast_vals)} samples)")
                continue
            
            self._train_parameter(param, forecast_vals, actual_vals)
        
        self.is_trained = True
        print("\n✓ Training completed!")
        
        return self.statistics
    

    def _train_parameter(self, param, forecast_vals, actual_vals):
        """
        Train calibration for a single parameter using distribution matching.
        
        Computes distribution parameters (mean, std) and validates calibration.
        """
        
        # Compute distribution parameters
        mean_forecast = np.mean(forecast_vals)
        std_forecast = np.std(forecast_vals, ddof=1)
        mean_actual = np.mean(actual_vals)
        std_actual = np.std(actual_vals, ddof=1)
        
        # Avoid division by zero
        if std_forecast < 1e-10:
            std_forecast = 1.0
            scaling_factor = 1.0
            print(f"\n⚠ Warning: {param} has zero variance in forecast, using scaling=1.0")
        else:
            scaling_factor = std_actual / std_forecast
        
        # Store calibration parameters
        self.calibration_params[param] = {
            'mean_forecast': float(mean_forecast),
            'std_forecast': float(std_forecast),
            'mean_actual': float(mean_actual),
            'std_actual': float(std_actual),
            'scaling_factor': float(scaling_factor),
            'mean_shift': float(mean_actual - mean_forecast)
        }
        
        # Apply calibration to training data for validation
        calibrated = (forecast_vals - mean_forecast) * scaling_factor + mean_actual
        
        # Calculate errors before and after calibration
        errors_original = actual_vals - forecast_vals
        errors_calibrated = actual_vals - calibrated
        
        mae_original = np.mean(np.abs(errors_original))
        mae_calibrated = np.mean(np.abs(errors_calibrated))
        rmse_original = np.sqrt(np.mean(errors_original**2))
        rmse_calibrated = np.sqrt(np.mean(errors_calibrated**2))
        
        bias_original = np.mean(errors_original)
        bias_calibrated = np.mean(errors_calibrated)
        
        # Calculate improvement percentage
        mae_improvement = (mae_original - mae_calibrated) / mae_original * 100 if mae_original > 0 else 0
        rmse_improvement = (rmse_original - rmse_calibrated) / rmse_original * 100 if rmse_original > 0 else 0
        
        # Store statistics
        self.statistics[param] = {
            'samples': int(len(forecast_vals)),
            'forecast_mean': float(mean_forecast),
            'forecast_std': float(std_forecast),
            'actual_mean': float(mean_actual),
            'actual_std': float(std_actual),
            'scaling_factor': float(scaling_factor),
            'mean_shift': float(mean_actual - mean_forecast),
            'mae_original': float(mae_original),
            'mae_calibrated': float(mae_calibrated),
            'mae_improvement_percent': float(mae_improvement),
            'rmse_original': float(rmse_original),
            'rmse_calibrated': float(rmse_calibrated),
            'rmse_improvement_percent': float(rmse_improvement),
            'bias_original': float(bias_original),
            'bias_calibrated': float(bias_calibrated)
        }
        
        # Print statistics
        print(f"\n{param}:")
        print(f"  Samples: {len(forecast_vals)}")
        print(f"  Distribution comparison:")
        print(f"    Forecast: μ={mean_forecast:.3f}, σ={std_forecast:.3f}")
        print(f"    Actual:   μ={mean_actual:.3f}, σ={std_actual:.3f}")
        print(f"  Calibration parameters:")
        print(f"    Scaling factor: {scaling_factor:.4f}")
        print(f"    Mean shift: {mean_actual-mean_forecast:.4f}")
        print(f"  Performance improvement:")
        print(f"    MAE: {mae_original:.4f} → {mae_calibrated:.4f} ({mae_improvement:+.1f}%)")
        print(f"    RMSE: {rmse_original:.4f} → {rmse_calibrated:.4f} ({rmse_improvement:+.1f}%)")
        print(f"    Bias: {bias_original:.4f} → {bias_calibrated:.4f}")
    

    def calibrate(self, forecast_df, verbose=1):
        """
        Apply calibration to new forecast data.
        
        Args:
            forecast_df (pd.DataFrame): DataFrame with forecast data to calibrate.
            verbose (int): Verbosity level:
                0 - No output
                1 - Summary only
                2 - Detailed output for each parameter.
        
        Returns:
            pd.DataFrame: Calibrated forecast data with original values preserved.
        """
        
        if not self.is_trained:
            raise ValueError("Calibrator must be trained before calibrating forecasts!")

        calibrated_df = forecast_df.copy()

        if verbose >= 1:
            logger.info("Applying Distribution-Based Calibration")
        
        summary = []

        for param in self.PARAMETERS:
            if param not in self.calibration_params:
                continue
            
            if param not in calibrated_df.columns:
                if verbose >= 2:
                    logger.warning(f"⚠ Skipping {param} - not found in forecast data")
                continue
            
            mask = calibrated_df[param].notna()
            values = calibrated_df.loc[mask, param].values
            
            if len(values) == 0:
                continue

             # Ensure the column is float
            if not np.issubdtype(calibrated_df[param].dtype, np.floating):
                calibrated_df[param] = calibrated_df[param].astype(float)

            # Apply calibration
            params = self.calibration_params[param]
            calibrated_values = (
                (values - params['mean_forecast']) * params['scaling_factor'] 
                + params['mean_actual']
            )
            
            calibrated_df.loc[mask, param] = calibrated_values
            
            # Compute adjustment stats
            adjustment = calibrated_values - values
            mean_adj = adjustment.mean()
            std_adj = adjustment.std()
            min_adj = adjustment.min()
            max_adj = adjustment.max()
            
            summary.append({
                "param": param,
                "count": mask.sum(),
                "mean_adj": mean_adj,
                "std_adj": std_adj,
                "min_adj": min_adj,
                "max_adj": max_adj
            })
            
            if verbose >= 2:
                print('')
                logger.info(f"{param}:")
                logger.info(f"  Adjusted {mask.sum()} values")
                logger.info(f"  Mean adjustment: {mean_adj:.4f}")
                logger.info(f"  Std adjustment: {std_adj:.4f}")
                logger.info(f"  Range: [{min_adj:.4f}, {max_adj:.4f}]")
        
        if verbose >= 1:
            print('')
            logger.info("✓ Calibration applied!")
            if summary and verbose == 1:
                print('')
                logger.info("Summary of Adjustments:")
                for s in summary:
                    logger.info(f"  {s['param']}: mean={s['mean_adj']:.4f}, std={s['std_adj']:.4f}")

        return calibrated_df


    def save(self, filename='forecast_calibrator.pkl'):
        """
        Save calibrator to file.
        
        Args:
            filename: Path to save file
        """
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained calibrator!")
        
        save_data = {
            'calibration_params': self.calibration_params,
            'statistics': self.statistics,
            'training_date': self.training_date,
            'training_samples': self.training_samples,
            'method': 'distribution',
            'version': '2.0'
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"\n✓ Calibrator saved to {filename}")
    

    def load(self, filename='forecast_calibrator.pkl', verbose=1):
        """
        Load calibrator from file.
        
        Args:
            filename: Path to load file
        """
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Calibrator file not found: {filename}")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.calibration_params = data['calibration_params']
        self.statistics = data['statistics']
        self.training_date = data['training_date']
        self.training_samples = data['training_samples']
        self.is_trained = True
        
        if verbose >= 1:
            logger.info(f"✓ Calibrator loaded from {filename}")
            logger.info(f"  Method: Distribution matching")
    

    def save_statistics(self, filename='calibration_stats.json'):
        """
        Save calibration statistics to JSON file.
        
        Args:
            filename: Path to save file
        """
        
        if not self.is_trained:
            raise ValueError("No statistics to save - calibrator not trained!")
        
        output = {
            'method': 'distribution',
            'training_date': self.training_date,
            'training_samples': self.training_samples,
            'parameters': self.statistics
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Statistics saved to {filename}")
    

    def get_summary(self):
        """
        Get a summary of calibration performance.
        
        Returns:
            dict: Summary statistics for all parameters
        """
        
        if not self.is_trained:
            return {"error": "Calibrator not trained"}
        
        summary = {
            'method': 'distribution',
            'training_date': self.training_date,
            'training_samples': self.training_samples,
            'parameters': {}
        }
        
        for param, stats in self.statistics.items():
            summary['parameters'][param] = {
                'mean_shift': stats['mean_shift'],
                'scaling_factor': stats['scaling_factor'],
                'mae_improvement': f"{stats['mae_improvement_percent']:.1f}%",
                'bias_reduction': f"{stats['bias_original']:.3f} → {stats['bias_calibrated']:.3f}"
            }
        
        return summary
    

    def print_summary(self):
        """Print a formatted summary of calibration performance."""
        
        if not self.is_trained:
            print("Calibrator not trained!")
            return
        
        print("\n" + "=" * 70)
        print("CALIBRATION SUMMARY")
        print("=" * 70)
        print(f"Method: Distribution Matching")
        print(f"Training date: {self.training_date}")
        print(f"Training samples: {self.training_samples}")
        
        for param, stats in self.statistics.items():
            print(f"\n{param}:")
            print(f"  Distribution shift: μ_shift={stats['mean_shift']:.3f}, σ_scale={stats['scaling_factor']:.3f}")
            print(f"  MAE improvement: {stats['mae_improvement_percent']:+.1f}%")
            print(f"  RMSE improvement: {stats['rmse_improvement_percent']:+.1f}%")
            print(f"  Bias: {stats['bias_original']:.3f} → {stats['bias_calibrated']:.3f}")





#=====================================================================
# Suplementary Code
#=====================================================================
def load_data(filepath):
    """
    Load weather data from CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame: Loaded data
    """
    
    df = pd.read_csv(filepath)
    
    # Ensure we have the date column
    if 'date' not in df.columns and 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date.astype(str)
    
    return df


def main():
    """Example usage of the calibration system."""
    
    print("=" * 70)
    print("Weather Forecast Calibration System")
    print("Distribution Matching Method")
    print("=" * 70)
    
    '''
    # Train calibrator
    print("\n### Training Calibrator ###\n")
    
    try:
        # Load historical data
        forecast_data = load_data('dataset/historical_forecast.csv')
        actual_data = load_data('dataset/historical_actual.csv')
        
        # Train calibrator
        calibrator = ForecastCalibrator()
        stats = calibrator.train(forecast_data, actual_data)
        
        # Save for future use
        calibrator.save('calibrator.pkl')
        calibrator.save_statistics('calibration_stats.json')
        
        # Print summary
        calibrator.print_summary()
        
    except FileNotFoundError as e:
        print(f"\n⚠ Training data not found: {e}")
        print("\nTo train the calibrator, you need:")
        print("  1. historical_forecast.csv - Past forecast data")
        print("  2. historical_actual.csv - Actual measured data")
        print("\nBoth files must have matching 'date' and 'hour' columns.")
    '''
        
    # Example 2: Use trained calibrator 
    print("\n\n### Testing Calibration ###\n")
    
    try:
        # Load trained calibrator
        calibrator = ForecastCalibrator()
        calibrator.load('calibrator.pkl')
        
        # Load new forecast
        new_forecast = load_data('test/tomorrow_forecast.csv')
        
        # Apply calibration
        calibrated_forecast = calibrator.calibrate(new_forecast, verbose=2)
        
        # Save calibrated forecast
        calibrated_forecast.to_csv('test/calibrated_forecast.csv', index=False)
        print("\n✓ Calibrated forecast saved to 'calibrated_forecast.csv'")
        
        # Show summary
        print("\n" + "=" * 70)
        print("Calibration Summary")
        print("=" * 70)
        summary = calibrator.get_summary()
        print(json.dumps(summary, indent=2))
        
    except FileNotFoundError as e:
        print(f"\n⚠ Could not load calibrator or forecast: {e}")
        print("\nMake sure you have:")
        print("  1. Trained the calibrator first")
        print("  2. A forecast file to calibrate (tomorrow_forecast.csv)")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    


if __name__ == "__main__":
    # Check required packages    
    main()